# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Sequence
from contextlib import ExitStack
from datetime import datetime, timezone
from functools import partial
from typing import Any
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor as A2AAgentExecutorBase
from a2a.server.agent_execution import RequestContext
from a2a.server.events import Event, EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, Task, TaskState, TaskStatus

from ag2.agent import Agent
from ag2.context import ConversationContext, strip_reserved_variables
from ag2.events import (
    BaseEvent,
    ClientToolCallEvent,
    ModelMessageChunk,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolResultsEvent,
)
from ag2.middleware.base import MiddlewareFactory
from ag2.stream import MemoryStream
from ag2.tools.final.client_tool import ClientTool
from ag2.tools.final.function_tool import FunctionToolSchema
from ag2.tools.tool import Tool

from .events import A2AEvent, A2ATaskStatusUpdate
from .extension import CONTEXT_UPDATE_METADATA_KEY
from .mappers import (
    ParsedMessage,
    a2a_event_to_sdk,
    chunk_to_text_artifact,
    client_call_to_artifact,
    parse_message,
    task_state_to_status_update,
)

logger = logging.getLogger(__name__)

_TERMINAL_STATES = frozenset({
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_REJECTED,
})


class _ClosingTaskUpdater(TaskUpdater):
    """``TaskUpdater`` that sends ``last_chunk=True`` on the streamed text artifact before the task leaves ``working``.

    Leaving ``working`` ends the turn's output: ``closed`` tells the stream mapper to drop whatever
    the turn still produces, such as a chunk racing a cancel, rather than send it after the close.

    The turn's own completion and a cancel race on one updater, as the SDK does not order
    ``cancel`` against ``execute``. Whichever claims a terminal state first ends the task; any
    later transition is a no-op instead of the ``RuntimeError`` the base raises, which the SDK
    would turn into a FAILED written over the winner.

    Each transition that reaches the wire is mirrored onto ``lifecycle`` as an
    ``A2ATaskStatusUpdate`` in the same order. A separate dispatcher finishes each notification
    before starting the next, even if a subscriber requests another transition. Transitions
    requested during dispatch only enqueue their notifications, so subscribers can await them
    without waiting on their own notification. Cancelling the caller does not cancel dispatch,
    and a failing subscriber is logged rather than raised: the transition is already on the wire.
    """

    def __init__(
        self,
        event_queue: EventQueue,
        task_id: str,
        context_id: str,
        *,
        lifecycle: ConversationContext,
        text_artifact_id: str,
        text_pieces: list[str],
        pending_client_calls: list[ClientToolCallEvent],
    ) -> None:
        super().__init__(event_queue, task_id, context_id)
        self._lifecycle = lifecycle
        self._text_artifact_id = text_artifact_id
        self._text_pieces = text_pieces
        self._pending_client_calls = pending_client_calls
        self.closed = False
        self.ended = False
        # One transition at a time: a cancel landing while another transition closes the text
        # waits for it rather than ending the task under it.
        self._transition = asyncio.Lock()
        self._lifecycle_events: deque[A2ATaskStatusUpdate] = deque()
        self._lifecycle_dispatcher: asyncio.Task[None] | None = None
        # Stream subscriptions made through ``AgentExecutor._observe``. Held, not unwound, for the
        # updater's life: its dispatcher may still deliver a notification after ``execute`` returns.
        self.subscriptions = ExitStack()

    async def send_text_chunk(self, event: ModelMessageChunk) -> None:
        """Stream a text chunk while the turn is open."""
        await self._send_while_open(partial(self._write_text_chunk, event))

    async def send_client_call(self, event: ClientToolCallEvent) -> None:
        """Publish a client tool call while the turn is open."""
        await self._send_while_open(partial(self._write_client_call, event))

    async def send_event(self, event: Event) -> None:
        """Forward an event while the turn is open."""
        await self._send_while_open(partial(self.event_queue.enqueue_event, event))

    async def _send_while_open(self, write: Callable[[], Awaitable[None]]) -> None:
        commit = asyncio.create_task(self._commit_while_open(write))
        await self._wait_for_commit(commit)

    async def _commit_while_open(self, write: Callable[[], Awaitable[None]]) -> None:
        # Under the transition lock, so a write lands either before the closing transition or not at all.
        async with self._transition:
            if not self.closed:
                await write()

    async def _write_text_chunk(self, event: ModelMessageChunk) -> None:
        # a2a-sdk rejects append=True before the artifact exists, so the first chunk creates it.
        chunk = chunk_to_text_artifact(
            event,
            artifact_id=self._text_artifact_id,
            task_id=self.task_id,
            context_id=self.context_id,
            append=bool(self._text_pieces),
        )
        await self.event_queue.enqueue_event(a2a_event_to_sdk(chunk))
        self._text_pieces.append(event.content)

    async def _write_client_call(self, event: ClientToolCallEvent) -> None:
        artifact = client_call_to_artifact(event, task_id=self.task_id, context_id=self.context_id)
        await self.event_queue.enqueue_event(a2a_event_to_sdk(artifact))
        self._pending_client_calls.append(event)

    async def update_status(
        self,
        state: TaskState,
        message: Message | None = None,
        timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        commit = asyncio.create_task(self._commit_status(state, message, timestamp, metadata))
        await self._wait_for_commit(commit)
        dispatcher = commit.result()
        if dispatcher is not None:
            await asyncio.shield(dispatcher)

    async def _commit_status(
        self,
        state: TaskState,
        message: Message | None,
        timestamp: str | None,
        metadata: dict[str, Any] | None,
    ) -> asyncio.Task[None] | None:
        async with self._transition:
            if self.ended:
                return None
            if state in _TERMINAL_STATES:
                self.ended = True
            if state != TaskState.TASK_STATE_WORKING and not self.closed:
                # Set before the first await, so nothing the turn emits meanwhile lands after the close.
                self.closed = True
                if self._text_pieces:
                    await self.add_artifact(
                        parts=[Part(text="")],
                        artifact_id=self._text_artifact_id,
                        append=True,
                        last_chunk=True,
                    )
            await super().update_status(state, message, timestamp, metadata)
            self._lifecycle_events.append(
                task_state_to_status_update(
                    state,
                    task_id=self.task_id,
                    context_id=self.context_id,
                    message=message,
                    timestamp=datetime.now(tz=timezone.utc),
                ),
            )
            if self._lifecycle_dispatcher is None:
                dispatcher = self._lifecycle.spawn_background(self._dispatch_lifecycle())
                # An eager task factory may finish dispatch before spawn_background returns.
                self._lifecycle_dispatcher = None if dispatcher.done() else dispatcher
            else:
                # A subscriber may be awaiting this transition, directly or through another task.
                # Its dispatcher must finish the current event before delivering this one.
                dispatcher = None

        return dispatcher

    async def _wait_for_commit(self, commit: asyncio.Task[Any]) -> None:
        """Wait for ``commit`` to finish even if the caller is cancelled, then re-raise that cancellation.

        A queue may publish to one consumer before awaiting another. Finish its write and
        bookkeeping before cancellation can release the transition lock or remove this turn.
        """
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(commit)
                break
            except asyncio.CancelledError as exc:
                if commit.cancelled():
                    raise
                cancellation = exc
        if cancellation is not None:
            raise cancellation

    async def _dispatch_lifecycle(self) -> None:
        try:
            while self._lifecycle_events:
                event = self._lifecycle_events.popleft()
                try:
                    await self._lifecycle.send(event)
                except (Exception, asyncio.CancelledError):
                    # The transition is already on the wire. Raising would drop the notifications
                    # queued behind it, and an error out of ``execute`` or ``cancel`` makes the SDK
                    # write FAILED after the terminal state.
                    logger.exception("A2A lifecycle subscriber failed on %s", TaskState.Name(event.state))
        finally:
            self._lifecycle_dispatcher = None


class AgentExecutor(A2AAgentExecutorBase):
    """Bridge ``Agent.ask()`` <-> A2A task lifecycle (stateless flavor).

    Each ``execute()`` call is a self-contained turn: the executor pulls
    the AG2 conversation history, tool schemas, and any tool-call results
    from the incoming Message, rebuilds a fresh ``MemoryStream`` plus
    ``Context``, and dispatches into ``Agent._execute(initial_event,...)``.

    No per-task session memory survives between calls — clients are
    expected to send their full ``ag2.history+json`` payload on every
    request. This trades wire-size for horizontal scalability: any
    server replica can process any incoming request without sticky
    routing.

    Note: ``Agent._execute`` is private API. We use it directly because
    ``Agent.ask`` only accepts string/Input arguments and constructs a
    ``ModelRequest``; it cannot resume a turn from a ``ToolResultsEvent``.
    Substituting that with a public wrapper would require core changes,
    which the stateless refactor intentionally avoids.

    Streaming events flow: AG2 events on the per-turn ``MemoryStream``
    are forwarded into the A2A ``EventQueue`` by a single subscribe-style
    mapper (``map_ag2_to_a2a`` below). The mapper translates AG2 events
    into typed ``A2AEvent`` wrappers, then unwraps them to bare protobuf
    via ``a2a_event_to_sdk`` for the queue. Status-lifecycle transitions
    (start_work / complete / failed / requires_input / cancel) still go
    through ``TaskUpdater`` because it builds the boilerplate
    ``TaskStatus`` / ``Message`` objects that those transitions need.
    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        # Updaters of the tasks running on this executor, so ``cancel`` can close their text artifact.
        self._running: dict[str, _ClosingTaskUpdater] = {}

    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        msg = request_context.message
        if msg is None:
            return
        parsed = parse_message(msg)

        # ``msg.task_id`` is normally populated by the SDK 1.x server-side
        # request handler (it auto-generates an id when the client didn't
        # send one). The ``or request_context.task_id or uuid4()`` chain
        # is a defensive fallback for non-standard handlers and for tests
        # that bypass the request builder. Same idea for ``context_id``.
        task_id = msg.task_id or request_context.task_id or uuid4().hex
        context_id = msg.context_id or request_context.context_id or uuid4().hex
        # ``msg.task_id`` cannot be used as a first-turn signal because it
        # is set by the SDK on every incoming message. The authoritative
        # signal is ``request_context.current_task`` — ``None`` exactly
        # when no task has been persisted yet for this id.
        is_first_turn = request_context.current_task is None

        # Per-request stream lifted out of ``_run_one_turn`` so it spans
        # the whole lifecycle. The updater mirrors each transition into this
        # stream as a typed ``A2ATaskStatusUpdate`` for AG2-side observers
        # (``stream.where(A2ATaskStatusUpdate)``); the ``map_ag2_to_a2a``
        # subscriber filters those events out of the queue forward path so
        # ``TaskUpdater`` (with timestamp + terminal-state lock) stays the
        # single source of truth on the wire.
        stream = MemoryStream()
        if parsed.history_events:
            await stream.history.replace(parsed.history_events)

        # Lightweight context used only to publish lifecycle status events
        # to ``stream``; the real ``Context`` for agent execution is built
        # inside ``_dispatch_to_agent``. Only ``dependency_provider`` is
        # required so subscribers can resolve DI on these events.
        lifecycle_ctx = ConversationContext(stream, dependency_provider=self._agent.dependency_provider)

        text_artifact_id = uuid4().hex
        text_pieces: list[str] = []
        pending_client_calls: list[ClientToolCallEvent] = []
        updater = _ClosingTaskUpdater(
            event_queue,
            task_id,
            context_id,
            lifecycle=lifecycle_ctx,
            text_artifact_id=text_artifact_id,
            text_pieces=text_pieces,
            pending_client_calls=pending_client_calls,
        )

        @stream.subscribe
        async def map_ag2_to_a2a(event: BaseEvent) -> None:
            if updater.closed:
                return

            if isinstance(event, ModelMessageChunk):
                await updater.send_text_chunk(event)
                return

            if isinstance(event, ClientToolCallEvent):
                await updater.send_client_call(event)
                return

            # Lifecycle status updates are already on the wire via
            # ``TaskUpdater`` (with timestamp + lock) — drop the AG2-stream
            # mirror here to avoid duplicating events on the queue.
            if isinstance(event, A2ATaskStatusUpdate):
                return

            # User code may emit A2AEvent objects directly via
            # ``await ctx.send(A2A...())`` from inside a tool — pass
            # through unchanged so transports surface them as-is.
            if isinstance(event, A2AEvent):
                await updater.send_event(a2a_event_to_sdk(event))

        self._observe(updater.subscriptions, lifecycle_ctx)

        if is_first_turn:
            # SDK 1.x consumer requires a ``Task`` object on the event queue
            # before any ``TaskStatusUpdateEvent`` — TaskUpdater only emits
            # status events, so we enqueue the bootstrap Task ourselves.
            await event_queue.enqueue_event(
                Task(
                    id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                ),
            )
            await updater.start_work()

        self._running[task_id] = updater
        try:
            await self._run_one_turn(
                parsed,
                updater,
                stream,
                text_pieces,
                pending_client_calls,
                extra_prompt=self._extra_system_prompt(request_context),
            )
        except Exception:
            await updater.failed()
            raise
        finally:
            self._running.pop(task_id, None)

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        # Per a2a-sdk contract: the framework awaits this hook first and only
        # afterwards cancels the asyncio Task running ``execute()`` (raising
        # ``asyncio.CancelledError``). Our only job is to publish the
        # running turn's CANCELED status update — the coroutine winds down
        # on its own. The ``except Exception`` in ``execute`` deliberately
        # does NOT catch ``CancelledError`` (it's a ``BaseException``),
        # so a cancel never flips the task to FAILED by accident.
        # The running turn's updater closes the streamed text before the task leaves ``working``,
        # and yields to the turn if it already ended the task.
        if updater := self._running.get(request_context.task_id or ""):
            await updater.cancel()
        # No turn running: the task waits for input, or its turn just ended and the SDK has not yet
        # consumed that terminal event. Writing CANCELED here would land over the latter; the SDK
        # closes a task still open once ``cancel`` returns, after re-reading it, so leave it to that.

    def _observe(self, stack: ExitStack, context: ConversationContext) -> None:
        """Subscribe to this request's stream before its first status update.

        Extension point: ``execute`` calls this once per request, so a subclass sees every
        ``A2ATaskStatusUpdate`` the request puts on the wire, unlike agent observers, which only
        run during the turn. Enter subscriptions on ``stack``; it lives as long as the request can
        still deliver a notification. The base subscribes nothing.
        """

    def _extra_system_prompt(self, request_context: RequestContext) -> Sequence[str]:
        """Per-request system-prompt fragments to fold into the turn.

        Extension point: ``execute`` calls this once per request and threads the
        result through ``_run_one_turn`` into the agent's prompt, so a subclass
        can inject request-derived instructions without reimplementing the
        shared turn machinery. ``request_context`` is per-request, so this is
        safe to read on a shared executor instance. The base returns nothing.
        """
        return ()

    async def _run_one_turn(
        self,
        parsed: ParsedMessage,
        updater: TaskUpdater,
        stream: MemoryStream,
        text_pieces: list[str],
        pending_client_calls: list[ClientToolCallEvent],
        extra_prompt: Sequence[str] = (),
    ) -> None:
        client_tools = [self._make_client_tool(s) for s in parsed.tool_schemas]
        initial_event = self._build_initial_event(parsed)

        response, final_variables = await self._dispatch_to_agent(
            initial_event,
            stream,
            client_tools,
            incoming_variables=parsed.context_update,
            extra_prompt=extra_prompt,
        )

        has_pending = bool(response.tool_calls and response.tool_calls.calls and response.response_force)
        if has_pending or pending_client_calls:
            await updater.requires_input()
            return

        # Avoid duplicating text that was already streamed via chunk
        # artifacts. ``response.message.content`` echoes the full assistant
        # text, but if streaming was active we already pushed it as
        # ``add_artifact(append=True)`` calls — sending it again on
        # ``complete(message=...)`` makes the client see it twice.
        streamed_text = "".join(text_pieces)
        full_text = response.message.content if response.message else streamed_text
        if full_text and full_text != streamed_text:
            agent_msg = self._build_final_message(updater, full_text, final_variables)
        elif final_variables:
            agent_msg = self._build_final_message(updater, "", final_variables)
        else:
            agent_msg = None
        await updater.complete(message=agent_msg)

    @staticmethod
    def _build_final_message(
        updater: TaskUpdater,
        final_text: str,
        final_variables: dict[str, Any],
    ) -> "Any | None":
        outgoing = strip_reserved_variables(final_variables, source="an outgoing A2A response", warn=False)
        if not final_text and not outgoing:
            return None
        metadata: dict[str, Any] | None = None
        if outgoing:
            metadata = {CONTEXT_UPDATE_METADATA_KEY: outgoing}
        parts = [Part(text=final_text)] if final_text else []
        return updater.new_agent_message(parts=parts, metadata=metadata)

    @staticmethod
    def _build_initial_event(parsed: ParsedMessage) -> BaseEvent:
        # Continuation turn: the client returned tool results for the
        # tool-calls the server emitted last turn. ``payload_to_results``
        # already yields ``ToolErrorEvent`` for failed calls so error
        # branches reach the LLM intact (provider mappers set is_error).
        if parsed.tool_results:
            return ToolResultsEvent(parsed.tool_results)

        inputs = parsed.inputs or [TextInput("")]
        return ModelRequest(list(inputs))

    @staticmethod
    def _make_client_tool(schema: FunctionToolSchema) -> ClientTool:
        return ClientTool({
            "function": {
                "name": schema.function.name,
                "description": schema.function.description,
                "parameters": schema.function.parameters,
            }
        })

    async def _dispatch_to_agent(
        self,
        initial_event: BaseEvent,
        stream: MemoryStream,
        client_tools: Sequence[Tool],
        *,
        incoming_variables: dict[str, Any],
        extra_prompt: Sequence[str] = (),
        additional_middleware: Iterable[MiddlewareFactory] = (),
    ) -> tuple[ModelResponse, dict[str, Any]]:
        agent = self._agent
        if agent.config is None:
            raise RuntimeError("Agent.config is not set; cannot serve via A2A")
        client = agent.config.create()

        merged_variables = {
            **dict(agent._agent_variables),
            **strip_reserved_variables(incoming_variables, source="an inbound A2A request"),
        }
        ctx = ConversationContext(
            stream,
            prompt=[*agent._system_prompt, *extra_prompt],
            dependencies=dict(agent._agent_dependencies),
            variables=merged_variables,
            dependency_provider=agent.dependency_provider,
        )

        # ``_execute`` is private but is the only entry point that accepts
        # a non-``ModelRequest`` initial event (``ToolResultsEvent`` for
        # continuation turns). See the class-level docstring for context.
        # ``additional_middleware`` lets a subclass (e.g. the A2UI executor)
        # inject per-turn middleware onto a plain agent without baking it in.
        reply = await agent._execute(
            initial_event,
            context=ctx,
            client=client,
            additional_tools=client_tools,
            additional_middleware=additional_middleware,
        )
        return reply.response, dict(ctx.variables)
