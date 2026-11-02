# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeAlias
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

from ._states import TERMINAL_STATES
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


@dataclass(slots=True)
class _Chunk:
    event: ModelMessageChunk


@dataclass(slots=True)
class _ClientCall:
    event: ClientToolCallEvent


@dataclass(slots=True)
class _PassThrough:
    event: Event


@dataclass(slots=True)
class _Transition:
    state: TaskState
    message: Message | None = None


@dataclass(slots=True)
class _Barrier:
    """Written once every command posted before it is."""


_Command: TypeAlias = _Chunk | _ClientCall | _PassThrough | _Transition | _Barrier


class _TaskWriter:
    """The single writer of one request's task: every event the request puts on the wire goes through it.

    Text chunks, client calls, pass-through events and status transitions are commands on one
    queue, written one at a time by one writer task, so the queue order is the wire order and no
    lock is needed. The turn's own transitions and ``cancel`` both arrive as commands, since the
    SDK does not order ``cancel`` against ``execute``. Whichever terminal state is written first
    ends the task. A later transition is a no-op instead of the ``RuntimeError`` a ``TaskUpdater``
    raises, which the SDK would turn into a FAILED written over the winner (ADR 0029). A poster
    waits until its command is written, even when it is cancelled meanwhile, because the SDK
    closes the event queue once ``execute`` returns. Once the writer has stopped, by shutdown or
    by cancellation, commands still queued and later ones are dropped, so no poster waits for it.

    The first transition away from ``working`` closes the turn's output: it sends ``last_chunk``
    on the streamed text artifact, and anything the turn still produces is dropped, so nothing
    lands after the close.

    Each written transition is mirrored onto ``lifecycle`` as an ``A2ATaskStatusUpdate`` by a
    separate notifier, in wire order. Nothing waits for the notifier while the request runs: a
    transition resolves once it is on the wire, so a slow subscriber delays neither the turn nor a
    cancel, and a subscriber that awaits a transition, such as a cancel on ``INPUT_REQUIRED``,
    cannot block it. A failing subscriber is logged rather than raised, because the transition
    is already on the wire, and so is one that is cancelled; cancelling the notifier itself stops
    delivery. ``shut_down`` delivers every notification, then closes ``subscriptions``. A command
    a subscriber posts without awaiting it, after the last notification, is dropped.
    """

    def __init__(
        self,
        event_queue: EventQueue,
        task_id: str,
        context_id: str,
        *,
        lifecycle: ConversationContext,
        text_artifact_id: str,
    ) -> None:
        self._updater = TaskUpdater(event_queue, task_id, context_id)
        self._lifecycle = lifecycle
        self._text_artifact_id = text_artifact_id
        # Text of every chunk written in the turn, and of those the current model call wrote.
        self._text_pieces: list[str] = []
        self._call_text: list[str] = []
        # Text the turn's last finished model call streamed.
        self.answer_text = ""
        self.pending_client_calls: list[ClientToolCallEvent] = []
        self.subscriptions = ExitStack()
        # The turn left ``working``: its remaining output is dropped.
        self._output_closed = False
        self._terminal_written = False
        self._commands: asyncio.Queue[tuple[_Command, asyncio.Future[None]] | None] = asyncio.Queue()
        self._notifications: asyncio.Queue[A2ATaskStatusUpdate | None] = asyncio.Queue()
        self._writer: asyncio.Task[None] | None = None
        self._notifier: asyncio.Task[None] | None = None
        self._shutdown: asyncio.Task[None] | None = None
        # The writer is stopping: a later transition is a no-op and later output is dropped.
        self._stopped = False

    def new_agent_message(self, parts: list[Part], metadata: dict[str, Any] | None = None) -> Message:
        return self._updater.new_agent_message(parts=parts, metadata=metadata)

    async def forward(self, event: BaseEvent) -> None:
        """Put a turn's AG2 event on the wire; subscribed to the request's stream."""
        if isinstance(event, ModelMessageChunk):
            await self._post(_Chunk(event))
        elif isinstance(event, ModelResponse):
            # Chunks precede the response of the call that streamed them.
            self.answer_text = "".join(self._call_text)
            self._call_text = []
        elif isinstance(event, ClientToolCallEvent):
            await self._post(_ClientCall(event))
        elif isinstance(event, A2ATaskStatusUpdate):
            # The notifier's own mirror of a transition that is already on the wire.
            return
        elif isinstance(event, A2AEvent):
            # User code may emit A2AEvent objects directly via ``await ctx.send(A2A...())`` from
            # inside a tool; pass them through unchanged so transports surface them as-is.
            await self._post(_PassThrough(a2a_event_to_sdk(event)))

    async def add_text(self, text: str) -> None:
        """Append text to the turn's text artifact, as a streamed chunk would."""
        await self._post(_Chunk(ModelMessageChunk(text)))

    async def start_work(self) -> None:
        await self._post(_Transition(TaskState.TASK_STATE_WORKING))

    async def requires_input(self, message: Message | None = None) -> None:
        await self._post(_Transition(TaskState.TASK_STATE_INPUT_REQUIRED, message))

    async def complete(self, message: Message | None = None) -> None:
        await self._post(_Transition(TaskState.TASK_STATE_COMPLETED, message))

    async def failed(self, message: Message | None = None) -> None:
        await self._post(_Transition(TaskState.TASK_STATE_FAILED, message))

    async def cancel(self) -> None:
        await self._post(_Transition(TaskState.TASK_STATE_CANCELED))

    def shut_down(self) -> "asyncio.Task[None]":
        """Finish the request in the background: flush the writes, deliver every notification, close ``subscriptions``.

        Commands posted while notifications are still being delivered, such as a subscriber's
        cancel, are written. Once delivery is done, the writer stops.
        """
        if self._shutdown is None:
            self._shutdown = self._lifecycle.spawn_background(self._shut_down())
        return self._shutdown

    async def _post(self, command: _Command) -> None:
        if self._stopped:
            return
        if self._writer is None:
            self._writer = self._lifecycle.spawn_background(self._write_commands())
            self._writer.add_done_callback(self._drop_queued)
            self._notifier = self._lifecycle.spawn_background(self._notify_subscribers())
        done: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._commands.put_nowait((command, done))
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(done)
                break
            except asyncio.CancelledError as exc:
                if done.cancelled():
                    raise
                cancellation = exc
        if cancellation is not None:
            raise cancellation

    async def _write_commands(self) -> None:
        while (item := await self._commands.get()) is not None:
            command, done = item
            try:
                await self._write(command)
            except asyncio.CancelledError:
                done.cancel()
                raise
            except Exception as exc:
                done.set_exception(exc)
            else:
                done.set_result(None)

    def _drop_queued(self, writer: "asyncio.Task[None]") -> None:
        """Nothing is written once the writer ends, by shutdown or by cancellation, even before it started.

        Commands still queued and later posts are dropped instead of waiting for a writer that is gone.
        """
        self._stopped = True
        while not self._commands.empty():
            if (item := self._commands.get_nowait()) is not None:
                _, done = item
                if not done.done():
                    done.set_result(None)

    async def _write(self, command: _Command) -> None:
        if isinstance(command, _Transition):
            await self._write_transition(command)
        elif isinstance(command, _Barrier) or self._output_closed:
            return
        elif isinstance(command, _Chunk):
            # a2a-sdk rejects append=True before the artifact exists, so the first chunk creates it.
            chunk = chunk_to_text_artifact(
                command.event,
                artifact_id=self._text_artifact_id,
                task_id=self._updater.task_id,
                context_id=self._updater.context_id,
                append=bool(self._text_pieces),
            )
            await self._updater.event_queue.enqueue_event(a2a_event_to_sdk(chunk))
            self._text_pieces.append(command.event.content)
            self._call_text.append(command.event.content)
        elif isinstance(command, _ClientCall):
            artifact = client_call_to_artifact(
                command.event, task_id=self._updater.task_id, context_id=self._updater.context_id
            )
            await self._updater.event_queue.enqueue_event(a2a_event_to_sdk(artifact))
            self.pending_client_calls.append(command.event)
        else:
            await self._updater.event_queue.enqueue_event(command.event)

    async def _write_transition(self, transition: _Transition) -> None:
        if self._terminal_written:
            return
        state = transition.state
        # A write that raises counts as made, as ``TaskUpdater`` counts it, since it may have reached
        # some consumers: the close is not sent twice, nor a terminal state after one. A failed close
        # leaves the status unwritten, so the next transition, such as ``failed``, still ends the task.
        if state != TaskState.TASK_STATE_WORKING and not self._output_closed:
            self._output_closed = True
            if self._text_pieces:
                await self._updater.add_artifact(
                    parts=[Part(text="")],
                    artifact_id=self._text_artifact_id,
                    append=True,
                    last_chunk=True,
                )
        if state in TERMINAL_STATES:
            self._terminal_written = True
        await self._updater.update_status(state, transition.message)
        self._notifications.put_nowait(
            task_state_to_status_update(
                state,
                task_id=self._updater.task_id,
                context_id=self._updater.context_id,
                message=transition.message,
                timestamp=datetime.now(tz=timezone.utc),
            )
        )

    async def _notify_subscribers(self) -> None:
        while (update := await self._notifications.get()) is not None:
            # Delivered in a task of its own, so a subscriber's cancellation is told apart from the
            # notifier's own, such as at loop shutdown, which stops delivery.
            delivery = asyncio.ensure_future(self._lifecycle.send(update))
            try:
                await asyncio.wait({delivery})
            except asyncio.CancelledError:
                delivery.cancel()
                self._notifications.task_done()
                raise
            try:
                delivery.result()
            except (Exception, asyncio.CancelledError):
                # The transition is already on the wire. Raising would drop the notifications queued
                # behind it, and an error out of ``execute`` or ``cancel`` makes the SDK write FAILED
                # after the terminal state.
                logger.exception("A2A lifecycle subscriber failed on %s", TaskState.Name(update.state))
            finally:
                self._notifications.task_done()

    async def _shut_down(self) -> None:
        try:
            if self._writer is not None and self._notifier is not None:
                await self._post(_Barrier())
                # A subscriber may post a transition while it is notified; that is written, and
                # notified, before ``join`` returns.
                await self._notifications.join()
                self._stopped = True
                self._commands.put_nowait(None)
                # Waited for without raising: either may already have ended by cancellation.
                await asyncio.wait({self._writer})
                self._notifications.put_nowait(None)
                await asyncio.wait({self._notifier})
        finally:
            try:
                self.subscriptions.close()
            except (Exception, asyncio.CancelledError):
                # Cleanup is observer code too: it must not overwrite an already committed terminal state.
                logger.exception("A2A lifecycle subscriber cleanup failed")


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

    Streaming events flow: AG2 events on the per-request ``MemoryStream``
    are forwarded into the A2A ``EventQueue`` by the request's
    ``_TaskWriter`` (``forward``), which translates them into typed
    ``A2AEvent`` wrappers and unwraps them to bare protobuf via
    ``a2a_event_to_sdk``. Status-lifecycle transitions (start_work /
    complete / failed / requires_input / cancel) go through the same
    writer, so the wire carries everything in the order it was produced.
    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        # Writers of the tasks running on this executor, so ``cancel`` can close their text artifact.
        self._running: dict[str, _TaskWriter] = {}

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
        # the whole lifecycle. The writer mirrors each transition into this
        # stream as a typed ``A2ATaskStatusUpdate`` for AG2-side observers
        # (``stream.where(A2ATaskStatusUpdate)``) and leaves those mirrors
        # off the queue, so its own status writes stay the single source of
        # truth on the wire.
        stream = MemoryStream()
        if parsed.history_events:
            await stream.history.replace(parsed.history_events)

        # Lightweight context used only to publish lifecycle status events
        # to ``stream``; the real ``Context`` for agent execution is built
        # inside ``_dispatch_to_agent``. Only ``dependency_provider`` is
        # required so subscribers can resolve DI on these events.
        lifecycle_ctx = ConversationContext(stream, dependency_provider=self._agent.dependency_provider)

        writer = _TaskWriter(
            event_queue,
            task_id,
            context_id,
            lifecycle=lifecycle_ctx,
            text_artifact_id=uuid4().hex,
        )
        stream.subscribe(writer.forward)

        cancelled = False
        try:
            self._observe(writer.subscriptions, lifecycle_ctx)

            if is_first_turn:
                # SDK 1.x consumer requires a ``Task`` object on the event queue
                # before any ``TaskStatusUpdateEvent`` — the writer only emits
                # status events, so we enqueue the bootstrap Task ourselves.
                await event_queue.enqueue_event(
                    Task(
                        id=task_id,
                        context_id=context_id,
                        status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                    ),
                )
            # A continuation is worked on as much as a first turn (spec §4.1.3): it must not read as
            # still waiting for input while it runs.
            await writer.start_work()

            self._running[task_id] = writer
            try:
                await self._run_one_turn(
                    parsed,
                    writer,
                    stream,
                    extra_prompt=self._extra_system_prompt(request_context),
                )
            except Exception:
                await writer.failed()
                raise
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            shutdown = writer.shut_down()
            try:
                # A subscriber may still cancel the task while it is notified, so stay cancelable
                # until delivery is done. A cancelled request leaves that to the background.
                if not cancelled:
                    await asyncio.shield(shutdown)
            finally:
                if self._running.get(task_id) is writer:
                    self._running.pop(task_id)

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        # Per a2a-sdk contract: the framework awaits this hook first and only
        # afterwards cancels the asyncio Task running ``execute()`` (raising
        # ``asyncio.CancelledError``). Our only job is to publish the
        # running turn's CANCELED status update — the coroutine winds down
        # on its own. The ``except Exception`` in ``execute`` deliberately
        # does NOT catch ``CancelledError`` (it's a ``BaseException``),
        # so a cancel never flips the task to FAILED by accident.
        # The running turn's writer closes the streamed text before the task leaves ``working``,
        # and yields to the turn if it already ended the task.
        if writer := self._running.get(request_context.task_id or ""):
            await writer.cancel()
        # A request counts as running until its notifications are delivered, unless ``execute`` is
        # cancelled, so a cancel after its ``INPUT_REQUIRED`` is still written above (ADR 0029).
        # No request running: the task waits for input, or its turn just ended and the SDK has not yet
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
        writer: _TaskWriter,
        stream: MemoryStream,
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
        if has_pending or writer.pending_client_calls:
            await writer.requires_input()
            return

        # The final answer is the last model call's text. Results are artifacts, not messages (spec
        # §3.7), so text that call did not stream goes into the turn's text artifact before it closes.
        # Text it streamed is there already. Earlier calls of the turn (before a server-side tool)
        # streamed text of their own, so compare with the last call only.
        final_text = response.message.content if response.message else ""
        if final_text and final_text != writer.answer_text:
            await writer.add_text(final_text)
        await writer.complete(message=self._build_final_message(writer, final_variables))

    @staticmethod
    def _build_final_message(writer: _TaskWriter, final_variables: dict[str, Any]) -> Message | None:
        """The completion's message: it carries only the turn's context update, the answer is an artifact."""
        outgoing = strip_reserved_variables(final_variables, source="an outgoing A2A response", warn=False)
        if not outgoing:
            return None
        return writer.new_agent_message(parts=[], metadata={CONTEXT_UPDATE_METADATA_KEY: outgoing})

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
