# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor as A2AAgentExecutorBase
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, Task, TaskState, TaskStatus

from autogen.beta.agent import Agent
from autogen.beta.context import ConversationContext
from autogen.beta.events import (
    BaseEvent,
    ClientToolCallEvent,
    ModelMessageChunk,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolResultsEvent,
)
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.final.client_tool import ClientTool
from autogen.beta.tools.final.function_tool import FunctionToolSchema

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

        updater = TaskUpdater(event_queue, task_id, context_id)

        # Per-request stream lifted out of ``_run_one_turn`` so it spans
        # the whole lifecycle. Each ``TaskUpdater`` transition is mirrored
        # into this stream as a typed ``A2ATaskStatusUpdate`` for AG2-side
        # observers (``stream.where(A2ATaskStatusUpdate)``); the ``map_ag2_to_a2a``
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

        @stream.subscribe
        async def map_ag2_to_a2a(event: BaseEvent) -> None:
            if isinstance(event, ModelMessageChunk):
                text_pieces.append(event.content)
                a2a_event = chunk_to_text_artifact(
                    event,
                    artifact_id=text_artifact_id,
                    task_id=task_id,
                    context_id=context_id,
                )
                await event_queue.enqueue_event(a2a_event_to_sdk(a2a_event))
                return

            if isinstance(event, ClientToolCallEvent):
                pending_client_calls.append(event)
                a2a_event = client_call_to_artifact(
                    event,
                    task_id=task_id,
                    context_id=context_id,
                )
                await event_queue.enqueue_event(a2a_event_to_sdk(a2a_event))
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
                await event_queue.enqueue_event(a2a_event_to_sdk(event))

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
            await lifecycle_ctx.send(
                task_state_to_status_update(
                    TaskState.TASK_STATE_WORKING,
                    task_id=task_id,
                    context_id=context_id,
                    timestamp=datetime.now(tz=timezone.utc),
                ),
            )

        try:
            await self._run_one_turn(
                parsed,
                updater,
                stream,
                lifecycle_ctx,
                text_pieces,
                pending_client_calls,
                task_id,
                context_id,
            )
        except Exception:
            await updater.failed()
            await lifecycle_ctx.send(
                task_state_to_status_update(
                    TaskState.TASK_STATE_FAILED,
                    task_id=task_id,
                    context_id=context_id,
                    timestamp=datetime.now(tz=timezone.utc),
                ),
            )
            raise

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        # Per a2a-sdk contract: the framework already cancels the asyncio
        # Task running ``execute()`` (raising ``asyncio.CancelledError``)
        # before this hook is invoked. Our only job is to publish the
        # CANCELED status update — the running coroutine winds down on
        # its own. The ``except Exception`` in ``execute`` deliberately
        # does NOT catch ``CancelledError`` (it's a ``BaseException``),
        # so a cancel never flips the task to FAILED by accident.
        task_id = request_context.task_id or ""
        context_id = request_context.context_id or ""
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.cancel()

    async def _run_one_turn(
        self,
        parsed: ParsedMessage,
        updater: TaskUpdater,
        stream: MemoryStream,
        lifecycle_ctx: ConversationContext,
        text_pieces: list[str],
        pending_client_calls: list[ClientToolCallEvent],
        task_id: str,
        context_id: str,
    ) -> None:
        client_tools = [self._make_client_tool(s) for s in parsed.tool_schemas]
        initial_event = self._build_initial_event(parsed)

        response, final_variables = await self._dispatch_to_agent(
            initial_event,
            stream,
            client_tools,
            incoming_variables=parsed.context_update,
        )

        has_pending = bool(response.tool_calls and response.tool_calls.calls and response.response_force)
        if has_pending or pending_client_calls:
            await updater.requires_input()
            await lifecycle_ctx.send(
                task_state_to_status_update(
                    TaskState.TASK_STATE_INPUT_REQUIRED,
                    task_id=task_id,
                    context_id=context_id,
                    timestamp=datetime.now(tz=timezone.utc),
                ),
            )
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
        await lifecycle_ctx.send(
            task_state_to_status_update(
                TaskState.TASK_STATE_COMPLETED,
                task_id=task_id,
                context_id=context_id,
                message=agent_msg,
                timestamp=datetime.now(tz=timezone.utc),
            ),
        )

    @staticmethod
    def _build_final_message(
        updater: TaskUpdater,
        final_text: str,
        final_variables: dict[str, Any],
    ) -> "Any | None":
        if not final_text and not final_variables:
            return None
        metadata: dict[str, Any] | None = None
        if final_variables:
            metadata = {CONTEXT_UPDATE_METADATA_KEY: final_variables}
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
        client_tools: list[ClientTool],
        *,
        incoming_variables: dict[str, Any],
    ) -> tuple[ModelResponse, dict[str, Any]]:
        agent = self._agent
        if agent.config is None:
            raise RuntimeError("Agent.config is not set; cannot serve via A2A")
        client = agent.config.create()

        merged_variables = {**dict(agent._agent_variables), **incoming_variables}
        ctx = ConversationContext(
            stream,
            prompt=list(agent._system_prompt),
            dependencies=dict(agent._agent_dependencies),
            variables=merged_variables,
            dependency_provider=agent.dependency_provider,
        )

        # ``_execute`` is private but is the only entry point that accepts
        # a non-``ModelRequest`` initial event (``ToolResultsEvent`` for
        # continuation turns). See the class-level docstring for context.
        reply = await agent._execute(
            initial_event,
            context=ctx,
            client=client,
            additional_tools=client_tools,
        )
        return reply.response, dict(ctx.variables)
