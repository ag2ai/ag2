# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack
from typing import Any
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor as A2AAgentExecutorBase
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, Task, TaskState, TaskStatus

from autogen.beta.agent import Agent
from autogen.beta.annotations import Context
from autogen.beta.events import (
    BaseEvent,
    ClientToolCallEvent,
    ModelMessageChunk,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
    ToolResultsEvent,
)
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.final.client_tool import ClientTool
from autogen.beta.tools.final.function_tool import FunctionToolSchema

from .extension import CONTEXT_UPDATE_METADATA_KEY, MIME_TOOL_CALL
from .mappers.messages import ParsedMessage, parse_message
from .mappers.parts import data_part
from .mappers.tools import call_to_payload


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
    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        msg = request_context.message
        if msg is None:
            return
        parsed = parse_message(msg)

        task_id = msg.task_id or request_context.task_id or uuid4().hex
        context_id = msg.context_id or request_context.context_id or uuid4().hex
        # SDK 1.x always populates ``msg.task_id`` (it generates one if the
        # client didn't supply it), so ``msg.task_id`` is unreliable as a
        # first-turn signal. ``request_context.current_task`` is ``None``
        # exactly when no task has been persisted yet for this id.
        is_first_turn = request_context.current_task is None

        updater = TaskUpdater(event_queue, task_id, context_id)
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

        try:
            await self._run_one_turn(parsed, updater)
        except Exception:
            await updater.failed()
            raise

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        task_id = request_context.task_id or ""
        context_id = request_context.context_id or ""
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.cancel()

    async def _run_one_turn(
        self,
        parsed: ParsedMessage,
        updater: TaskUpdater,
    ) -> None:
        stream = MemoryStream()
        if parsed.history_events:
            await stream.history.replace(parsed.history_events)

        client_tools = [self._make_client_tool(s) for s in parsed.tool_schemas]
        initial_event = self._build_initial_event(parsed)

        text_artifact_id = uuid4().hex
        text_pieces: list[str] = []
        pending_client_calls: list[ClientToolCallEvent] = []

        with ExitStack() as stack:
            stack.enter_context(
                stream.where(ModelMessageChunk).sub_scope(
                    _make_chunk_handler(updater, text_artifact_id, text_pieces),
                ),
            )
            stack.enter_context(
                stream.where(ClientToolCallEvent).sub_scope(
                    _make_client_tool_call_handler(updater, pending_client_calls),
                ),
            )
            response, final_variables = await self._dispatch_to_agent(
                initial_event,
                stream,
                client_tools,
                incoming_variables=parsed.context_update,
            )

        has_pending = bool(response.tool_calls and response.tool_calls.calls and response.response_force)
        if has_pending or pending_client_calls:
            await updater.requires_input()
            return

        final_text = response.message.content if response.message else "".join(text_pieces)
        agent_msg = self._build_final_message(updater, final_text, final_variables)
        await updater.complete(message=agent_msg)

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
        # tool-calls the server emitted last turn. Rebuild a
        # ``ToolResultsEvent`` directly from the wire payload — every
        # field we need (id, name, content, error) is included.
        if parsed.tool_results:
            events = [
                ToolResultEvent(
                    parent_id=str(r.get("id", "")),
                    name=r.get("name"),
                    result=ToolResult(str(r.get("content", "") or "")),
                )
                for r in parsed.tool_results
            ]
            return ToolResultsEvent(events)

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
        ctx = Context(
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


def _make_chunk_handler(
    updater: TaskUpdater,
    text_artifact_id: str,
    text_pieces: list[str],
):
    async def handler(ev: ModelMessageChunk, _: Context) -> None:
        text_pieces.append(ev.content)
        await updater.add_artifact(
            parts=[Part(text=ev.content)],
            artifact_id=text_artifact_id,
            append=True,
        )

    return handler


def _make_client_tool_call_handler(
    updater: TaskUpdater,
    pending: list[ClientToolCallEvent],
):
    async def handler(ev: ClientToolCallEvent, _: Context) -> None:
        pending.append(ev)
        payload = call_to_payload(ToolCallEvent(id=ev.id, name=ev.name, arguments=ev.arguments))
        await updater.add_artifact(
            parts=[data_part(payload, media_type=MIME_TOOL_CALL)],
            artifact_id=ev.id,
            last_chunk=True,
        )

    return handler
