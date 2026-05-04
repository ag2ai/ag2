# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor as A2AAgentExecutorBase
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part

from autogen.beta.annotations import Context as ContextWithInput
from autogen.beta.events import (
    ClientToolCallEvent,
    ModelMessageChunk,
    ModelRequest,
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

if TYPE_CHECKING:
    from autogen.beta.agent import Agent
    from autogen.beta.events import BaseEvent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Session:
    """Per-task state on the server side.

    Lives across multiple ``execute()`` calls for the same ``task_id`` —
    used to thread client-side tool round-trips through a single
    conversation.
    """

    task_id: str
    context_id: str
    stream: MemoryStream
    client_tool_schemas: list[FunctionToolSchema] = field(default_factory=list)
    pending_calls: dict[str, ClientToolCallEvent] = field(default_factory=dict)


class AgentExecutor(A2AAgentExecutorBase):
    """Bridge ``Agent.ask()`` <-> A2A task lifecycle.

    Each AG2 turn runs to completion inside a single ``execute()`` call.
    If the model invokes a ``ClientTool`` mid-turn, ``Agent.ask()``
    returns immediately with ``response_force=True`` and the executor
    emits ``tool-call+json`` artifacts, transitions the Task to
    ``input_required`` and yields control. The next ``execute()`` for
    the same ``task_id`` brings in tool-result data, which is fed back
    via ``Agent._execute(ToolResultsEvent(...))`` to continue the turn.
    """

    def __init__(self, agent: "Agent") -> None:
        self._agent = agent
        self._sessions: dict[str, _Session] = {}

    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        request = request_context.request
        if request is None or request.message is None:
            return
        msg = request.message
        parsed = parse_message(msg)

        task_id = msg.task_id or request_context.task_id or uuid4().hex
        context_id = msg.context_id or request_context.context_id or uuid4().hex
        is_new = task_id not in self._sessions

        session = self._sessions.get(task_id)
        if session is None:
            session = _Session(
                task_id=task_id,
                context_id=context_id,
                stream=MemoryStream(),
                client_tool_schemas=list(parsed.tool_schemas),
            )
            self._sessions[task_id] = session

        updater = TaskUpdater(event_queue, task_id, context_id)
        if is_new:
            await updater.submit()
            await updater.start_work()

        try:
            await self._run_one_turn(session, parsed, updater)
        except Exception:
            self._sessions.pop(task_id, None)
            logger.exception("A2A AgentExecutor failed for task=%s", task_id)
            await updater.failed()
            return

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        task_id = request_context.task_id or ""
        context_id = request_context.context_id or ""
        self._sessions.pop(task_id, None)
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.cancel()

    async def _run_one_turn(
        self,
        session: _Session,
        parsed: ParsedMessage,
        updater: TaskUpdater,
    ) -> None:
        initial_event = self._build_initial_event(session, parsed)
        client_tools = [self._make_client_tool(s) for s in session.client_tool_schemas]

        text_artifact_id = uuid4().hex
        text_pieces: list[str] = []

        with ExitStack() as stack:
            stack.enter_context(
                session.stream.where(ModelMessageChunk).sub_scope(
                    _make_chunk_handler(updater, text_artifact_id, text_pieces),
                ),
            )
            stack.enter_context(
                session.stream.where(ClientToolCallEvent).sub_scope(
                    _make_client_tool_call_handler(updater, session),
                ),
            )
            response, final_variables = await self._dispatch_to_agent(
                initial_event,
                session,
                client_tools,
                incoming_variables=parsed.context_update,
            )

        has_pending = bool(response.tool_calls and response.tool_calls.calls and response.response_force)
        if has_pending:
            await updater.requires_input()
            return

        final_text = response.message.content if response.message else "".join(text_pieces)
        agent_msg = self._build_final_message(updater, final_text, final_variables)
        await updater.complete(message=agent_msg)
        self._sessions.pop(session.task_id, None)

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
    def _build_initial_event(session: _Session, parsed: ParsedMessage) -> "BaseEvent":
        if parsed.tool_results and session.pending_calls:
            events = []
            for r in parsed.tool_results:
                call = session.pending_calls.pop(r["id"], None)
                if call is None:
                    continue
                events.append(
                    ToolResultEvent(
                        parent_id=call.id,
                        name=call.name,
                        result=ToolResult(r["content"]),
                    )
                )
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
        initial_event: "BaseEvent",
        session: _Session,
        client_tools: list[ClientTool],
        *,
        incoming_variables: dict[str, Any],
    ) -> tuple[object, dict[str, Any]]:
        # We use the (private) ``Agent._execute`` to start the turn from a
        # ``ToolResultsEvent`` when continuing a client-side tool round-trip.
        # ``Agent.ask()`` only accepts ``ModelRequest`` initial events, so for
        # continuation turns we have to bypass that wrapper.
        agent = self._agent
        if agent.config is None:
            raise RuntimeError("Agent.config is not set; cannot serve via A2A")
        client = agent.config.create()

        merged_variables = {**dict(agent._agent_variables), **incoming_variables}
        ctx = ContextWithInput(
            session.stream,
            prompt=list(agent._system_prompt),
            dependencies=dict(agent._agent_dependencies),
            variables=merged_variables,
            dependency_provider=agent.dependency_provider,
        )

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
    async def handler(ev: "ModelMessageChunk", _: ContextWithInput) -> None:
        text_pieces.append(ev.content)
        await updater.add_artifact(
            parts=[Part(text=ev.content)],
            artifact_id=text_artifact_id,
            append=True,
        )

    return handler


def _make_client_tool_call_handler(updater: TaskUpdater, session: _Session):
    async def handler(ev: "ClientToolCallEvent", _: ContextWithInput) -> None:
        session.pending_calls[ev.id] = ev
        payload = call_to_payload(ToolCallEvent(id=ev.id, name=ev.name, arguments=ev.arguments))
        await updater.add_artifact(
            parts=[data_part(payload, media_type=MIME_TOOL_CALL)],
            artifact_id=ev.id,
            last_chunk=True,
        )

    return handler
