# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from nlip_sdk.nlip import NLIP_Message

from ag2.agent import Agent
from ag2.context import ConversationContext
from ag2.events import BaseEvent, ModelRequest, ModelResponse, TextInput, ToolResultsEvent
from ag2.stream import MemoryStream
from ag2.tools.final.client_tool import ClientTool
from ag2.tools.final.function_tool import FunctionToolSchema

from .mappers import build_response_message, parse_request_message


class NlipExecutor:
    """Bridge a single NLIP request/response exchange to ``Agent._execute``.

    NLIP sessions are stateless — each call rebuilds a fresh ``MemoryStream``
    and ``Context`` from the history carried on the incoming message,
    without any task/queue machinery since NLIP has no streaming or
    multi-turn task lifecycle on the wire: one request in, one response out.
    """

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    async def execute(self, msg: NLIP_Message) -> NLIP_Message:
        parsed = parse_request_message(msg)

        stream = MemoryStream()
        if parsed.history_events:
            await stream.history.replace(parsed.history_events)

        client_tools = [self._make_client_tool(s) for s in parsed.client_tools]
        initial_event = (
            ToolResultsEvent(parsed.tool_results) if parsed.tool_results else ModelRequest([TextInput(parsed.text)])
        )

        response, final_variables = await self._dispatch(
            initial_event,
            stream,
            client_tools,
            incoming_variables=parsed.context_update,
        )

        text = response.message.content if response.message else ""
        return build_response_message(
            text,
            context_update=final_variables or None,
            tool_calls=response.tool_calls.calls if response.tool_calls else (),
        )

    @staticmethod
    def _make_client_tool(schema: FunctionToolSchema) -> ClientTool:
        return ClientTool({
            "function": {
                "name": schema.function.name,
                "description": schema.function.description,
                "parameters": schema.function.parameters,
            }
        })

    async def _dispatch(
        self,
        initial_event: BaseEvent,
        stream: MemoryStream,
        client_tools: list[ClientTool],
        *,
        incoming_variables: dict[str, Any],
    ) -> tuple[ModelResponse, dict[str, Any]]:
        agent = self._agent
        if agent.config is None:
            raise RuntimeError("Agent.config is not set; cannot serve via NLIP")
        client = agent.config.create()

        merged_variables = {**dict(agent._agent_variables), **incoming_variables}
        ctx = ConversationContext(
            stream,
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


__all__: tuple[str, ...] = ("NlipExecutor",)
