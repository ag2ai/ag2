# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Awaitable, Callable, Iterable, MutableMapping
from typing import Annotated, Any, TypeAlias

from fastapi import FastAPI, HTTPException, Path, Response, status

from autogen.agentchat import Agent, ConversableAgent
from autogen.agentchat.conversable_agent import normilize_message_to_oai

from .protocol import AgentBusMessage

AgentName: TypeAlias = str


class HTTPAgentBus:
    def __init__(self, agents: Iterable[Agent]) -> None:
        self.agents: dict[AgentName, Agent] = {agent.name: agent for agent in agents}

        self.app = FastAPI()
        self.app.post("/{agent_name}")(make_remote_call_processor(self.agents))

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        """ASGI interface."""
        await self.app(scope, receive, send)


def make_remote_call_processor(
    agents: dict[AgentName, Agent],
) -> Callable[[], Awaitable[AgentBusMessage | None]]:
    # TODO: stream remote job status to the client
    async def remote_call_processor(
        history: AgentBusMessage,
        agent_name: Annotated[str, Path()],
    ) -> AgentBusMessage | None:
        try:
            agent = agents[agent_name]
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")

        # TODO: support input guardrails
        reply = await agent.a_generate_reply(
            history.messages,
            None,
            exclude={
                ConversableAgent.a_generate_function_call_reply,
                ConversableAgent.a_generate_tool_calls_reply,
                ConversableAgent.generate_code_execution_reply,
                ConversableAgent._generate_code_execution_reply_using_executor,
            },
        )

        # TODO: add ToolExecutionAgent to call local tools
        # TODO: catch update context events

        if reply is None:
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        is_valid, out_message = normilize_message_to_oai(reply, agent_name, role="assistant")

        # TODO: support output guardrails
        if is_valid:
            # TODO: reply with whole local history
            return AgentBusMessage(messages=[out_message])

    return remote_call_processor
