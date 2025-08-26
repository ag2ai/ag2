# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable
from typing import Annotated, Any, TypeAlias
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Path, Request

from autogen.agentchat import Agent, ConversableAgent
from autogen.agentchat.conversable_agent import normilize_message_to_oai

from .protocol import (
    AgentBusEvent,
    ProtocolEvents,
    SendEvent,
    StopEvent,
    serialize_event,
)

ChatId: TypeAlias = int
AgentName: TypeAlias = str


class AgentBus:
    def __init__(self, agents: Iterable[Agent]) -> None:
        self.agents: dict[AgentName, Agent] = {agent.name: agent for agent in agents}
        self.chats_history: defaultdict[ChatId, list[dict[str | Any] | str | None]] = defaultdict(list)

        self.app = FastAPI()
        self.app.post("/{agent_name}")(make_handler(self.chats_history, self.agents))


async def _serialize_request_to_event(request: Request) -> AgentBusEvent:
    return serialize_event(await request.json())


def _chat_id(request: Request) -> ChatId:
    if "X-Chat-Id" in request.headers:
        return int(request.headers["X-Chat-Id"])

    return uuid4().int


def make_handler(
    chats_history: defaultdict[ChatId, list[dict[str | Any] | str | None]],
    agents: dict[AgentName, Agent],
) -> Callable[[], Awaitable[AgentBusEvent | None]]:
    async def handler(
        agent_name: Annotated[str, Path()],
        event: Annotated[AgentBusEvent, Depends(_serialize_request_to_event)],
        chat_id: Annotated[ChatId, Depends(_chat_id)],
    ) -> AgentBusEvent | None:
        print(event)
        try:
            agent = agents[agent_name]
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")

        if event.event_type is ProtocolEvents.STOP_CHAT:
            del chats_history[chat_id]
            return

        messages = chats_history[chat_id]

        if event.event_type is ProtocolEvents.SEND_MESSAGE:
            _, in_message = normilize_message_to_oai(event.content, str(chat_id), role="user")
            messages.append(in_message)
            return

        if event.event_type is ProtocolEvents.NEXT_SPEAKER:
            reply = agent.generate_reply(
                messages,
                None,
                exclude={
                    ConversableAgent.generate_function_call_reply,
                    ConversableAgent.generate_tool_calls_reply,
                    ConversableAgent.generate_code_execution_reply,
                },
            )

            # TODO: how to call local tools?

            if reply is None:
                return StopEvent()

            _, out_message = normilize_message_to_oai(reply, agent_name, role="assistant")
            messages.append(out_message)
            return SendEvent(content=out_message)

        raise NotImplementedError("unsupported protocol method")

    return handler
