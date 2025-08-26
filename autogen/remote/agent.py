# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from typing import Any

import httpx

from autogen import Agent
from autogen.agentchat.conversable_agent import normilize_message_to_oai
from autogen.agentchat.group.handoffs import Handoffs

from .protocol import NextSpeakerEvent, ProtocolEvents, SendEvent, StopEvent, serialize_event


class RemoteAgent(Agent):
    def __init__(self, url: str, name: str) -> None:
        self.url = f"{url}/{name}"
        self.headers = {}

        # Regular Agent options
        self.name = name  # name is an ID in Distributed protocol

        # AgentChat requires
        self.client_cache = None
        self.silent = True

        # GroupChat requires
        self.description = ""
        self.handoffs = Handoffs()
        self._function_map = {}
        self.input_guardrails = []
        self.output_guardrails = []
        self.tools = ()
        self._oai_messages: defaultdict[Agent, list[dict[str, Any] | str]] = defaultdict(list)

    def register_hook(self, *args: Any) -> None:
        print(args)

    def register_reply(self, *args: Any, **kwargs: Any) -> None:
        print(args, kwargs)

    def _raise_exception_on_async_reply_functions(self) -> None:
        pass

    def _prepare_chat(
        self,
        recipient: "Agent",
        chat_id: int,
        clear_history: bool = False,
        prepare_recipient: bool = True,
        reply_at_receive: bool = True,
    ) -> None:
        # TODO: set headers in `receive` after explicit `chat_id` option added
        self.headers["X-Chat-Id"] = str(chat_id)

    def send(
        self,
        message: dict[str, Any] | str,
        recipient: Agent,
        request_reply: bool | None = None,
        silent: bool | None = False,
    ):
        valid, oai_message = normilize_message_to_oai(message, name=self.name, role="assistant")
        self._oai_messages[recipient].append(oai_message)

        if valid:
            recipient.receive(oai_message, self, request_reply, silent)

        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

    def receive(
        self,
        message: dict[str, Any] | str,
        sender: Agent,
        request_reply: bool | None = None,
        silent: bool | None = False,
    ):
        _, oai_message = normilize_message_to_oai(message, name=sender.name, role="user")
        self._oai_messages[sender].append(oai_message)

        httpx.post(
            self.url,
            content=SendEvent(content=oai_message).model_dump_json(),
            headers=self.headers,
            timeout=30,
        )

        if request_reply is False:
            return

        reply = self.generate_reply(messages=oai_message, sender=sender)

        if reply is not None:
            self.send(reply, sender, silent=silent)

    def generate_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: "Agent | None" = None,
        **kwargs: Any,
    ) -> str | dict[str, Any] | None:
        reply_response = httpx.post(
            self.url,
            content=NextSpeakerEvent().model_dump_json(),
            headers=self.headers,
            timeout=30,
        )
        event = serialize_event(reply_response.json())

        if event.event_type is ProtocolEvents.STOP_CHAT:
            # confirm chat stopped
            httpx.post(
                self.url,
                content=StopEvent().model_dump_json(),
                headers=self.headers,
                timeout=30,
            )
            self.headers.clear()
            return None

        if event.event_type is ProtocolEvents.SEND_MESSAGE:
            return event.content

    # copied methods
    def last_message(self, agent: Agent | None = None) -> dict[str, Any] | None:
        """Copy of ConversableAgent.last_message method."""
        if agent is None:
            n_conversations = len(self._oai_messages)
            if n_conversations == 0:
                return None
            if n_conversations == 1:
                for conversation in self._oai_messages.values():
                    return conversation[-1]
            raise ValueError("More than one conversation is found. Please specify the sender to get the last message.")
        if agent not in self._oai_messages:
            raise KeyError(
                f"The agent '{agent.name}' is not present in any conversation. No history available for this agent."
            )
        return self._oai_messages[agent][-1]
