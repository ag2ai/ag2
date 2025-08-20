# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from autogen import Agent, ConversableAgent
from autogen.doc_utils import export_module

if TYPE_CHECKING:
    from .runtime import AgentBus


@export_module("autogen.remote")
class RemoteAgent(ConversableAgent):
    def __init__(self, name: str, runtime: "AgentBus") -> None:
        super().__init__(name, human_input_mode="NEVER")

        self.runtime = runtime
        self.chats: defaultdict[int, list[ConversableAgent]] = defaultdict(list)

    def _prepare_chat(
        self,
        recipient: ConversableAgent,
        chat_id: int,
        clear_history: bool,
        prepare_recipient: bool = True,
        reply_at_receive: bool = True,
    ) -> None:
        self.chats[chat_id].append(recipient)
        return super()._prepare_chat(recipient, chat_id, clear_history, prepare_recipient, reply_at_receive)

    async def a_generate_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: "Agent | None" = None,
        **kwargs: Any,
    ) -> str | dict[str, Any] | None:
        print(messages, sender.name)
        return {
            "role": "assistant",
            "content": "Hello!",
        }
