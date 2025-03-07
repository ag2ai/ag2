# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from .... import ConversableAgent
from ....doc_utils import export_module
from ....tools.experimental import TelegramRetrieveTool, TelegramSendTool

__all__ = ["TelegramAgent"]


@export_module("autogen.agents.experimental")
class TelegramAgent(ConversableAgent):
    """An agent that can send messages and retrieve messages on Telegram."""

    DEFAULT_SYSTEM_MESSAGE = (
        "You are a helpful AI assistant that communicates through Telegram. "
        "Remember that Telegram uses Markdown-like formatting and has message length limits. "
        "Keep messages clear and concise, and consider using appropriate formatting when helpful."
    )

    def __init__(
        self,
        name: str,
        system_message: Optional[str] = None,
        *,
        api_id: str,
        api_hash: str,
        chat_id: str,
        has_writing_instructions: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the TelegramAgent.

        Args:
            name: The name of the agent.
            system_message: The system message for the agent.
            api_id: Telegram API ID from https://my.telegram.org/apps.
            api_hash: Telegram API hash from https://my.telegram.org/apps.
            chat_id: The ID of the destination (Channel, Group, or User ID).
            has_writing_instructions (bool): Whether to add writing instructions to the system message. Defaults to True.
            **kwargs: Additional keyword arguments passed to the parent ConversableAgent class.
        """

        telegram_system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE

        self._send_tool = TelegramSendTool(api_id=api_id, api_hash=api_hash, chat_id=chat_id)
        self._retrieve_tool = TelegramRetrieveTool(api_id=api_id, api_hash=api_hash, chat_id=chat_id)

        # Add formatting instructions
        if has_writing_instructions:
            formatting_instructions = (
                "\nFormat guidelines for Telegram:\n"
                "1. Max message length: 4096 characters\n"
                "2. HTML formatting:\n"
                "   - <b>bold</b>\n"
                "   - <i>italic</i>\n"
                "   - <code>code</code>\n"
                "   - <pre>code block</pre>\n"
                "   - <a href='URL'>link</a>\n"
                "   - <u>underline</u>\n"
                "   - <s>strikethrough</s>\n"
                "3. HTML rules:\n"
                "   - Tags must be properly closed\n"
                "   - Can't nest identical tags\n"
                "   - Links require full URLs (with http://)\n"
                "4. Supports @mentions and emoji"
            )

            telegram_system_message = telegram_system_message + formatting_instructions

        super().__init__(name=name, system_message=telegram_system_message, **kwargs)

        self.register_for_llm()(self._send_tool)
        self.register_for_llm()(self._retrieve_tool)
