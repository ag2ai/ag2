# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.agents.experimental import TelegramAgent
from autogen.import_utils import skip_on_missing_imports

from .....conftest import Credentials


@skip_on_missing_imports("telethon", "commsagent-telegram")
class TestTelegramAgent:
    def test_init(self, mock_credentials: Credentials) -> None:
        telegram_agent = TelegramAgent(
            name="TelegramAgent", llm_config=mock_credentials.llm_config, api_id="", api_hash="", chat_id=""
        )

        expected_tools = [
            {
                "type": "function",
                "function": {
                    "description": "Sends a message to a personal channel, bot channel, group, or channel.",
                    "name": "telegram_send",
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string", "description": "Message to send to the chat."}},
                        "required": ["message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "description": "Retrieves messages from a Telegram chat based on datetime/message ID and/or number of latest messages.",
                    "name": "telegram_retrieve",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "messages_since": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                                "description": "Date to retrieve messages from (ISO format) OR message ID. If None, retrieves latest messages.",
                            },
                            "maximum_messages": {
                                "anyOf": [{"type": "integer"}, {"type": "null"}],
                                "default": None,
                                "description": "Maximum number of messages to retrieve. If None, retrieves all messages since date.",
                            },
                            "search": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                                "description": "Optional string to search for in messages.",
                            },
                        },
                        "required": [],
                    },
                },
            },
        ]

        assert set(tool.name for tool in telegram_agent.tools) == {"telegram_send", "telegram_retrieve"}
        assert isinstance(telegram_agent.llm_config, dict), "llm_config should be a dictionary"
        assert telegram_agent.llm_config["tools"] == expected_tools
        assert telegram_agent.system_message == (
            "You are a helpful AI assistant that communicates through Telegram. "
            "Remember that Telegram uses Markdown-like formatting and has message length limits. "
            "Keep messages clear and concise, and consider using appropriate formatting when helpful."
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
