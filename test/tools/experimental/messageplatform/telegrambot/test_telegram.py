# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

from autogen.import_utils import skip_on_missing_imports
from autogen.tools.experimental.messageplatform import TelegramRetrieveTool, TelegramSendTool


@skip_on_missing_imports(["telethon"], "commsagent-telegram")
class TestTelegramSendTool:
    def test_telegram_send_tool_init(self) -> None:
        telegram_send_tool = TelegramSendTool(api_id="-100", api_hash="my_api_hash", chat_id="my_chat")
        assert telegram_send_tool.name == "telegram_send"
        assert (
            telegram_send_tool.description == "Sends a message to a personal channel, bot channel, group, or channel."
        )
        assert isinstance(telegram_send_tool.func, Callable)  # type: ignore[arg-type]

        expected_schema = {
            "description": "Sends a message to a personal channel, bot channel, group, or channel.",
            "name": "telegram_send",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string", "description": "Message to send to the chat."}},
                "required": ["message"],
            },
        }

        assert telegram_send_tool.function_schema == expected_schema


@skip_on_missing_imports(["telethon"], "commsagent-telegram")
class TestTelegramRetrieveTool:
    def test_telegram_retrieve_tool_init(self) -> None:
        telegram_retrieve_tool = TelegramRetrieveTool(api_id="-100", api_hash="my_api_hash", chat_id="my_chat")
        assert telegram_retrieve_tool.name == "telegram_retrieve"
        assert (
            telegram_retrieve_tool.description
            == "Retrieves messages from a Telegram chat based on datetime/message ID and/or number of latest messages."
        )
        assert isinstance(telegram_retrieve_tool.func, Callable)  # type: ignore[arg-type]

        expected_schema = {
            "description": "Retrieves messages from a Telegram chat based on datetime/message ID and/or number of latest messages.",
            "name": "telegram_retrieve",
            "parameters": {
                "type": "object",
                "properties": {
                    "messages_since": {
                        "type": "string",
                        "default": None,
                        "description": "Date to retrieve messages from (ISO format) OR message ID. If None, retrieves latest messages.",
                    },
                    "maximum_messages": {
                        "type": "integer",
                        "default": None,
                        "description": "Maximum number of messages to retrieve. If None, retrieves all messages since date.",
                    },
                    "search": {
                        "type": "string",
                        "default": None,
                        "description": "Optional string to search for in messages.",
                    },
                },
                "required": [],
            },
        }

        assert telegram_retrieve_tool.function_schema == expected_schema
