# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

from autogen.import_utils import skip_on_missing_imports
from autogen.tools.experimental.messageplatform import TelegramSendTool


@skip_on_missing_imports(["telegram"], "commsagent-telegram")
class TestTelegramSendTool:
    def test_telegram_send_tool_init(self) -> None:
        telegram_send_tool = TelegramSendTool(api_id="my_api_id", api_hash="my_api_hash", chat_id="my_chat")
        assert telegram_send_tool.name == "telegram_send"
        assert telegram_send_tool.description == "Sends a message to a Telegram bot, group, or channel."
        assert isinstance(telegram_send_tool.func, Callable)  # type: ignore[arg-type]

        expected_schema = {
            "description": "Sends a message to a Telegram bot, group, or channel.",
            "name": "telegram_send",
            "parameters": {
                "properties": {
                    "message": {
                        "description": "Message to send to the bot's channel, group, or channel.",
                        "type": "string",
                    }
                },
                "required": ["message"],
                "type": "object",
            },
        }
        assert telegram_send_tool.function_schema == expected_schema
