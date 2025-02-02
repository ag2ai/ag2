# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

from autogen.tools.experimental.messageplatform import DiscordRetrieveTool, DiscordSendTool


class TestDiscordSendTool:
    def test_discord_send_tool_init(self) -> None:
        discord_send_tool = DiscordSendTool(bot_token="my_bot_token", guild_name="my_guild", channel_name="my_channel")
        assert discord_send_tool.name == "discord_send"
        assert discord_send_tool.description == "Sends a message to a Discord channel."
        assert isinstance(discord_send_tool.func, Callable)  # type: ignore[arg-type]

        expected_schema = {
            "description": "Sends a message to a Discord channel.",
            "name": "discord_send",
            "parameters": {
                "properties": {"message": {"description": "Message to send to the channel.", "type": "string"}},
                "required": ["message"],
                "type": "object",
            },
        }
        assert discord_send_tool.function_schema == expected_schema


class TestDiscordRetrieveTool:
    def test_discord_retrieve_tool_init(self) -> None:
        discord_retrieve_tool = DiscordRetrieveTool(
            bot_token="my_bot_token", guild_name="my_guild", channel_name="my_channel"
        )
        assert discord_retrieve_tool.name == "discord_retrieve"
        assert (
            discord_retrieve_tool.description
            == "Retrieves messages from a Discord channel based datetime/message ID and/or number of latest messages."
        )
        assert isinstance(discord_retrieve_tool.func, Callable)  # type: ignore[arg-type]

        expected_schema = {
            "description": "Retrieves messages from a Discord channel based datetime/message ID and/or number of latest messages.",
            "name": "discord_retrieve",
            "parameters": {
                "type": "object",
                "properties": {
                    "messages_since": {
                        "type": "string",
                        "default": None,
                        "description": "Date to retrieve messages from (ISO format) OR Discord snowflake ID. If None, retrieves latest messages.",
                    },
                    "maximum_messages": {
                        "type": "integer",
                        "default": None,
                        "description": "Maximum number of messages to retrieve. If None, retrieves all messages since date.",
                    },
                },
                "required": [],
            },
        }

        assert discord_retrieve_tool.function_schema == expected_schema
