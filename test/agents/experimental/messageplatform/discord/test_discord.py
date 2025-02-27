# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.agents.experimental import DiscordAgent
from autogen.import_utils import skip_on_missing_imports

from .....conftest import Credentials


@skip_on_missing_imports("discord", "commsagent-discord")
class TestDiscordAgent:
    def test_init(self, mock_credentials: Credentials) -> None:
        discord_agent = DiscordAgent(
            name="DiscordAgent", llm_config=mock_credentials.llm_config, bot_token="", channel_name="", guild_name=""
        )

        expected_tools = [
            {
                "type": "function",
                "function": {
                    "description": "Sends a message to a Discord channel.",
                    "name": "discord_send",
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string", "description": "Message to send to the channel."}},
                        "required": ["message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "description": "Retrieves messages from a Discord channel based datetime/message ID and/or number of latest messages.",
                    "name": "discord_retrieve",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "messages_since": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                                "description": "Date to retrieve messages from (ISO format) OR Discord snowflake ID. If None, retrieves latest messages.",
                            },
                            "maximum_messages": {
                                "anyOf": [{"type": "integer"}, {"type": "null"}],
                                "default": None,
                                "description": "Maximum number of messages to retrieve. If None, retrieves all messages since date.",
                            },
                        },
                        "required": [],
                    },
                },
            },
        ]

        assert set(tool.name for tool in discord_agent.tools) == {"discord_send", "discord_retrieve"}
        assert isinstance(discord_agent.llm_config, dict), "llm_config should be a dictionary"
        assert discord_agent.llm_config["tools"] == expected_tools
        assert discord_agent.system_message == (
            "You are a helpful AI assistant that communicates through Discord. "
            "Remember that Discord uses Markdown for formatting and has a character limit. "
            "Keep messages clear and concise, and consider using appropriate formatting when helpful."
            "\nFormat guidelines for Discord:\n"
            "1. Max message length: 2000 characters\n"
            "2. Supports Markdown formatting\n"
            "3. Can use ** for bold, * for italic, ``` for code blocks\n"
            "4. Consider using appropriate emojis when suitable\n"
        )
