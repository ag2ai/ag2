# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.agents.experimental import SlackAgent
from autogen.import_utils import skip_on_missing_imports

from .....conftest import Credentials


@skip_on_missing_imports("slack_sdk", "commsagent-slack")
class TestSlackAgent:
    def test_init(self, mock_credentials: Credentials) -> None:
        slack_agent = SlackAgent(
            name="SlackAgent",
            llm_config=mock_credentials.llm_config,
            bot_token="",
            channel_id="",
        )

        expected_tools = [
            {
                "type": "function",
                "function": {
                    "description": "Sends a message to a Slack channel.",
                    "name": "slack_send",
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
                    "description": "Retrieves messages from a Slack channel based datetime/message ID and/or number of latest messages.",
                    "name": "slack_retrieve",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "messages_since": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                                "description": "Date to retrieve messages from (ISO format) OR Slack message ID. If None, retrieves latest messages.",
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

        assert set(tool.name for tool in slack_agent.tools) == {"slack_send", "slack_retrieve"}
        assert isinstance(slack_agent.llm_config, dict), "llm_config should be a dictionary"
        assert slack_agent.llm_config["tools"] == expected_tools
        assert slack_agent.system_message == (
            "You are a helpful AI assistant that communicates through Slack. "
            "Remember that Slack uses Markdown-like formatting and has message length limits. "
            "Keep messages clear and concise, and consider using appropriate formatting when helpful."
            "\nFormat guidelines for Slack:\n"
            "Format guidelines for Slack:\n"
            "1. Max message length: 40,000 characters\n"
            "2. Supports Markdown-like formatting:\n"
            "   - *text* for italic\n"
            "   - **text** for bold\n"
            "   - `code` for inline code\n"
            "   - ```code block``` for multi-line code\n"
            "3. Supports message threading for organized discussions\n"
            "4. Can use :emoji_name: for emoji reactions\n"
            "5. Supports block quotes with > prefix\n"
            "6. Can use <!here> or <!channel> for notifications"
        )
