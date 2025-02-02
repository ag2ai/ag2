# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

from autogen.tools.experimental.messageplatform import SlackRetrieveTool, SlackSendTool


class TestSlackSendTool:
    def test_slack_send_tool_init(self) -> None:
        slack_send_tool = SlackSendTool(bot_token="my_bot_token", channel_id="my_channel")
        assert slack_send_tool.name == "slack_send"
        assert slack_send_tool.description == "Sends a message to a Slack channel."
        assert isinstance(slack_send_tool.func, Callable)  # type: ignore[arg-type]

        expected_schema = {
            "description": "Sends a message to a Slack channel.",
            "name": "slack_send",
            "parameters": {
                "properties": {"message": {"description": "Message to send to the channel.", "type": "string"}},
                "required": ["message"],
                "type": "object",
            },
        }
        assert slack_send_tool.function_schema == expected_schema


class TestSlackRetrieveTool:
    def test_slack_retrieve_tool_init(self) -> None:
        slack_retrieve_tool = SlackRetrieveTool(bot_token="my_bot_token", channel_id="my_channel")
        assert slack_retrieve_tool.name == "slack_retrieve"
        assert (
            slack_retrieve_tool.description
            == "Retrieves messages from a Slack channel based datetime/message ID and/or number of latest messages."
        )
        assert isinstance(slack_retrieve_tool.func, Callable)  # type: ignore[arg-type]

        expected_schema = {
            "description": "Retrieves messages from a Slack channel based datetime/message ID and/or number of latest messages.",
            "name": "slack_retrieve",
            "parameters": {
                "type": "object",
                "properties": {
                    "messages_since": {
                        "type": "string",
                        "default": None,
                        "description": "Date to retrieve messages from (ISO format) OR Slack message ID. If None, retrieves latest messages.",
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

        assert slack_retrieve_tool.function_schema == expected_schema
