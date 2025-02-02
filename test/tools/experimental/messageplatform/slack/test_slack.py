# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

from autogen.tools.experimental.messageplatform import SlackSendTool


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
