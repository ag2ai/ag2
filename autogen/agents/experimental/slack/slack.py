# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

from .... import ConversableAgent
from ....doc_utils import export_module
from ....tools import Tool
from ....tools.experimental.messageplatform.slack import SlackRetrieveTool, SlackSendTool

__all__ = ["SlackAgent"]


@export_module("autogen.agents")
class SlackAgent(ConversableAgent):
    """An agent that can send messages and retrieve messages on Slack."""

    def __init__(
        self,
        llm_config: dict[str, Any],
        system_message: Optional[Union[str, list]] = (
            "You are a helpful AI assistant that communicates through Slack. "
            "Remember that Slack uses Markdown-like formatting and has message length limits. "
            "Keep messages clear and concise, and consider using appropriate formatting when helpful."
        ),
        *args,
        bot_token: str,
        channel_id: str,
        has_writing_instructions: bool = True,
        **kwargs,
    ) -> None:
        """Initialize the SlackAgent.

        Args:
            llm_config (dict[str, Any]): The LLM configuration.
            bot_token (str): Bot User OAuth Token starting with "xoxb-".
            channel_id (str): Channel ID where messages will be sent.
            has_writing_instructions (bool): Whether to add writing instructions to the system message. Defaults to True.
        """
        self._send_tool = SlackSendTool(bot_token=bot_token, channel_id=channel_id)
        self._retrieve_tool = SlackRetrieveTool(bot_token=bot_token, channel_id=channel_id)

        # Add formatting instructions
        if has_writing_instructions:
            system_message = system_message + (
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

        super().__init__(*args, llm_config=llm_config, system_message=system_message, **kwargs)

        self.register_for_llm()(self._send_tool)
        self.register_for_llm()(self._retrieve_tool)

    @property
    def tools(self) -> list[Tool]:
        return [self._send_tool, self._retrieve_tool]
