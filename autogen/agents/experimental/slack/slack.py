# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from .... import ConversableAgent
from ....doc_utils import export_module
from ....tools.experimental import SlackRetrieveTool, SlackSendTool

__all__ = ["SlackAgent"]


@export_module("autogen.agents.experimental")
class SlackAgent(ConversableAgent):
    """An agent that can send messages and retrieve messages on Slack."""

    DEFAULT_SYSTEM_MESSAGE = (
        "You are a helpful AI assistant that communicates through Slack. "
        "Remember that Slack uses Markdown-like formatting and has message length limits. "
        "Keep messages clear and concise, and consider using appropriate formatting when helpful."
    )

    def __init__(
        self,
        name: str,
        system_message: Optional[str] = None,
        *,
        bot_token: str,
        channel_id: str,
        has_writing_instructions: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the SlackAgent.

        Args:
            name: name of the agent.
            system_message: system message for the ChatCompletion inference.
            bot_token: Bot User OAuth Token starting with "xoxb-".
            channel_id: Channel ID where messages will be sent.
            has_writing_instructions: Whether to add writing instructions to the system message. Defaults to True.
            **kwargs: Additional keyword arguments passed to the parent ConversableAgent class.
        """
        slack_system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE

        self._send_tool = SlackSendTool(bot_token=bot_token, channel_id=channel_id)
        self._retrieve_tool = SlackRetrieveTool(bot_token=bot_token, channel_id=channel_id)

        # Add formatting instructions
        if has_writing_instructions:
            formatting_instructions = (
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

            slack_system_message = slack_system_message + formatting_instructions

        super().__init__(name=name, system_message=slack_system_message, **kwargs)

        self.register_for_llm()(self._send_tool)
        self.register_for_llm()(self._retrieve_tool)
