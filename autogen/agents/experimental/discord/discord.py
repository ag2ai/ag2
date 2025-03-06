# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from .... import ConversableAgent
from ....doc_utils import export_module
from ....tools.experimental import DiscordRetrieveTool, DiscordSendTool

__all__ = ["DiscordAgent"]


@export_module("autogen.agents.experimental")
class DiscordAgent(ConversableAgent):
    """An agent that can send messages and retrieve messages on Discord."""

    DEFAULT_SYSTEM_MESSAGE = (
        "You are a helpful AI assistant that communicates through Discord. "
        "Remember that Discord uses Markdown for formatting and has a character limit. "
        "Keep messages clear and concise, and consider using appropriate formatting when helpful."
    )

    def __init__(
        self,
        name: str,
        system_message: Optional[str] = None,
        *,
        bot_token: str,
        channel_name: str,
        guild_name: str,
        has_writing_instructions: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the DiscordAgent.

        Args:
            name: name of the agent.
            system_message: system message for the ChatCompletion inference.
            bot_token: Discord bot token
            channel_name: Channel name where messages will be sent / retrieved
            guild_name: Guild (server) name where the channel is located
            has_writing_instructions: Whether to add writing instructions to the system message. Defaults to True.
            **kwargs: Additional keyword arguments passed to the parent ConversableAgent class.
        """
        discord_system_message = system_message or self.DEFAULT_SYSTEM_MESSAGE

        self._send_tool = DiscordSendTool(bot_token=bot_token, channel_name=channel_name, guild_name=guild_name)
        self._retrieve_tool = DiscordRetrieveTool(bot_token=bot_token, channel_name=channel_name, guild_name=guild_name)

        # Add formatting instructions
        if has_writing_instructions:
            formatting_instructions = (
                "\nFormat guidelines for Discord:\n"
                "1. Max message length: 2000 characters\n"
                "2. Supports Markdown formatting\n"
                "3. Can use ** for bold, * for italic, ``` for code blocks\n"
                "4. Consider using appropriate emojis when suitable\n"
            )

            discord_system_message = discord_system_message + formatting_instructions

        super().__init__(name=name, system_message=discord_system_message, **kwargs)

        self.register_for_llm()(self._send_tool)
        self.register_for_llm()(self._retrieve_tool)
