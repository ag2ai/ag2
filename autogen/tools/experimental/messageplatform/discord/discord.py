# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Annotated, Any

from .....doc_utils import export_module
from .....import_utils import optional_import_block, require_optional_import
from .... import Tool
from ....dependency_injection import Depends, on

__all__ = ["DiscordSendTool"]

with optional_import_block():
    import discord


@require_optional_import(["discord"], "commsagent-discord")
@export_module("autogen.tools.experimental")
class DiscordSendTool(Tool):
    """Sends a message to a Discord channel."""

    def __init__(self, *, bot_token: str, channel_name: str, guild_name: str) -> None:
        """
        Initialize the DiscordSendTool.

        Args:
            bot_token: The bot token to use for sending messages.
            channel_name: The name of the channel to send messages to.
            guild_name: The name of the guild to send messages to.
        """

        # Function that sends the message, uses dependency injection for bot token / channel / guild
        async def discord_send_message(
            message: Annotated[str, "Message to send to the channel"],
            bot_token: Annotated[str, Depends(on(bot_token))],
            guild_name: Annotated[str, Depends(on(guild_name))],
            channel_name: Annotated[str, Depends(on(channel_name))],
        ) -> Any:
            """
            Sends a message to a Discord channel.

            Args:
                message: The message to send to the channel.
                bot_token: The bot token to use for Discord. (uses dependency injection)
                guild_name: The name of the server. (uses dependency injection)
                channel_name: The name of the channel. (uses dependency injection)
            """
            intents = discord.Intents.default()
            intents.message_content = True
            intents.guilds = True
            intents.guild_messages = True

            client = discord.Client(intents=intents)
            result_future: asyncio.Future[str] = asyncio.Future()  # Stores the result of the send

            # When the client is ready, we'll send the message
            @client.event  # type: ignore[misc]
            async def on_ready() -> None:
                try:
                    # Server
                    guild = discord.utils.get(client.guilds, name=guild_name)
                    if guild:
                        # Channel
                        channel = discord.utils.get(guild.text_channels, name=channel_name)
                        if channel:
                            # Send the message
                            sent = await channel.send(message)

                            result_future.set_result(f"Message sent successfully (ID: {sent.id}):\n{message}")
                        else:
                            result_future.set_result(f"Message send failed, could not find channel: {channel_name}")
                    else:
                        result_future.set_result(f"Message send failed, could not find guild: {guild_name}")

                except Exception as e:
                    result_future.set_exception(e)
                finally:
                    try:
                        await client.close()
                    except Exception as e:
                        raise Exception(f"Unable to close Discord client: {e}")

            # Start the client and when it's ready it'll send the message in on_ready
            try:
                await client.start(bot_token)

                # Capture the result of the send
                return await result_future
            except Exception as e:
                raise Exception(f"Failed to start Discord client: {e}")

        super().__init__(
            name="discord_send",
            description="Sends a message to a Discord channel.",
            func_or_tool=discord_send_message,
        )
