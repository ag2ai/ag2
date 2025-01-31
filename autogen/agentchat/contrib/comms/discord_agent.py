# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
"""Agent for sending messages on Discord.

This agent is able to:
- Decide if it should send a message
- Send a message to a specific Telegram channel, group, or the bot's channel
- Monitor the channel for replies to that message

Installation:
pip install ag2[commsagent-discord]
"""

import asyncio
import signal
import sys
import threading
import time
from datetime import datetime
from typing import List, Optional, Tuple

import discord
from pydantic import Field

from .comms_platform_agent import (
    BaseCommsPlatformConfig,
    CommsPlatformAgent,
    PlatformExecutorAgent,
)
from .platform_configs import ReplyMonitorConfig
from .platform_errors import (
    PlatformAuthenticationError,
    PlatformConnectionError,
    PlatformError,
    PlatformRateLimitError,
    PlatformTimeoutError,
)

__PLATFORM_NAME__ = "Discord"  # Platform name for messages
__TIMEOUT__ = 5  # Timeout in seconds


class DiscordConfig(BaseCommsPlatformConfig):
    """Discord-specific configuration."""

    bot_token: str = Field(..., description="Discord bot token")
    channel_name: str = Field(..., description="Channel name where messages will be sent")
    guild_name: str = Field(..., description="Guild (server) name where the channel is located")

    @property
    def intents(self) -> discord.Intents:
        """Get the required Discord intents required to send a message and check for messages."""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.guild_messages = True
        return intents

    def validate_config(self) -> bool:
        """Validate Discord configuration."""
        if not all([self.bot_token, self.guild_name, self.channel_name]):
            raise ValueError("bot_token, guild_name, and channel_name are required")
        return True

    model_config = ConfigDict(extra="allow")


class DiscordHandler:
    """Handles Discord client operations and connection management."""

    def __init__(self, bot_token: str, guild_name: str, channel_name: str):
        self._bot_token = bot_token
        self._guild_name = guild_name
        self._channel_name = channel_name

        self._client = None
        self._guild = None
        self._channel = None
        self._ready = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._thread = None
        self._loop = None
        self._error = None
        self._is_closed = False  # New flag to track connection state
        self._message_replies = {}  # Store message replies by message ID
        self._reply_events = {}  # Store completion events by message ID

        self._setup_client()

    def _setup_client(self):
        """Set up the Discord client with proper intents."""
        if self._is_closed:  # Check if we need to create a new client
            self._client = None
            self._guild = None
            self._channel = None
            self._ready.clear()
            self._shutdown.clear()
            self._is_closed = False

        if self._client is None:
            intents = discord.Intents.default()
            intents.message_content = True
            intents.guilds = True
            intents.guild_messages = True

            self._client = discord.Client(intents=intents)
            self._setup_events()

    def _setup_events(self):
        """Set up Discord client event handlers."""

        @self._client.event
        async def on_ready():
            try:
                # Validate guild
                self._guild = discord.utils.get(self._client.guilds, name=self._guild_name)
                if not self._guild:
                    self._error = PlatformConnectionError(
                        message=f"Could not find guild: {self._guild_name}", platform_name=__PLATFORM_NAME__
                    )
                    self._ready.set()  # Set ready to unblock validation
                    return

                # Validate channel
                self._channel = discord.utils.get(self._guild.text_channels, name=self._channel_name)
                if not self._channel:
                    self._error = PlatformConnectionError(
                        message=f"Could not find channel: {self._channel_name}", platform_name=__PLATFORM_NAME__
                    )
                    self._ready.set()  # Set ready to unblock validation
                    return

                # Signal that we're ready
                self._ready.set()

            except Exception as e:
                self._error = e
                self._ready.set()  # Set ready to unblock validation

        @self._client.event
        async def on_message(message):
            # Ignore our own messages
            if message.author == self._client.user:
                return

            # Check if this is a reply to one of our monitored messages
            if message.reference and message.reference.message_id:
                ref_id = str(message.reference.message_id)
                if ref_id in self._message_replies:
                    reply_data = {
                        "content": message.content,
                        "author": str(message.author),
                        "timestamp": message.created_at.isoformat(),
                        "id": str(message.id),
                    }

                    # Attachments are concatenated to message content like "(Attachment: )"
                    if message.attachments:
                        attachment_string = " ".join(
                            [f"(Attachment: {a.filename}, URL: {str(a.url)})" for a in message.attachments]
                        )
                        """
                        reply_data['attachments'] = [
                            {'filename': a.filename, 'url': str(a.url)}
                            for a in message.attachments
                        ]
                        """
                        reply_data["content"] += f" {attachment_string}"

                    # Add any embeds
                    if message.embeds:
                        reply_data["embeds"] = [embed.to_dict() for embed in message.embeds]

                    self._message_replies[ref_id].append(reply_data)

                    # Check if we've hit max replies
                    max_replies = self._reply_events[ref_id].get("max_replies", 0)
                    if max_replies > 0 and len(self._message_replies[ref_id]) >= max_replies:
                        self._reply_events[ref_id]["event"].set()

    async def validate(self):
        """Wait for validation to complete and return result."""
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=__TIMEOUT__)
            if self._error:
                # Catch initialisation exceptions, incorrect token, guild, or channel.
                raise PlatformError(
                    message=f"Error initialising {__PLATFORM_NAME__} client",
                    platform_error=self._error,
                    platform_name=__PLATFORM_NAME__,
                )
            return True
        except asyncio.TimeoutError:
            raise PlatformTimeoutError(
                message=f"Timeout waiting for {__PLATFORM_NAME__} validation", platform_name=__PLATFORM_NAME__
            )

    def start(self):
        """Start the Discord client and wait for validation."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._loop = loop
                asyncio.create_task(self._client.start(self._bot_token))
            else:
                self._start_background_thread()
        except RuntimeError:
            self._start_background_thread()

        # Wait for validation in appropriate context
        if self._loop and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self.validate(), self._loop)
            return future.result(timeout=__TIMEOUT__)
        else:
            # For non-async context, we need to use a new event loop
            # since the client is running in its own loop
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.validate())
            finally:
                loop.close()

    def _start_background_thread(self):
        """Start Discord client in a background thread."""

        def run_client():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def start_client():
                try:
                    await self._client.start(self._bot_token)
                except discord.LoginFailure as e:
                    self._error = PlatformAuthenticationError(
                        message="Failed to login with provided token", platform_error=e, platform_name=__PLATFORM_NAME__
                    )
                    self._ready.set()  # Set ready to trigger validation check
                except Exception as e:
                    self._error = PlatformError(
                        message=f"Error during client initialization: {str(e)}",
                        platform_error=e,
                        platform_name=__PLATFORM_NAME__,
                    )
                    self._ready.set()

            try:
                self._loop.run_until_complete(start_client())
                self._loop.run_forever()
            except Exception as e:
                if not isinstance(self._error, (PlatformAuthenticationError, PlatformError)):
                    self._error = PlatformError(
                        message=f"Error in client thread: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__
                    )
                    self._ready.set()
            finally:
                if not self._client.is_closed():
                    self._loop.run_until_complete(self._client.close())
                self._loop.close()

        self._thread = threading.Thread(target=run_client, daemon=True)
        self._thread.start()

        # Give the thread a second to start
        time.sleep(1)

    async def send_message(self, message: str) -> Tuple[str, Optional[str]]:
        """Send a message to the Discord channel.

        Args:
            message: The message to send

        Returns:
            Tuple[str, Optional[str]]: Status message and message ID if successful
        """
        if self._is_closed:
            # If closed, reinitialize and restart
            self._setup_client()
            self.start()

        try:
            await asyncio.wait_for(self._ready.wait(), timeout=5)
        except asyncio.TimeoutError:
            raise PlatformTimeoutError(
                message=f"Timeout waiting for {__PLATFORM_NAME__} client to be ready", platform_name=__PLATFORM_NAME__
            )

        try:
            sent_message_id = None
            if len(message) > 2000:
                chunks = [message[i : i + 1999] for i in range(0, len(message), 1999)]
                for i, chunk in enumerate(chunks):
                    sent = await self._channel.send(chunk)
                    # Store ID of first chunk for reply tracking
                    if i == 0:
                        sent_message_id = str(sent.id)
                return "Message sent (split into chunks)", sent_message_id
            else:
                sent = await self._channel.send(message)
                return "Message sent successfully", str(sent.id)

        except discord.Forbidden as e:
            raise PlatformAuthenticationError(
                message="Bot lacks permission to send messages", platform_error=e, platform_name=__PLATFORM_NAME__
            )
        except discord.HTTPException as e:
            if e.status == 429:  # Rate limit error
                raise PlatformRateLimitError(
                    message="Rate limit exceeded", platform_error=e, platform_name=__PLATFORM_NAME__
                )
            raise PlatformError(
                message=f"Failed to send message: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__
            )

    async def wait_for_replies(self, message_id: str, timeout_minutes: int = 5, max_replies: int = 0) -> List[dict]:
        """Wait for replies to a specific message.

        Args:
            message_id: ID of message to monitor for replies
            timeout_minutes: How long to wait for replies (0 = no timeout)
            max_replies: Maximum number of replies to collect (0 = unlimited)

        Returns:
            List of reply messages with content, author, timestamp, and any attachments/embeds
        """
        if not message_id:
            return []

        # Initialize reply tracking for this message
        self._message_replies[message_id] = []
        event = asyncio.Event()
        self._reply_events[message_id] = {"event": event, "max_replies": max_replies}

        try:
            if timeout_minutes > 0:
                try:
                    await asyncio.wait_for(event.wait(), timeout=timeout_minutes * 60)
                except asyncio.TimeoutError:
                    if not self._message_replies[message_id]:
                        raise PlatformError(
                            message=f"Timeout waiting for replies after {timeout_minutes} minutes",
                            platform_name=__PLATFORM_NAME__,
                        )
        finally:
            # Cleanup and return any replies we got
            replies = self._message_replies.pop(message_id, [])
            self._reply_events.pop(message_id, None)
            print(f"Returning {len(replies)} replies")
            return replies

    def cleanup_reply_monitoring(self, message_id: str):
        """Clean up reply monitoring for a specific message."""
        self._message_replies.pop(message_id, None)
        self._reply_events.pop(message_id, None)

    def shutdown(self):
        """Shutdown the Discord client."""
        if self._client and not self._client.is_closed():
            if self._loop and self._loop.is_running():
                self._loop.create_task(self._client.close())
            else:
                asyncio.run(self._client.close())

        if self._thread and self._thread.is_alive():
            self._shutdown.set()
            self._thread.join(timeout=2)

        self._is_closed = True  # Mark as closed


class DiscordExecutor(PlatformExecutorAgent):
    """Discord-specific executor agent.

    See the PlatformExecutorAgent for further details.
    """

    def __init__(self, platform_config: DiscordConfig, reply_monitor_config: Optional[ReplyMonitorConfig] = None):
        super().__init__(platform_config, reply_monitor_config)

        # Create Discord handler
        self._discord = DiscordHandler(
            bot_token=platform_config.bot_token,
            guild_name=platform_config.guild_name,
            channel_name=platform_config.channel_name,
        )

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start Discord client
        self._discord.start()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.shutdown()
        sys.exit(0)

    def shutdown(self):
        """Clean shutdown of the Discord client."""
        self._discord.shutdown()

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.shutdown()

    def send_to_platform(self, message: str) -> Tuple[str, Optional[str]]:
        """Send a message to Discord channel.

        Args:
            message: The message to send

        Returns:
            Tuple[str, Optional[str]]: Status message and message ID if successful

        Raises:
            PlatformError: For any platform-specific errors
        """
        # Get the event loop
        loop = self._discord._loop

        # If we're in the event loop thread, run directly
        if loop and loop.is_running() and threading.current_thread() == self._discord._thread:
            return loop.run_until_complete(self._discord.send_message(message))

        # Otherwise, run in the appropriate context
        future = asyncio.run_coroutine_threadsafe(self._discord.send_message(message), loop)
        return future.result(timeout=__TIMEOUT__)

    def _format_replies(self, replies: List[dict]) -> List[str]:
        """Format replies for display."""
        formatted_replies = []
        for reply in replies:
            # Time in UTC format
            timestamp = datetime.fromisoformat(reply["timestamp"])
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M UTC")

            formatted_reply = f"[{formatted_time}] {reply['author']}: {reply['content']}"
            formatted_replies.append(formatted_reply)
        return formatted_replies

    def wait_for_reply(self, msg_id: str) -> List[str]:
        """Wait for reply from platform.

        Args:
            msg_id: Message ID to monitor for replies

        Returns:
            List of reply messages, one for each message formatted as a string
        """
        if not self.reply_monitor_config:
            return []

        # Get the event loop
        loop = self._discord._loop

        try:
            # If we're in the event loop thread, run directly
            if loop and loop.is_running() and threading.current_thread() == self._discord._thread:
                replies = loop.run_until_complete(
                    self._discord.wait_for_replies(
                        msg_id,
                        timeout_minutes=self.reply_monitor_config.timeout_minutes,
                        max_replies=self.reply_monitor_config.max_reply_messages,
                    )
                )
                return self._format_replies(replies)

            # Otherwise, run in the appropriate context
            future = asyncio.run_coroutine_threadsafe(
                self._discord.wait_for_replies(
                    msg_id,
                    timeout_minutes=self.reply_monitor_config.timeout_minutes,
                    max_replies=self.reply_monitor_config.max_reply_messages,
                ),
                loop,
            )

            try:
                # No timeout here - we rely on the timeout in wait_for_replies
                replies = future.result()
                return self._format_replies(replies)
            except Exception as e:
                # Log other errors but return empty list to avoid breaking the chat flow
                print(f"Error waiting for replies: {str(e)}")
                return []

        except Exception as e:
            # Catch any other exceptions that might occur
            print(f"Unexpected error in wait_for_reply: {str(e)}")
            return []

    def cleanup_monitoring(self, msg_id: str):
        """Clean up reply monitoring for a specific message."""
        self._discord.cleanup_reply_monitoring(msg_id)


class DiscordAgent(CommsPlatformAgent):
    """Agent for Discord communication.

    See the CommsPlatformAgent for further details.
    """

    def __init__(
        self,
        name: str,
        platform_config: DiscordConfig,
        message_to_send: Optional[
            callable
        ] = None,  # The function to determine the message to send, returns None to indicate do not send a message, otherwise determined automatically
        reply_monitor_config: Optional[ReplyMonitorConfig] = None,
        auto_reply: str = "Message sent to Discord",
        system_message: Optional[str] = None,
        *args,
        **kwargs,
    ):
        if system_message is None:
            system_message = (
                "You are a helpful AI assistant that communicates through Discord. "
                "Remember that Discord uses Markdown for formatting and has a character limit. "
                "Keep messages clear and concise, and consider using appropriate formatting when helpful."
            )

        # Create Discord-specific executor
        discord_executor = DiscordExecutor(platform_config, reply_monitor_config)

        super().__init__(
            name=name,
            platform_config=platform_config,
            executor_agent=discord_executor,
            message_to_send=message_to_send,
            reply_monitor_config=reply_monitor_config,
            auto_reply=auto_reply,
            system_message=system_message,
            *args,
            **kwargs,
        )

        # Update decision agent's knowledge of Discord specifics
        self.decision_agent.update_system_message(
            self.decision_agent.system_message
            + "\n"
            + "Format guidelines for Discord:\n"
            + "1. Max message length: 2000 characters\n"
            + "2. Supports Markdown formatting\n"
            + "3. Can use ** for bold, * for italic, ``` for code blocks\n"
            + "4. Consider using appropriate emojis when suitable"
        )
