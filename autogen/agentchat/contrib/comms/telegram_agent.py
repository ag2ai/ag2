# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
"""Agent for sending messages on Telegram.

This agent is able to:
- Decide if it should send a message
- Send a message to a specific Discord channel
- Monitor the channel for replies to that message

Installation:
pip install ag2[commsagent-telegram]
"""

import asyncio
import threading
import time
from datetime import datetime
from typing import Callable, Optional, Tuple, Union

import telegram
from pydantic import Field
from telegram.ext import (
    Application,
    ApplicationBuilder,
)

from .comms_platform_agent import (
    CommsPlatformAgent,
    PlatformExecutorAgent,
)
from .platform_configs import BaseCommsPlatformConfig, ReplyMonitorConfig
from .platform_errors import (
    PlatformAuthenticationError,
    PlatformConnectionError,
    PlatformError,
    PlatformRateLimitError,
    PlatformTimeoutError,
)

__PLATFORM_NAME__ = "Telegram"
__TIMEOUT__ = 5  # Timeout in seconds


class TelegramConfig(BaseCommsPlatformConfig):
    """Telegram configuration using Bot API.

    Args:
        bot_token (str): Bot token from BotFather (starts with numbers:ABC...).
        destination_id (str): Bot's channel Id, Group Id with bot in it, or Channel with bot in it

    Instructions on finding the right id:
    https://gist.github.com/nafiesl/4ad622f344cd1dc3bb1ecbe468ff9f8a
    """

    bot_token: str = Field(..., description="Bot token")
    destination_id: str = Field(..., description="Bot's Channel Id, Group's Id, or Channel's Id")

    def validate_config(self) -> bool:
        if not self.bot_token:
            raise ValueError("bot_token is required")
        if not self.destination_id:
            raise ValueError("destination_id is required")
        return True

    model_config = ConfigDict(extra="allow")


class TelegramHandler:
    """Handles Telegram client operations using Application pattern."""

    def __init__(self, config: TelegramConfig):
        self._config = config
        self._app: Optional[Application] = None
        self._message_replies: dict[str, list[dict]] = {}
        self._reply_events: dict[str, asyncio.Event] = {}
        self._ready = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._thread = None
        self._loop = None
        self._error: Optional[Exception] = None
        self._is_closed = False
        self._initialized = False

    def _start_background_thread(self):
        """Start Telegram client in a background thread."""

        def run_client():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def start_client():
                try:
                    self._app = (
                        ApplicationBuilder()
                        .token(self._config.bot_token)
                        .connection_pool_size(8)
                        .get_updates_connection_pool_size(8)
                        .pool_timeout(30.0)
                        .build()
                    )
                    await self._app.initialize()
                    await self._app.start()

                    try:
                        await self._app.bot.get_chat(self._config.destination_id)
                        self._ready.set()
                    except telegram.error.Forbidden:
                        self._error = PlatformConnectionError(
                            message=f"Could not access chat: {self._config.destination_id}",
                            platform_name=__PLATFORM_NAME__,
                        )
                        self._ready.set()
                except telegram.error.InvalidToken as e:
                    self._error = PlatformAuthenticationError(
                        message="Invalid bot token", platform_error=e, platform_name=__PLATFORM_NAME__
                    )
                    self._ready.set()
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
                if self._app:
                    self._loop.run_until_complete(self._app.stop())
                self._loop.close()

        self._thread = threading.Thread(target=run_client, daemon=True)
        self._thread.start()

        # Give thread time to start
        time.sleep(1)

    async def initialize(self) -> None:
        """Initialize the Telegram application."""
        if self._initialized:
            return

        try:
            builder = ApplicationBuilder().token(self._config.bot_token)
            builder.connection_pool_size(8)
            builder.get_updates_connection_pool_size(8)
            builder.pool_timeout(30.0)
            self._app = builder.build()

            if not self._app:
                raise PlatformError(
                    message="Failed to create Telegram application",
                    platform_name=__PLATFORM_NAME__
                )

            # Start the application
            await self._app.initialize()
            await self._app.start()

            # Verify chat access
            try:
                await self._app.bot.get_chat(self._config.destination_id)
            except telegram.error.Forbidden:
                raise PlatformConnectionError(
                    message=f"Could not access chat: {self._config.destination_id}",
                    platform_name=__PLATFORM_NAME__
                )

            self._initialized = True
        except telegram.error.InvalidToken as e:
            raise PlatformAuthenticationError(
                message="Invalid bot token",
                platform_error=e,
                platform_name=__PLATFORM_NAME__
            )
        except Exception as e:
            raise PlatformError(
                message=f"Error initializing Telegram: {str(e)}",
                platform_error=e,
                platform_name=__PLATFORM_NAME__
            )

    async def send_message(self, message: str) -> Tuple[str, Optional[str]]:
        """Send a message to the Telegram chat."""
        if not self._initialized:
            await self.initialize()

        if not self._app:
            raise PlatformError(
                message="Telegram application not initialized",
                platform_name=__PLATFORM_NAME__
            )

        try:
            # Split message if it exceeds Telegram's limit (4096 characters)
            if len(message) > 4096:
                chunks = [message[i : i + 4095] for i in range(0, len(message), 4095)]
                first_message = None

                for chunk in chunks:
                    sent_message = await self._app.bot.send_message(
                        chat_id=self._config.destination_id,
                        text=chunk,
                        parse_mode="HTML",
                        reply_to_message_id=first_message.message_id if first_message else None,
                    )
                    if not first_message:
                        first_message = sent_message

                if not first_message:
                    raise PlatformError(
                        message="Failed to send message chunks",
                        platform_name=__PLATFORM_NAME__
                    )

                return "Message sent (split into chunks)", str(first_message.message_id)
            else:
                sent_message = await self._app.bot.send_message(
                    chat_id=self._config.destination_id,
                    text=message,
                    parse_mode="HTML"
                )
                return "Message sent successfully", str(sent_message.message_id)

        except telegram.error.TimedOut as e:
            raise PlatformError(
                message="Telegram request timed out",
                platform_error=e,
                platform_name=__PLATFORM_NAME__
            )
        except telegram.error.TelegramError as e:
            raise PlatformError(
                message=f"Telegram API error: {str(e)}",
                platform_error=e,
                platform_name=__PLATFORM_NAME__
            )
        except Exception as e:
            raise PlatformError(
                message=f"Error sending message: {str(e)}",
                platform_error=e,
                platform_name=__PLATFORM_NAME__
            )

    def start(self):
        """Start the Telegram client and wait for validation."""
        try:
            self._start_background_thread()
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.validate())
            finally:
                loop.close()
        except Exception as e:
            raise PlatformError(
                message=f"Error starting Telegram client: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__
            )

    async def validate(self):
        """Wait for validation to complete and return result."""
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=__TIMEOUT__)
            if self._error:
                raise self._error
            return True
        except asyncio.TimeoutError:
            raise PlatformTimeoutError(
                message=f"Timeout waiting for {__PLATFORM_NAME__} validation", platform_name=__PLATFORM_NAME__
            )

    async def wait_for_replies(self, message_id: str, timeout_minutes: int = 5, max_replies: int = 0) -> list[dict]:
        """Wait for replies to a specific message."""
        if not message_id:
            return []

        if not self._initialized:
            await self.initialize()

        if not self._app:
            raise PlatformError(
                message="Telegram application not initialized",
                platform_name=__PLATFORM_NAME__
            )

        # Initialize tracking for this message
        self._message_replies[message_id] = []
        start_time = datetime.now()
        timeout_seconds = timeout_minutes * 60 if timeout_minutes > 0 else float("inf")

        try:
            offset = 0
            while (datetime.now() - start_time).total_seconds() < timeout_seconds:
                try:
                    updates = await self._app.bot.get_updates(
                        offset=offset,
                        timeout=1,
                        allowed_updates=["message", "channel_post"],
                        write_timeout=20,
                    )

                    for update in updates:
                        if update.update_id >= offset:
                            offset = update.update_id + 1

                        msg = update.message or update.channel_post
                        if not msg or not msg.reply_to_message:
                            continue

                        if str(msg.reply_to_message.message_id) == message_id:
                            reply_data = {
                                "content": msg.text or "",
                                "author": msg.from_user.username if msg.from_user else "Channel",
                                "timestamp": msg.date.isoformat(),
                                "id": str(msg.message_id),
                            }

                            # Handle media attachments
                            if msg.photo:
                                reply_data["content"] += " (Attachment: photo)"
                            elif msg.document:
                                reply_data["content"] += f" (Attachment: document - {msg.document.file_name})"
                            elif msg.video:
                                reply_data["content"] += " (Attachment: video)"
                            elif msg.audio:
                                reply_data["content"] += " (Attachment: audio)"
                            elif msg.voice:
                                reply_data["content"] += " (Attachment: voice)"

                            self._message_replies[message_id].append(reply_data)

                            if max_replies > 0 and len(self._message_replies[message_id]) >= max_replies:
                                return self._message_replies[message_id][:max_replies]

                except telegram.error.TimedOut:
                    # Ignore timeout errors during polling
                    pass
                except telegram.error.TelegramError as e:
                    raise PlatformError(
                        message=f"Telegram API error during polling: {str(e)}",
                        platform_error=e,
                        platform_name=__PLATFORM_NAME__
                    )
                except Exception as e:
                    raise PlatformError(
                        message=f"Error during reply polling: {str(e)}",
                        platform_error=e,
                        platform_name=__PLATFORM_NAME__
                    )

                await asyncio.sleep(1)

            # If we reach here, we've timed out
            raise asyncio.TimeoutError()

        except asyncio.TimeoutError:
            # Clean up and re-raise as TimeoutError for consistent handling
            self._message_replies.pop(message_id, [])
            raise
        except Exception as e:
            # Clean up and re-raise
            self._message_replies.pop(message_id, [])
            raise e

    async def cleanup(self):
        """Cleanup monitoring resources without shutting down app."""
        try:
            self._message_replies.clear()
        except Exception:
            pass

    def cleanup_reply_monitoring(self, message_id: str):
        """Clean up reply monitoring for a specific message."""
        if message_id:
            self._message_replies.pop(message_id, None)


class TelegramExecutor(PlatformExecutorAgent):
    """Telegram-specific executor agent.

    See the PlatformExecutorAgent for further details.
    """

    def __init__(self, platform_config: TelegramConfig, reply_monitor_config: Optional[ReplyMonitorConfig] = None):
        super().__init__(platform_config, reply_monitor_config)
        self._telegram = TelegramHandler(platform_config)

    async def initialize(self) -> None:
        """Initialize the Telegram executor."""
        await self._telegram.initialize()

    async def send_to_platform(self, message: str) -> Tuple[str, Optional[str]]:
        """Send a message to Telegram channel."""
        try:
            return await self._telegram.send_message(message)
        except telegram.error.InvalidToken as e:
            raise PlatformAuthenticationError(
                message="Invalid bot token",
                platform_error=e,
                platform_name=__PLATFORM_NAME__
            ) from e
        except telegram.error.BadRequest as e:
            raise PlatformConnectionError(
                message=str(e),
                platform_error=e,
                platform_name=__PLATFORM_NAME__
            ) from e
        except telegram.error.RetryAfter as e:
            raise PlatformRateLimitError(
                message=str(e),
                platform_error=e,
                platform_name=__PLATFORM_NAME__,
                retry_after=e.retry_after
            ) from e
        except telegram.error.NetworkError as e:
            raise PlatformConnectionError(
                message=str(e),
                platform_error=e,
                platform_name=__PLATFORM_NAME__
            ) from e
        except Exception as e:
            if isinstance(e, (PlatformError, PlatformTimeoutError, PlatformConnectionError)):
                raise
            raise PlatformError(
                message=f"Error sending message: {str(e)}",
                platform_error=e,
                platform_name=__PLATFORM_NAME__
            ) from e

    def _format_replies(self, replies: list[dict]) -> list[str]:
        """Format replies for display."""
        formatted_replies = []
        for reply in replies:
            timestamp = datetime.fromisoformat(reply["timestamp"])
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M UTC")
            formatted_reply = f"[{formatted_time}] {reply['author']}: {reply['content']}"
            formatted_replies.append(formatted_reply)
            if self.reply_monitor_config and len(formatted_replies) >= self.reply_monitor_config.max_reply_messages:
                break
        return formatted_replies

    async def wait_for_reply(self, msg_id: str) -> list[str]:
        """Wait for reply from platform."""
        if not self.reply_monitor_config:
            return []

        try:
            replies = await self._telegram.wait_for_replies(
                msg_id,
                timeout_minutes=self.reply_monitor_config.timeout_minutes,
                max_replies=self.reply_monitor_config.max_reply_messages,
            )
            return self._format_replies(replies)
        except asyncio.TimeoutError as e:
            raise PlatformTimeoutError(
                message="Timeout waiting for replies",
                platform_name="Telegram"
            ) from e
        except Exception:
            return []

    async def cleanup_monitoring(self, msg_id: str) -> None:
        """Clean up reply monitoring for a specific message."""
        self._telegram.cleanup_reply_monitoring(msg_id)
        if not self._telegram._message_replies:
            try:
                await self._telegram.cleanup()
            except Exception as e:
                raise PlatformError(
                    message=f"Error during cleanup: {str(e)}",
                    platform_error=e,
                    platform_name=__PLATFORM_NAME__
                ) from e


class TelegramAgent(CommsPlatformAgent):
    """Agent for Telegram communication.

    See the CommsPlatformAgent for further details.
    """

    def __init__(
        self,
        name: str,
        platform_config: TelegramConfig,
        message_to_send: Optional[Callable[..., Union[str, None]]] = None,
        reply_monitor_config: Optional[ReplyMonitorConfig] = None,
        auto_reply: str = "Message sent to Telegram",
        system_message: Optional[str] = None,
        *args,
        **kwargs,
    ):
        if system_message is None:
            system_message = (
                "You are a helpful AI assistant that communicates through Telegram. "
                "Remember that Telegram uses HTML for formatting and has message length limits. "
                "Keep messages clear and concise, and consider using appropriate formatting when helpful."
            )

        # Create Telegram-specific executor
        self._executor = TelegramExecutor(platform_config, reply_monitor_config)

        super().__init__(
            name=name,
            platform_config=platform_config,
            executor_agent=self._executor,
            message_to_send=message_to_send,
            reply_monitor_config=reply_monitor_config,
            auto_reply=auto_reply,
            system_message=system_message,
            *args,
            **kwargs,
        )

    async def initialize(self) -> None:
        """Initialize the Telegram agent."""
        await self._executor.initialize()
