# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
"""Agent for sending messages on Slack.

This agent is able to:
- Decide if it should send a message
- Send a message to a specific Slack channel
- Monitor the channel for replies to that message

Installation:
pip install ag2[commsagent-slack]
"""

import time
from datetime import datetime
from typing import List, Optional, Tuple

from pydantic import Field
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

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
)

__PLATFORM_NAME__ = "Slack"  # Platform name for messages
__TIMEOUT__ = 5  # Timeout in seconds
__REPLY_POLL_INTERVAL__ = 2  # Interval in seconds for polling for replies


class SlackConfig(BaseCommsPlatformConfig):
    """Slack-specific configuration.

    To get started:
    1. Create Slack App in your workspace
    2. Add Bot User OAuth token under OAuth & Permissions
    3. Install app to workspace
    4. Add bot to desired channel
    """

    bot_token: str = Field(..., description="Bot User OAuth Token starting with xoxb-")
    channel_id: str = Field(..., description="Channel ID where messages will be sent")
    signing_secret: str = Field(..., description="Signing secret for verifying requests from Slack")
    app_token: Optional[str] = Field(None, description="App-level token starting with xapp- (required for Socket Mode)")

    def validate_config(self) -> bool:
        if not self.bot_token or not self.bot_token.startswith("xoxb-"):
            raise ValueError("Valid Slack bot_token required (should start with xoxb-)")
        if not self.channel_id:
            raise ValueError("channel_id is required")
        if not self.signing_secret:
            raise ValueError("signing_secret is required")
        return True

    class Config:
        extra = "allow"


class SlackHandler:
    """Handles Slack client operations with synchronous functionality."""

    def __init__(self, config: SlackConfig):
        self._config = config
        self._web_client = WebClient(token=config.bot_token)
        self._message_replies = {}

    def _get_user_name(self, user_id: str) -> str:
        """Get user name from user ID."""
        try:
            response = self._web_client.users_info(user=user_id)
            if response["ok"]:
                return response["user"]["real_name"]
        except SlackApiError:
            pass
        return user_id

    def validate(self) -> bool:
        """Validate Slack configuration and connection."""
        try:
            # Test API connection
            response = self._web_client.auth_test()
            if not response["ok"]:
                raise PlatformAuthenticationError(
                    message="Failed to authenticate with Slack", platform_name=__PLATFORM_NAME__
                )

            # Verify channel access
            try:
                response = self._web_client.conversations_info(channel=self._config.channel_id)
                if not response["ok"]:
                    raise PlatformError(
                        message=f"Could not access channel: {self._config.channel_id}", platform_name=__PLATFORM_NAME__
                    )
            except SlackApiError as e:
                if e.response["error"] == "channel_not_found":
                    raise PlatformConnectionError(
                        message=f"Channel not found: {self._config.channel_id}", platform_name=__PLATFORM_NAME__
                    )
                raise

            return True

        except SlackApiError as e:
            if e.response["error"] == "invalid_auth":
                raise PlatformAuthenticationError(
                    message="Invalid authentication token", platform_error=e, platform_name=__PLATFORM_NAME__
                )
            elif e.response["error"] == "rate_limited":
                raise PlatformRateLimitError(
                    message="Rate limit exceeded",
                    platform_error=e,
                    platform_name=__PLATFORM_NAME__,
                    retry_after=float(e.response.headers.get("Retry-After", 60)),
                )
            raise PlatformError(message=f"Slack API error: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__)

    def start(self) -> bool:
        """Start the Slack client and validate connection."""
        try:
            return self.validate()
        except Exception as e:
            raise PlatformError(
                message=f"Error starting Slack client: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__
            )

    def send_message(self, message: str) -> Tuple[str, Optional[str]]:
        """Send a message to the Slack channel.

        Args:
            message: The message to send

        Returns:
            Tuple[str, Optional[str]]: Status message and message ID if successful
        """
        try:
            # Split message if it exceeds Slack's limit (40k characters)
            if len(message) > 40000:
                chunks = [message[i : i + 39999] for i in range(0, len(message), 39999)]
                first_response = None
                for i, chunk in enumerate(chunks):
                    response = self._web_client.chat_postMessage(
                        channel=self._config.channel_id,
                        text=chunk,
                        thread_ts=first_response["ts"] if first_response else None,
                    )
                    if i == 0:
                        first_response = response
                return "Message sent (split into chunks)", first_response["ts"]
            else:
                response = self._web_client.chat_postMessage(channel=self._config.channel_id, text=message)
                return "Message sent successfully", response["ts"]

        except SlackApiError as e:
            if e.response["error"] == "invalid_auth":
                raise PlatformAuthenticationError(
                    message="Invalid authentication token", platform_error=e, platform_name=__PLATFORM_NAME__
                )
            elif e.response["error"] == "rate_limited":
                raise PlatformRateLimitError(
                    message="Rate limit exceeded",
                    platform_error=e,
                    platform_name=__PLATFORM_NAME__,
                    retry_after=float(e.response.headers.get("Retry-After", 60)),
                )
            raise PlatformError(
                message=f"Failed to send message: {str(e)}", platform_error=e, platform_name=__PLATFORM_NAME__
            )

    def wait_for_replies(self, message_id: str, timeout_minutes: int = 5, max_replies: int = 0) -> List[dict]:
        """Wait for replies to a specific message using polling.

        Args:
            message_id: ID of message to monitor for replies
            timeout_minutes: How long to wait for replies (0 = no timeout)
            max_replies: Maximum number of replies to collect (0 = unlimited)

        Returns:
            List of reply messages with content, author, timestamp, and attachments
        """
        if not message_id:
            return []

        # Initialize tracking for this message
        self._message_replies[message_id] = []
        end_time = time.time() + (timeout_minutes * 60) if timeout_minutes > 0 else None

        while True:
            try:
                # Check if timeout has expired and return any replies we have
                if end_time and time.time() > end_time:
                    return self._message_replies[message_id]

                # Get replies from Slack
                response = self._web_client.conversations_replies(channel=self._config.channel_id, ts=message_id)

                # Process any new replies
                for msg in response["messages"][1:]:  # Skip the first message (original)
                    if msg["ts"] not in [r["id"] for r in self._message_replies[message_id]]:
                        reply_data = {
                            "content": msg.get("text", ""),
                            "author": self._get_user_name(msg.get("user")),
                            "timestamp": datetime.fromtimestamp(float(msg.get("ts", 0))).isoformat(),
                            "id": msg.get("ts"),
                        }

                        if "files" in msg:
                            attachments = [
                                f"(Attachment: {f.get('name')}, URL: {f.get('url_private')})" for f in msg["files"]
                            ]
                            if attachments:
                                reply_data["content"] += f" {' '.join(attachments)}"

                        self._message_replies[message_id].append(reply_data)

                        # Check if we've hit max replies
                        if max_replies > 0 and len(self._message_replies[message_id]) >= max_replies:
                            return self._message_replies[message_id]

            except SlackApiError as e:
                if e.response["error"] == "rate_limited":
                    time.sleep(float(e.response.headers.get("Retry-After", 1)))
                else:
                    raise

            # Sleep before next poll
            time.sleep(__REPLY_POLL_INTERVAL__)

    def cleanup_reply_monitoring(self, message_id: str):
        """Clean up reply monitoring for a specific message."""
        self._message_replies.pop(message_id, None)

    def shutdown(self):
        """Shutdown the Slack client."""
        # Nothing needed for synchronous operation
        pass


class SlackExecutor(PlatformExecutorAgent):
    """Slack-specific executor agent.

    See the PlatformExecutorAgent for further details.
    """

    def __init__(self, platform_config: SlackConfig, reply_monitor_config: Optional[ReplyMonitorConfig] = None):
        super().__init__(platform_config, reply_monitor_config)
        self._slack = SlackHandler(platform_config)
        # Validate connection on initialization
        self._slack.start()

    def send_to_platform(self, message: str) -> Tuple[str, Optional[str]]:
        """Send a message to Slack channel.

        Args:
            message: The message to send

        Returns:
            Tuple[str, Optional[str]]: Status message and message ID if successful
        """
        return self._slack.send_message(message)

    def _format_replies(self, replies: List[dict]) -> List[str]:
        """Format replies for display."""
        formatted_replies = []
        for reply in replies:
            timestamp = datetime.fromisoformat(reply["timestamp"])
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M UTC")
            formatted_reply = f"[{formatted_time}] {reply['author']}: {reply['content']}"
            formatted_replies.append(formatted_reply)
        return formatted_replies

    def wait_for_reply(self, msg_id: str) -> List[str]:
        """Wait for reply from Slack.

        Args:
            msg_id: Message ID to monitor for replies

        Returns:
            List of reply messages formatted as strings
        """
        if not self.reply_monitor_config:
            return []

        replies = self._slack.wait_for_replies(
            msg_id,
            timeout_minutes=self.reply_monitor_config.timeout_minutes,
            max_replies=self.reply_monitor_config.max_reply_messages,
        )
        return self._format_replies(replies)

    def cleanup_monitoring(self, msg_id: str):
        """Clean up reply monitoring for a specific message."""
        self._slack.cleanup_reply_monitoring(msg_id)

    def shutdown(self):
        """Clean shutdown of the Slack client."""
        self._slack.cleanup_reply_monitoring(None)  # Cleanup any ongoing monitoring


class SlackAgent(CommsPlatformAgent):
    """Agent for Slack communication.

    See the CommsPlatformAgent for further details.
    """

    def __init__(
        self,
        name: str,
        platform_config: SlackConfig,
        message_to_send: Optional[
            callable
        ] = None,  # The function to determine the message to send, returns None to indicate do not send a message, otherwise determined automatically
        reply_monitor_config: Optional[ReplyMonitorConfig] = None,
        auto_reply: str = "Message sent to Slack",
        system_message: Optional[str] = None,
        *args,
        **kwargs,
    ):
        if system_message is None:
            system_message = (
                "You are a helpful AI assistant that communicates through Slack. "
                "Remember that Slack uses Markdown-like formatting and has message length limits. "
                "Keep messages clear and concise, and consider using appropriate formatting when helpful."
            )

        # Create Slack-specific executor
        slack_executor = SlackExecutor(platform_config, reply_monitor_config)

        super().__init__(
            name=name,
            platform_config=platform_config,
            executor_agent=slack_executor,
            message_to_send=message_to_send,
            reply_monitor_config=reply_monitor_config,
            auto_reply=auto_reply,
            system_message=system_message,
            *args,
            **kwargs,
        )

        # Update decision agent's knowledge of Slack specifics
        self.decision_agent.update_system_message(
            self.decision_agent.system_message
            + "\n"
            + "Format guidelines for Slack:\n"
            + "1. Max message length: 40,000 characters\n"
            + "2. Supports Markdown-like formatting:\n"
            + "   - *text* for italic\n"
            + "   - **text** for bold\n"
            + "   - `code` for inline code\n"
            + "   - ```code block``` for multi-line code\n"
            + "3. Supports message threading for organized discussions\n"
            + "4. Can use :emoji_name: for emoji reactions\n"
            + "5. Supports block quotes with > prefix\n"
            + "6. Can use <!here> or <!channel> for notifications"
        )
