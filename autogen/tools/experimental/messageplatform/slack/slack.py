# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Annotated, Any, Union

from .....doc_utils import export_module
from .....import_utils import optional_import_block, require_optional_import
from .... import Tool
from ....dependency_injection import Depends, on

__all__ = ["SlackSendTool"]

with optional_import_block():
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

MAX_MESSAGE_LENGTH = 40000


@require_optional_import(["slack_sdk"], "commsagent-slack")
@export_module("autogen.tools.experimental")
class SlackSendTool(Tool):
    """Sends a message to a Slack channel."""

    def __init__(self, *, bot_token: str, channel_id: str) -> None:
        """
        Initialize the SlackSendTool.

        Args:
            bot_token: Bot User OAuth Token starting with "xoxb-".
            channel_id: Channel ID where messages will be sent.
        """

        # Function that sends the message, uses dependency injection for bot token / channel / guild
        async def slack_send_message(
            message: Annotated[str, "Message to send to the channel."],
            bot_token: Annotated[str, Depends(on(bot_token))],
            channel_id: Annotated[str, Depends(on(channel_id))],
        ) -> Any:
            """
            Sends a message to a Slack channel.

            Args:
                message: The message to send to the channel.
                bot_token: The bot token to use for Slack. (uses dependency injection)
                channel_id: The ID of the channel. (uses dependency injection)
            """
            try:
                web_client = WebClient(token=bot_token)

                # Send the message
                if len(message) > MAX_MESSAGE_LENGTH:
                    chunks = [
                        message[i : i + (MAX_MESSAGE_LENGTH - 1)]
                        for i in range(0, len(message), (MAX_MESSAGE_LENGTH - 1))
                    ]
                    for i, chunk in enumerate(chunks):
                        response = web_client.chat_postMessage(channel=channel_id, text=chunk)

                        if not response["ok"]:
                            return f"Message send failed on chunk {i + 1}, Slack response error: {response['error']}"

                        # Store ID for the first chunk
                        if i == 0:
                            sent_message_id = response["ts"]

                    return f"Message sent successfully ({len(chunks)} chunks, first ID: {sent_message_id}):\n{message}"
                else:
                    response = web_client.chat_postMessage(channel=channel_id, text=message)

                    if not response["ok"]:
                        return f"Message send failed, Slack response error: {response['error']}"

                    return f"Message sent successfully (ID: {response['ts']}):\n{message}"
            except SlackApiError as e:
                return f"Message send failed, Slack API exception: {e.response['error']} (See https://api.slack.com/automation/cli/errors#{e.response['error']})"
            except Exception as e:
                return f"Message send failed, exception: {e}"

        super().__init__(
            name="slack_send",
            description="Sends a message to a Slack channel.",
            func_or_tool=slack_send_message,
        )


@require_optional_import(["slack_sdk"], "commsagent-slack")
@export_module("autogen.tools.experimental")
class SlackRetrieveTool(Tool):
    """Retrieves messages from a Slack channel."""

    def __init__(self, *, bot_token: str, channel_id: str) -> None:
        """
        Initialize the SlackRetrieveTool.

        Args:
            bot_token: Bot User OAuth Token starting with "xoxb-".
            channel_id: Channel ID where messages will be sent.
        """

        async def slack_retrieve_messages(
            bot_token: Annotated[str, Depends(on(bot_token))],
            channel_id: Annotated[str, Depends(on(channel_id))],
            messages_since: Annotated[
                Union[str, None],
                "Date to retrieve messages from (ISO format) OR Slack message ID. If None, retrieves latest messages.",
            ] = None,
            maximum_messages: Annotated[
                Union[int, None], "Maximum number of messages to retrieve. If None, retrieves all messages since date."
            ] = None,
        ) -> Any:
            """
            Retrieves messages from a Discord channel.

            Args:
                bot_token: The bot token to use for Discord. (uses dependency injection)
                channel_id: The ID of the channel. (uses dependency injection)
                messages_since: ISO format date string OR Slack message ID, to retrieve messages from. If None, retrieves latest messages.
                maximum_messages: Maximum number of messages to retrieve. If None, retrieves all messages since date.
            """
            try:
                web_client = WebClient(token=bot_token)

                # Convert ISO datetime to Unix timestamp if needed
                oldest = None
                if messages_since:
                    if "." in messages_since:  # Likely a Slack message ID
                        oldest = messages_since
                    else:  # Assume ISO format
                        try:
                            dt = datetime.fromisoformat(messages_since.replace("Z", "+00:00"))
                            oldest = str(dt.timestamp())
                        except ValueError as e:
                            return f"Invalid date format. Please provide either a Slack message ID or ISO format date (e.g., '2025-01-25T00:00:00Z'). Error: {e}"

                messages = []
                cursor = None

                while True:
                    try:
                        # Prepare API call parameters
                        params = {
                            "channel": channel_id,
                            "limit": min(1000, maximum_messages) if maximum_messages else 1000,
                        }
                        if oldest:
                            params["oldest"] = oldest
                        if cursor:
                            params["cursor"] = cursor

                        # Make API call
                        response = web_client.conversations_history(**params)  # type: ignore[arg-type]

                        if not response["ok"]:
                            return f"Message retrieval failed, Slack response error: {response['error']}"

                        # Add messages to our list
                        messages.extend(response["messages"])

                        # Check if we've hit our maximum
                        if maximum_messages and len(messages) >= maximum_messages:
                            messages = messages[:maximum_messages]
                            break

                        # Check if there are more messages
                        if not response["has_more"]:
                            break

                        cursor = response["response_metadata"]["next_cursor"]

                    except SlackApiError as e:
                        return f"Message retrieval failed on pagination, Slack API error: {e.response['error']}"

                return {
                    "message_count": len(messages),
                    "messages": messages,
                    "start_time": oldest or "latest",
                }

            except SlackApiError as e:
                return f"Message retrieval failed, Slack API exception: {e.response['error']} (See https://api.slack.com/automation/cli/errors#{e.response['error']})"
            except Exception as e:
                return f"Message retrieval failed, exception: {e}"

        super().__init__(
            name="slack_retrieve",
            description="Retrieves messages from a Slack channel based datetime/message ID and/or number of latest messages.",
            func_or_tool=slack_retrieve_messages,
        )
