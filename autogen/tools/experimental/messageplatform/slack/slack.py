# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any

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
            message: Annotated[str, "Message to send to the channel"],
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
