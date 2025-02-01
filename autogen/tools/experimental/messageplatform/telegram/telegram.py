# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Any

from .....doc_utils import export_module
from .....import_utils import optional_import_block, require_optional_import
from .... import Tool
from ....dependency_injection import Depends, on

__all__ = ["TelegramSendTool"]

with optional_import_block():
    from telegram import Bot
    from telegram.error import TelegramError


MAX_MESSAGE_LENGTH = 4096


@require_optional_import(["telegram"], "commsagent-telegram")
@export_module("autogen.tools.experimental")
class TelegramSendTool(Tool):
    """Sends a message to a Telegram channel."""

    def __init__(self, *, bot_token: str, chat_id: str) -> None:
        """
        Initialize the TelegramSendTool.

        Args:
            bot_token: Bot token from BotFather (starts with numbers:ABC...).
            chat_id: Bot's channel Id, Group Id with bot in it, or Channel with bot in it
        """
        self._bot = Bot(token=bot_token)

        async def telegram_send_message(
            message: Annotated[str, "Message to send to the channel"],
            chat_id: Annotated[str, Depends(on(chat_id))],
        ) -> Any:
            """
            Sends a message to a Telegram Bot, Group, or Channel (based on the destination_id).

            Args:
                message: The message to send to the channel.
                bot_token: The bot token to use for Telegram. (uses dependency injection)
                chat_id: The ID of the destination. (uses dependency injection)
            """
            try:
                # Send the message
                if len(message) > MAX_MESSAGE_LENGTH:
                    chunks = [
                        message[i : i + (MAX_MESSAGE_LENGTH - 1)]
                        for i in range(0, len(message), (MAX_MESSAGE_LENGTH - 1))
                    ]
                    first_message = None

                    for i, chunk in enumerate(chunks):
                        sent = await self._bot.send_message(
                            chat_id=chat_id,
                            text=chunk,
                            parse_mode="HTML",
                            reply_to_message_id=first_message.message_id if first_message else None,  # type: ignore
                        )

                        # Store the first message to chain replies
                        if i == 0:
                            first_message = sent
                            sent_message_id = str(sent.message_id)

                    return f"Message sent successfully ({len(chunks)} chunks, first ID: {sent_message_id}):\n{message}"
                else:
                    sent = await self._bot.send_message(chat_id=chat_id, text=message, parse_mode="HTML")
                    return f"Message sent successfully (ID: {sent.message_id}):\n{message}"

            except TelegramError as e:
                return f"Message send failed, Telegram API error: {str(e)}"
            except Exception as e:
                return f"Message send failed, exception: {str(e)}"

        super().__init__(
            name="telegram_send",
            description="Sends a message to a Telegram bot, group, or channel.",
            func_or_tool=telegram_send_message,
        )
