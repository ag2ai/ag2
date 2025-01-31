import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import telegram
import asyncio
import threading
from typing import Dict, List, Optional, Union, Tuple

from autogen.agentchat.contrib.comms.telegram_agent import TelegramHandler, TelegramConfig
from autogen.agentchat.contrib.comms.platform_errors import (
    PlatformAuthenticationError,
    PlatformConnectionError,
    PlatformRateLimitError,
    PlatformTimeoutError,
)

@pytest.fixture
def telegram_handler():
    config = TelegramConfig(bot_token="mock_test_token", destination_id="mock_channel")
    handler = TelegramHandler(config=config)
    with patch.object(handler, '_telegram_client', new=MagicMock()) as mock_client:
        mock_client.send_message = AsyncMock()
        with patch.object(handler, '_ready', new=asyncio.Event()) as ready_event:
            ready_event.set()
            return handler

@pytest.mark.asyncio
async def test_telegram_handler_start(telegram_handler):
    with patch.object(telegram_handler, '_bot') as mock_bot:
        mock_bot.get_me = AsyncMock(return_value=True)
        assert await telegram_handler.start() is True
        mock_bot.get_me.assert_called_once()

@pytest.mark.asyncio
async def test_telegram_handler_start_auth_error(telegram_handler):
    with patch.object(telegram_handler, '_bot') as mock_bot:
        mock_bot.get_me = AsyncMock(side_effect=telegram.error.InvalidToken())
        with pytest.raises(PlatformAuthenticationError):
            await telegram_handler.start()

@pytest.mark.asyncio
async def test_telegram_handler_send_message(telegram_handler):
    with patch.object(telegram_handler, '_bot') as mock_bot:
        mock_bot.send_message = AsyncMock(return_value=MagicMock(message_id=123))
    result = await telegram_handler.send_message("Test message")
    assert result == ("Message sent successfully", "123")
    telegram_handler._bot.send_message.assert_called_once_with(chat_id="@testchannel", text="Test message", parse_mode="Markdown")

@pytest.mark.asyncio
async def test_telegram_handler_send_long_message(telegram_handler):
    long_message = "x" * 4097  # Telegram's limit is 4096
    with patch.object(telegram_handler, "_bot") as mock_bot:
        mock_bot.send_message = AsyncMock(side_effect=[MagicMock(message_id=123), MagicMock(message_id=124)])
    result = await telegram_handler.send_message(long_message)
    assert result[0] == "Message sent successfully"
    assert mock_bot.send_message.call_count == 2

@pytest.mark.asyncio
async def test_telegram_handler_send_message_rate_limit(telegram_handler):
    with patch.object(telegram_handler, "_bot") as mock_bot:
        mock_bot.send_message = AsyncMock(side_effect=telegram.error.RetryAfter(5))
    with pytest.raises(PlatformRateLimitError) as exc_info:
        await telegram_handler.send_message("Test message")
    assert exc_info.value.retry_after == 5

@pytest.mark.asyncio
async def test_telegram_handler_send_message_network_error(telegram_handler):
    with patch.object(telegram_handler, "_bot") as mock_bot:
        mock_bot.send_message = AsyncMock(side_effect=telegram.error.NetworkError())
    with pytest.raises(PlatformConnectionError):
        await telegram_handler.send_message("Test message")

@pytest.mark.asyncio
async def test_telegram_handler_wait_for_replies(telegram_handler):
    telegram_handler._message_replies["123"] = [{"content": "Reply 1", "author": "User1", "timestamp": "2023-01-01T12:00:00"}, {"content": "Reply 2", "author": "User2", "timestamp": "2023-01-01T12:00:01"}]
    replies = await telegram_handler.wait_for_replies("123", timeout_minutes=1)
    assert len(replies) == 2
    assert replies[0]["content"] == "Reply 1"
    assert replies[1]["content"] == "Reply 2"

@pytest.mark.asyncio
async def test_telegram_handler_wait_for_replies_timeout(telegram_handler):
    with pytest.raises(PlatformTimeoutError):
        await telegram_handler.wait_for_replies("123", timeout_minutes=0.001)

@pytest.mark.asyncio
async def test_telegram_handler_cleanup_monitoring(telegram_handler):
    telegram_handler._message_replies["123"] = [{"content": "Reply"}]
    telegram_handler.cleanup_reply_monitoring("123")
    assert "123" not in telegram_handler._message_replies
