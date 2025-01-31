import pytest
from unittest.mock import MagicMock, AsyncMock
import discord
import asyncio

from typing import Dict, List, Optional, Union, Tuple

from autogen.agentchat.contrib.comms.discord_agent import DiscordHandler
from autogen.agentchat.contrib.comms.platform_errors import (
    PlatformAuthenticationError,
    PlatformConnectionError,
    PlatformRateLimitError,
    PlatformTimeoutError,
)

@pytest.fixture
def discord_handler():
    handler = DiscordHandler(bot_token="fake_token", guild_name="TestGuild", channel_name="TestChannel")
    mock_client = MagicMock()
    mock_client.login = AsyncMock()
    mock_client.start = AsyncMock()
    mock_channel = MagicMock()
    
    # Set up handler attributes through __dict__ to avoid descriptor issues
    handler.__dict__['_client'] = mock_client
    handler.__dict__['_channel'] = mock_channel
    handler.__dict__['_loop'] = asyncio.new_event_loop()
    handler.__dict__['_thread'] = threading.current_thread()
    handler.__dict__['_message_replies'] = {}
    handler.__dict__['silent'] = False
    return handler

@pytest.mark.asyncio
async def test_discord_handler_start(discord_handler):
    discord_handler._client.login = AsyncMock(return_value=True)
    discord_handler._client.start = AsyncMock(return_value=True)
    assert await discord_handler.start() is True
    discord_handler._client.login.assert_called_once_with("fake_token")
    discord_handler._client.start.assert_called_once()

@pytest.mark.asyncio
async def test_discord_handler_start_auth_error(discord_handler):
    discord_handler._client.login = AsyncMock(side_effect=discord.LoginFailure())
    with pytest.raises(PlatformAuthenticationError):
        await discord_handler.start()

@pytest.mark.asyncio
async def test_discord_handler_send_message(discord_handler):
    mock_message = MagicMock()
    mock_message.id = 123
    discord_handler._channel.send = AsyncMock(return_value=mock_message)
    
    result = await discord_handler.send_message("Test message")
    assert result == ("Message sent successfully", "123")
    discord_handler._channel.send.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_discord_handler_send_long_message(discord_handler):
    long_message = "x" * 2001  # Discord's limit is 2000
    mock_message1 = MagicMock(id=123)
    mock_message2 = MagicMock(id=124)
    
    discord_handler._channel.send = AsyncMock(side_effect=[mock_message1, mock_message2])
    
    result = await discord_handler.send_message(long_message)
    assert result[0] == "Message sent successfully"
    assert discord_handler._channel.send.call_count == 2

@pytest.mark.asyncio
async def test_discord_handler_send_message_rate_limit(discord_handler):
    discord_handler._channel.send = AsyncMock(side_effect=discord.RateLimited(5.0))
    with pytest.raises(PlatformRateLimitError) as exc_info:
        await discord_handler.send_message("Test message")
    assert exc_info.value.retry_after == 5

@pytest.mark.asyncio
async def test_discord_handler_send_message_connection_error(discord_handler):
    discord_handler._channel.send = AsyncMock(side_effect=discord.ConnectionClosed())
    with pytest.raises(PlatformConnectionError):
        await discord_handler.send_message("Test message")

@pytest.mark.asyncio
async def test_discord_handler_wait_for_replies(discord_handler):
    discord_handler._message_replies["123"] = [{"content": "Reply 1", "author": "User1", "timestamp": "2023-01-01T12:00:00"}, {"content": "Reply 2", "author": "User2", "timestamp": "2023-01-01T12:00:01"}]
    replies = await discord_handler.wait_for_replies("123", timeout_minutes=1)
    assert len(replies) == 2
    assert replies[0]["content"] == "Reply 1"
    assert replies[1]["content"] == "Reply 2"

@pytest.mark.asyncio
async def test_discord_handler_wait_for_replies_timeout(discord_handler):
    with pytest.raises(PlatformTimeoutError):
        await discord_handler.wait_for_replies("123", timeout_minutes=0.001)

@pytest.mark.asyncio
async def test_discord_handler_cleanup_monitoring(discord_handler):
    discord_handler._message_replies["123"] = [{"content": "Reply"}]
    discord_handler.cleanup_reply_monitoring("123")
    assert "123" not in discord_handler._message_replies
