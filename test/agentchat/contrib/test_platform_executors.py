import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, List, Optional, Union, Tuple
import asyncio

from autogen.agentchat.contrib.comms.telegram_agent import TelegramExecutor, TelegramConfig
from autogen.agentchat.contrib.comms.slack_agent import SlackExecutor, SlackConfig
from autogen.agentchat.contrib.comms.discord_agent import DiscordExecutor, DiscordConfig
from autogen.agentchat.contrib.comms.platform_errors import (
    PlatformError,
    PlatformAuthenticationError,
    PlatformConnectionError,
    PlatformRateLimitError,
    PlatformTimeoutError,
)

@pytest.fixture
def mock_handler(request):
    handler = MagicMock()
    handler.send_message = AsyncMock(return_value=("Message sent", "123"))
    handler.wait_for_replies = AsyncMock(return_value=[{"content": "Reply"}])
    handler.cleanup_reply_monitoring = MagicMock()
    handler.start = AsyncMock(return_value=True)
    handler.validate = AsyncMock(return_value=True)
    return handler

@pytest.fixture
def telegram_handler():
    handler = MagicMock()
    handler.send_message = AsyncMock(return_value=("Message sent", "123"))
    handler.wait_for_replies = AsyncMock(return_value=[{"content": "Reply"}])
    handler.cleanup_reply_monitoring = MagicMock()
    handler.start = AsyncMock(return_value=True)
    handler.validate = AsyncMock(return_value=True)
    handler._ready = MagicMock()
    handler._ready.wait = AsyncMock(return_value=True)
    handler._error = None
    return handler

@pytest.fixture
def telegram_executor(monkeypatch):
    handler = MagicMock()
    handler.send_message = AsyncMock(return_value=("Message sent", "123"))
    handler.wait_for_replies = AsyncMock(return_value=[{"content": "Reply"}])
    handler.cleanup_reply_monitoring = MagicMock()
    handler.start = MagicMock(return_value=True)
    handler.validate = AsyncMock(return_value=True)
    
    def mock_init(*args, **kwargs):
        return handler
    
    monkeypatch.setattr("autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler", mock_init)
    
    config = TelegramConfig(bot_token="123456:ABC-DEF", destination_id="@testchannel")
    executor = TelegramExecutor(platform_config=config)
    return executor

@pytest.fixture
def slack_handler():
    handler = MagicMock()
    handler.send_message = AsyncMock(return_value=("Message sent", "123"))
    handler.wait_for_replies = AsyncMock(return_value=[{"content": "Reply"}])
    handler.cleanup_reply_monitoring = MagicMock()
    handler.start = AsyncMock(return_value=True)
    handler.validate = AsyncMock(return_value=True)
    handler._ready = MagicMock()
    handler._ready.wait = AsyncMock(return_value=True)
    handler._error = None
    return handler

@pytest.fixture
def slack_executor(monkeypatch):
    handler = MagicMock()
    handler.send_message = AsyncMock(return_value=("Message sent", "123"))
    handler.wait_for_replies = AsyncMock(return_value=[{"content": "Reply"}])
    handler.cleanup_reply_monitoring = MagicMock()
    handler.start = MagicMock(return_value=True)
    handler.validate = AsyncMock(return_value=True)
    
    def mock_init(*args, **kwargs):
        return handler
    
    monkeypatch.setattr("autogen.agentchat.contrib.comms.slack_agent.SlackHandler", mock_init)
    
    config = SlackConfig(bot_token="xoxb-fake", app_token="xapp-fake", channel_id="ABC123", signing_secret="secret")
    executor = SlackExecutor(platform_config=config)
    return executor

@pytest.fixture
def discord_handler():
    handler = MagicMock()
    handler.send_message = AsyncMock(return_value=("Message sent", "123"))
    handler.wait_for_replies = AsyncMock(return_value=[{"content": "Reply"}])
    handler.cleanup_reply_monitoring = MagicMock()
    handler.start = AsyncMock(return_value=True)
    handler.validate = AsyncMock(return_value=True)
    handler._ready = MagicMock()
    handler._ready.wait = AsyncMock(return_value=True)
    handler._error = None
    return handler

@pytest.fixture
def discord_executor(monkeypatch):
    handler = MagicMock()
    handler.send_message = AsyncMock(return_value=("Message sent", "123"))
    handler.wait_for_replies = AsyncMock(return_value=[{"content": "Reply"}])
    handler.cleanup_reply_monitoring = MagicMock()
    handler.start = MagicMock(return_value=True)
    handler.validate = AsyncMock(return_value=True)
    
    def mock_init(*args, **kwargs):
        return handler
    
    monkeypatch.setattr("autogen.agentchat.contrib.comms.discord_agent.DiscordHandler", mock_init)
    
    config = DiscordConfig(bot_token="fake_token", guild_name="TestGuild", channel_name="TestChannel")
    executor = DiscordExecutor(platform_config=config)
    return executor



@pytest.mark.asyncio
async def test_telegram_executor_send_to_platform(telegram_executor):
    result = await telegram_executor.send_to_platform("Test message")
    assert result == ("Message sent", "123")
    telegram_executor._telegram.send_message.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_telegram_executor_wait_for_reply(telegram_executor):
    replies = await telegram_executor.wait_for_reply("123")
    assert replies == [{"content": "Reply"}]
    telegram_executor._telegram.wait_for_replies.assert_called_once_with("123", timeout_minutes=1)

@pytest.mark.asyncio
async def test_telegram_executor_error_handling(telegram_executor):
    telegram_executor._telegram.send_message = AsyncMock(
        side_effect=PlatformError("Test error", "Telegram")
    )
    with pytest.raises(PlatformError) as exc_info:
        await telegram_executor.send_to_platform("Test message")
    assert "Test error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_slack_executor_send_to_platform(slack_executor):
    result = await slack_executor.send_to_platform("Test message")
    assert result == ("Message sent", "123")
    slack_executor._slack.send_message.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_slack_executor_wait_for_reply(slack_executor):
    replies = await slack_executor.wait_for_reply("123")
    assert replies == [{"content": "Reply"}]
    slack_executor._slack.wait_for_replies.assert_called_once_with("123", timeout_minutes=1)

@pytest.mark.asyncio
async def test_slack_executor_error_handling(slack_executor):
    slack_executor._slack.send_message = AsyncMock(
        side_effect=PlatformError("Test error", "Slack")
    )
    with pytest.raises(PlatformError) as exc_info:
        await slack_executor.send_to_platform("Test message")
    assert "Test error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_discord_executor_send_to_platform(discord_executor):
    result = await discord_executor.send_to_platform("Test message")
    assert result == ("Message sent", "123")
    discord_executor._discord.send_message.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_discord_executor_wait_for_reply(discord_executor):
    replies = await discord_executor.wait_for_reply("123")
    assert replies == [{"content": "Reply"}]
    discord_executor._discord.wait_for_replies.assert_called_once_with("123", timeout_minutes=1)

@pytest.mark.asyncio
async def test_discord_executor_error_handling(discord_executor):
    discord_executor._discord.send_message = AsyncMock(
        side_effect=PlatformError("Test error", "Discord")
    )
    with pytest.raises(PlatformError) as exc_info:
        await discord_executor.send_to_platform("Test message")
    assert "Test error" in str(exc_info.value)
