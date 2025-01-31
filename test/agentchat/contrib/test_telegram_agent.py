import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, List, Optional, Union
import telegram
import asyncio
import threading
from autogen import Agent

DEFAULT_TEST_CONFIG = {
    "config_list": [
        {
            "model": "gpt-4",
            "api_key": "fake-key"
        }
    ]
}

from autogen.agentchat.contrib.comms.telegram_agent import (
    TelegramAgent,
    TelegramConfig,
    TelegramExecutor,
)
from autogen.agentchat.contrib.comms.platform_configs import ReplyMonitorConfig
from autogen.agentchat.contrib.comms.platform_errors import (
    PlatformAuthenticationError,
    PlatformConnectionError,
    PlatformError,
    PlatformRateLimitError,
    PlatformTimeoutError,
)

@pytest.mark.asyncio
async def test_telegram_config_validation():
    config = TelegramConfig(bot_token="test-token", destination_id="test-channel")
    assert config.validate_config() is True

    with pytest.raises(ValueError, match="bot_token is required"):
        TelegramConfig(bot_token="", destination_id="@testchannel").validate_config()

    with pytest.raises(ValueError, match="destination_id is required"):
        TelegramConfig(bot_token="123456:ABC-DEF", destination_id="").validate_config()

@pytest.mark.asyncio
async def test_telegram_agent_initialization(mocker, telegram_config):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler")
    handler = mock_handler.return_value
    handler.start = AsyncMock(return_value=True)
    handler.initialize = AsyncMock()
    handler.send_message = AsyncMock(return_value=("Message sent successfully", "123456789"))
    handler._message_replies = {}
    await handler.initialize()
    
    agent = TelegramAgent(name="test_telegram_agent", platform_config=telegram_config, llm_config=DEFAULT_TEST_CONFIG)
    assert agent.name == "test_telegram_agent"
    assert isinstance(agent.executor_agent, TelegramExecutor)
    mock_handler.assert_called_once()

@pytest.mark.asyncio
async def test_telegram_agent_invalid_token(mocker, telegram_config):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler")
    handler = mock_handler.return_value
    handler.start = AsyncMock(return_value=True)
    handler.initialize = AsyncMock()
    handler.send_message = AsyncMock(side_effect=telegram.error.InvalidToken())
    
    agent = TelegramAgent(name="test_telegram_agent", platform_config=telegram_config, llm_config=DEFAULT_TEST_CONFIG)
    await handler.initialize()
    
    agent.executor_agent.send_to_platform = AsyncMock(side_effect=PlatformAuthenticationError("Invalid bot token", "Telegram"))
    with pytest.raises(PlatformAuthenticationError) as exc_info:
        await agent.executor_agent.send_to_platform("test message")
    assert "Invalid bot token" in str(exc_info.value)
    assert exc_info.value.platform_name == "Telegram"
    assert handler.send_message.call_count == 1

@pytest.mark.asyncio
async def test_telegram_agent_invalid_destination(mocker, telegram_config):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler")
    handler = mock_handler.return_value
    handler.start = AsyncMock(return_value=True)
    handler.initialize = AsyncMock()
    handler.send_message = AsyncMock(side_effect=telegram.error.BadRequest("Chat not found"))
    
    agent = TelegramAgent(name="test_telegram_agent", platform_config=telegram_config, llm_config=DEFAULT_TEST_CONFIG)
    await handler.initialize()
    
    agent.executor_agent.send_to_platform = AsyncMock(side_effect=PlatformConnectionError("Chat not found", "Telegram"))
    with pytest.raises(PlatformConnectionError) as exc_info:
        await agent.executor_agent.send_to_platform("test message")
    assert "Chat not found" in str(exc_info.value)
    assert exc_info.value.platform_name == "Telegram"
    assert handler.send_message.call_count == 1

@pytest.mark.asyncio
async def test_telegram_agent_send_message(mocker, telegram_config):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler")
    handler = mock_handler.return_value
    handler.start = AsyncMock(return_value=True)
    handler.initialize = AsyncMock()
    handler.send_message = AsyncMock(return_value=("Message sent successfully", "123456789"))
    
    agent = TelegramAgent(name="test_telegram_agent", platform_config=telegram_config, llm_config=DEFAULT_TEST_CONFIG)
    await handler.initialize()
    
    agent.executor_agent.send_to_platform = AsyncMock(return_value=("Message sent successfully", "123456789"))
    response = await agent.executor_agent.send_to_platform("Hello Telegram!")
    assert response[0] == "Message sent successfully"
    assert response[1] == "123456789"
    handler.send_message.assert_called_once_with("Hello Telegram!")

@pytest.mark.asyncio
async def test_telegram_agent_send_long_message(mocker, telegram_config):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler")
    handler = mock_handler.return_value
    handler.start = AsyncMock(return_value=True)
    handler.initialize = AsyncMock()
    handler.send_message = AsyncMock(return_value=("Message sent successfully", "123"))
    
    agent = TelegramAgent(name="test_telegram_agent", platform_config=telegram_config, llm_config=DEFAULT_TEST_CONFIG)
    await handler.initialize()
    
    agent.executor_agent.send_to_platform = AsyncMock(return_value=("Message sent successfully", "123"))
    long_message = "x" * 4097  # Telegram's limit is 4096
    response = await agent.executor_agent.send_to_platform(long_message)
    assert response[0] == "Message sent successfully"
    assert response[1] == "123"
    handler.send_message.assert_called_once_with(long_message)

@pytest.mark.asyncio
async def test_telegram_agent_wait_for_replies(mocker, telegram_config):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler")
    handler = mock_handler.return_value
    handler.start = AsyncMock(return_value=True)
    handler.initialize = AsyncMock()
    handler.wait_for_replies = AsyncMock(return_value=[
        {"content": "Reply 1", "author": "User1", "timestamp": "2023-01-01T12:00:00", "id": "1"},
        {"content": "Reply 2", "author": "User2", "timestamp": "2023-01-01T12:00:01", "id": "2"}
    ])
    
    agent = TelegramAgent(
        name="test_telegram_agent",
        platform_config=telegram_config,
        reply_monitor_config=ReplyMonitorConfig(timeout_minutes=1, max_reply_messages=2),
        llm_config=DEFAULT_TEST_CONFIG
    )
    await agent.initialize()
    
    agent.executor_agent.wait_for_reply = AsyncMock(return_value=[
        "[2023-01-01 12:00 UTC] User1: Reply 1",
        "[2023-01-01 12:00 UTC] User2: Reply 2"
    ])
    replies = await agent.executor_agent.wait_for_reply("123456789")
    assert len(replies) == 2
    assert "[2023-01-01 12:00 UTC] User1: Reply 1" == replies[0]
    assert "[2023-01-01 12:00 UTC] User2: Reply 2" == replies[1]
    handler.wait_for_replies.assert_called_once_with("123456789", timeout_minutes=1, max_replies=2)

@pytest.mark.asyncio
async def test_telegram_agent_wait_for_replies_timeout(mocker, telegram_config):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler")
    handler = mock_handler.return_value
    handler.start = AsyncMock(return_value=True)
    handler.initialize = AsyncMock()
    handler.wait_for_replies = AsyncMock(side_effect=asyncio.TimeoutError())
    
    agent = TelegramAgent(
        name="test_telegram_agent",
        platform_config=telegram_config,
        reply_monitor_config=ReplyMonitorConfig(timeout_minutes=1, max_reply_messages=2),
        llm_config=DEFAULT_TEST_CONFIG
    )
    await agent.initialize()
    
    agent.executor_agent.wait_for_reply = AsyncMock(side_effect=PlatformTimeoutError("Timeout waiting for replies", "Telegram"))
    with pytest.raises(PlatformTimeoutError) as exc_info:
        await agent.executor_agent.wait_for_reply("123456789")
    assert "Timeout waiting for replies" in str(exc_info.value)
    assert exc_info.value.platform_name == "Telegram"
    handler.wait_for_replies.assert_called_once_with("123456789", timeout_minutes=1, max_replies=2)

@pytest.mark.asyncio
async def test_telegram_agent_wait_for_replies_with_max_replies(mocker, telegram_config):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler")
    handler = mock_handler.return_value
    handler.start = AsyncMock(return_value=True)
    handler.initialize = AsyncMock()
    handler.wait_for_replies = AsyncMock(return_value=[
        {"content": "Reply 1", "author": "User1", "timestamp": "2023-01-01T12:00:00", "id": "1"},
        {"content": "Reply 2", "author": "User2", "timestamp": "2023-01-01T12:00:01", "id": "2"}
    ])
    
    agent = TelegramAgent(
        name="test_telegram_agent",
        platform_config=telegram_config,
        reply_monitor_config=ReplyMonitorConfig(timeout_minutes=1, max_reply_messages=1),
        llm_config=DEFAULT_TEST_CONFIG
    )
    await agent.initialize()
    
    replies = await agent.executor_agent.wait_for_reply("123456789")
    assert len(replies) == 1
    assert "[2023-01-01 12:00 UTC] User1: Reply 1" == replies[0]
    handler.wait_for_replies.assert_called_once_with("123456789", timeout_minutes=1, max_replies=1)

@pytest.mark.asyncio
async def test_telegram_agent_rate_limit(mocker, telegram_config):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler")
    handler = mock_handler.return_value
    handler.start = AsyncMock(return_value=True)
    handler.initialize = AsyncMock()
    handler.send_message = AsyncMock(side_effect=telegram.error.RetryAfter(5))
    
    agent = TelegramAgent(name="test_telegram_agent", platform_config=telegram_config, llm_config=DEFAULT_TEST_CONFIG)
    await handler.initialize()
    
    with pytest.raises(PlatformRateLimitError) as exc_info:
        await agent.executor_agent.send_to_platform("test message")
    assert exc_info.value.retry_after == 5
    assert exc_info.value.platform_name == "Telegram"
    assert handler.send_message.call_count == 1

@pytest.mark.asyncio
async def test_telegram_agent_network_error(mocker, telegram_config):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler")
    handler = mock_handler.return_value
    handler.start = AsyncMock(return_value=True)
    handler.initialize = AsyncMock()
    handler.send_message = AsyncMock(side_effect=telegram.error.NetworkError("Network error occurred"))
    
    agent = TelegramAgent(name="test_telegram_agent", platform_config=telegram_config, llm_config=DEFAULT_TEST_CONFIG)
    await handler.initialize()
    
    with pytest.raises(PlatformConnectionError) as exc_info:
        await agent.executor_agent.send_to_platform("test message")
    assert "Network error occurred" in str(exc_info.value)
    assert exc_info.value.platform_name == "Telegram"
    assert handler.send_message.call_count == 1
