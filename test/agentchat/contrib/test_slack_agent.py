import pytest
from unittest.mock import MagicMock
from typing import Dict, List, Optional, Union
from autogen import Agent

from autogen.agentchat.contrib.comms.slack_agent import SlackAgent, SlackConfig, SlackExecutor
from autogen.agentchat.contrib.comms.platform_configs import ReplyMonitorConfig
from autogen.agentchat.contrib.comms.platform_errors import (
    PlatformAuthenticationError,
    PlatformConnectionError,
    PlatformError,
    PlatformRateLimitError,
    PlatformTimeoutError,
)

@pytest.mark.asyncio
async def test_slack_config_validation():
    # Test valid configuration
    config = SlackConfig(bot_token="xoxb-fake", channel_id="ABC123", signing_secret="secret")
    assert config.validate_config() is True

    # Test missing required fields
    with pytest.raises(ValueError, match="bot_token, channel_id, and signing_secret are required"):
        SlackConfig(bot_token="", channel_id="", signing_secret="").validate_config()

    # Test invalid bot token format
    with pytest.raises(ValueError, match="bot_token must start with 'xoxb-'"):
        SlackConfig(bot_token="invalid", channel_id="ABC123", signing_secret="secret").validate_config()

@pytest.mark.asyncio
async def test_slack_agent_initialization(slack_config):
    agent = SlackAgent(name="test_slack_agent", platform_config=slack_config)
    assert agent.name == "test_slack_agent"
    assert isinstance(agent._platform_config, SlackConfig)
    assert isinstance(agent.executor_agent, SlackExecutor)

@pytest.mark.asyncio
async def test_slack_agent_invalid_token(mocker, slack_config):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.slack_agent.SlackHandler.start")
    mock_handler.side_effect = PlatformAuthenticationError(
        message="Invalid bot token",
        platform_name="Slack"
    )

    agent = SlackAgent(name="test_slack_agent", platform_config=slack_config)

    with pytest.raises(PlatformAuthenticationError) as exc_info:
        agent.executor_agent.send_to_platform("test message")
    assert "Invalid bot token" in str(exc_info.value)

@pytest.mark.asyncio
async def test_slack_agent_invalid_channel(mocker):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.slack_agent.SlackHandler.start")
    mock_handler.side_effect = PlatformConnectionError(
        message="Channel not found: INVALID",
        platform_name="Slack"
    )

    config = SlackConfig(bot_token="xoxb-fake", channel_id="INVALID", signing_secret="secret")
    agent = SlackAgent(name="test_slack_agent", platform_config=config)

    with pytest.raises(PlatformConnectionError) as exc_info:
        agent.executor_agent.send_to_platform("test message")
    assert "Channel not found" in str(exc_info.value)

@pytest.mark.asyncio
async def test_slack_agent_invalid_signing_secret(mocker):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.slack_agent.SlackHandler.start")
    mock_handler.side_effect = PlatformAuthenticationError(
        message="Invalid signing secret",
        platform_name="Slack"
    )

    config = SlackConfig(bot_token="xoxb-fake", channel_id="ABC123", signing_secret="invalid")
    agent = SlackAgent(name="test_slack_agent", platform_config=config)

    with pytest.raises(PlatformAuthenticationError) as exc_info:
        agent.executor_agent.send_to_platform("test message")
    assert "Invalid signing secret" in str(exc_info.value)

@pytest.mark.asyncio
async def test_slack_agent_send_message_sync_and_async(mocker, slack_config, mock_slack_handler, mock_sender):
    agent = SlackAgent(name="test_slack_agent", platform_config=slack_config)

    # Test synchronous operation through executor agent
    response = agent.executor_agent.send_to_platform("Hello Slack!")
    assert response[0] == "Message sent successfully"
    assert response[1] == "123456789"
    mock_slack_handler.assert_called_once()

    # Test asynchronous operation through nested chat sequence
    test_message: Dict[str, str] = {"role": "user", "content": "Send this message to Slack"}
    async_response = await agent.a_receive(test_message, mock_sender)
    assert isinstance(async_response, str)
    assert "Message sent successfully" in async_response

@pytest.mark.asyncio
async def test_slack_agent_send_long_message(mocker, mock_platform_agent):
    config = SlackConfig(bot_token="xoxb-fake", channel_id="ABC123", signing_secret="secret")
    agent = SlackAgent(name="test_slack_agent", platform_config=config)
    mock_platform_agent

    long_message = "x" * 4001  # Slack's message limit is 4000 characters
    response = agent.executor_agent.send_to_platform(long_message)
    assert response[0] == "Message sent successfully"
    assert response[1] == "123.456"
    assert mock_platform_agent.call_count == 2

@pytest.mark.asyncio
async def test_slack_agent_send_message_rate_limit(mocker, mock_platform_agent):
    mock_platform_agent.side_effect = PlatformRateLimitError(
        message="Rate limit exceeded",
        platform_name="Slack",
        retry_after=5
    )

    config = SlackConfig(bot_token="xoxb-fake", channel_id="ABC123", signing_secret="secret")
    agent = SlackAgent(name="test_slack_agent", platform_config=config)

    with pytest.raises(PlatformRateLimitError) as exc_info:
        agent.executor_agent.send_to_platform("Test message")
    assert "Rate limit exceeded" in str(exc_info.value)
    assert exc_info.value.retry_after == 5

@pytest.mark.asyncio
async def test_slack_agent_send_message_network_error(mocker):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.slack_agent.SlackHandler.send_message")
    mock_handler.side_effect = PlatformConnectionError(
        message="Network connection failed",
        platform_name="Slack"
    )

    config = SlackConfig(bot_token="xoxb-fake", channel_id="ABC123", signing_secret="secret")
    agent = SlackAgent(name="test_slack_agent", platform_config=config)

    with pytest.raises(PlatformConnectionError) as exc_info:
        agent.executor_agent.send_to_platform("Test message")
    assert "Network connection failed" in str(exc_info.value)

@pytest.mark.asyncio
async def test_slack_agent_wait_for_replies(mocker, mock_wait_handler):
    config = SlackConfig(bot_token="xoxb-fake", channel_id="ABC123", signing_secret="secret")
    agent = SlackAgent(
        name="test_slack_agent",
        platform_config=config,
        reply_monitor_config=ReplyMonitorConfig(timeout_minutes=1)
    )
    replies = agent.executor_agent.wait_for_reply("msg_ts")
    assert len(replies) == 1
    assert "Hi" in replies[0]
    mock_wait_handler.assert_called_once_with("msg_ts", timeout_minutes=1)

@pytest.mark.asyncio
async def test_slack_agent_wait_for_replies_with_timeout(mocker, slack_config, reply_monitor_config, mock_wait_handler):
    mock_wait_handler.side_effect = PlatformTimeoutError(
        message="Timeout waiting for replies after 1 minutes",
        platform_name="Slack"
    )

    agent = SlackAgent(
        name="test_slack_agent",
        platform_config=slack_config,
        reply_monitor_config=reply_monitor_config
    )
    with pytest.raises(PlatformTimeoutError) as exc_info:
        agent.executor_agent.wait_for_reply("msg_ts")
    assert "Timeout waiting for replies" in str(exc_info.value)

@pytest.mark.asyncio
async def test_slack_agent_wait_for_replies_max_replies(mocker):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.slack_agent.SlackHandler.wait_for_replies")
    mock_handler.return_value = [
        {"content": "Reply 1", "author": "User1", "timestamp": "2023-01-01T12:00:00", "id": "123.456"},
        {"content": "Reply 2", "author": "User2", "timestamp": "2023-01-01T12:00:01", "id": "123.457"}
    ]

    config = SlackConfig(bot_token="xoxb-fake", channel_id="ABC123", signing_secret="secret")
    agent = SlackAgent(
        name="test_slack_agent",
        platform_config=config,
        reply_monitor_config=ReplyMonitorConfig(timeout_minutes=1, max_reply_messages=2)
    )
    replies = agent.executor_agent.wait_for_reply("msg_ts")
    assert len(replies) == 2
    mock_handler.assert_called_once_with("msg_ts", timeout_minutes=1, max_reply_messages=2)

@pytest.mark.asyncio
async def test_slack_agent_wait_for_replies_filter_author(mocker):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.slack_agent.SlackHandler.wait_for_replies")
    mock_handler.return_value = [
        {"content": "Hi from human", "author": "HumanUser", "timestamp": "2023-01-01T12:00:00", "id": "123.456"},
        {"content": "Hi from bot", "author": "BotUser", "timestamp": "2023-01-01T12:00:01", "id": "123.457"}
    ]

    config = SlackConfig(bot_token="xoxb-fake", channel_id="ABC123", signing_secret="secret")
    agent = SlackAgent(
        name="test_slack_agent",
        platform_config=config,
        reply_monitor_config=ReplyMonitorConfig(timeout_minutes=1)
    )
    replies = agent.executor_agent.wait_for_reply("msg_ts")
    assert len(replies) == 1
    assert "Hi from human" in replies[0]
    mock_handler.assert_called_once_with("msg_ts", timeout_minutes=1, allowed_authors=["HumanUser"])

@pytest.mark.asyncio
async def test_slack_agent_cleanup_monitoring(mocker):
    mock_cleanup = mocker.patch("autogen.agentchat.contrib.comms.slack_agent.SlackHandler.cleanup_reply_monitoring")
    
    config = SlackConfig(bot_token="xoxb-fake", channel_id="ABC123", signing_secret="secret")
    agent = SlackAgent(
        name="test_slack_agent",
        platform_config=config,
        reply_monitor_config=ReplyMonitorConfig(timeout_minutes=1)
    )
    
    agent.executor_agent.cleanup_monitoring("msg_ts")
    mock_cleanup.assert_called_once_with("msg_ts")
