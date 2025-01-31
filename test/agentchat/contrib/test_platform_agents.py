import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, List, Optional, Union, Tuple
import asyncio

from autogen import Agent
from autogen.agentchat.contrib.comms.telegram_agent import TelegramAgent, TelegramConfig
from autogen.agentchat.contrib.comms.slack_agent import SlackAgent, SlackConfig
from autogen.agentchat.contrib.comms.discord_agent import DiscordAgent, DiscordConfig
from autogen.agentchat.contrib.comms.platform_configs import ReplyMonitorConfig
from autogen.agentchat.contrib.comms.platform_errors import PlatformError

DEFAULT_TEST_CONFIG = {
    "config_list": [
        {
            "model": "gpt-4",
            "api_key": "fake-key"
        }
    ]
}

@pytest.fixture
def telegram_config():
    return TelegramConfig(bot_token="mock_telegram_token", destination_id="mock_telegram_channel")

@pytest.fixture
def slack_config():
    return SlackConfig(bot_token="mock_slack_token", channel_id="mock_slack_channel", signing_secret="mock_slack_secret")

@pytest.fixture
def discord_config():
    return DiscordConfig(bot_token="mock_discord_token", guild_name="mock_discord_guild", channel_name="mock_discord_channel")

@pytest.mark.asyncio
async def test_telegram_agent_init(telegram_config):
    agent = TelegramAgent(
        name="test_telegram_agent",
        platform_config=telegram_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    assert agent.name == "test_telegram_agent"
    assert isinstance(agent, TelegramAgent)

@pytest.mark.asyncio
async def test_telegram_agent_send_message(telegram_config, mock_sender):
    agent = TelegramAgent(
        name="test_telegram_agent",
        platform_config=telegram_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    with patch.object(agent, "executor_agent") as mock_executor:
        mock_executor.send_to_platform = MagicMock(return_value=("Message sent", "123"))
        result = await agent.a_receive({"role": "user", "content": "Test message"}, mock_sender)
        assert "Message sent" in str(result)
        mock_executor.send_to_platform.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_telegram_agent_async_send(telegram_config, mock_sender):
    agent = TelegramAgent(
        name="test_telegram_agent",
        platform_config=telegram_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    with patch.object(agent, "executor_agent") as mock_executor:
        mock_executor.send_to_platform = MagicMock(return_value=("Message sent", "123"))
        result = await agent.a_receive({"role": "user", "content": "Test message"}, mock_sender)
        assert "Message sent" in str(result)
        mock_executor.send_to_platform.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_slack_agent_init(slack_config):
    agent = SlackAgent(
        name="test_slack_agent",
        platform_config=slack_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    assert agent.name == "test_slack_agent"
    assert isinstance(agent, SlackAgent)

@pytest.mark.asyncio
async def test_slack_agent_send_message(slack_config, mock_sender):
    agent = SlackAgent(
        name="test_slack_agent",
        platform_config=slack_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    with patch.object(agent, "executor_agent") as mock_executor:
        mock_executor.send_to_platform = MagicMock(return_value=("Message sent", "123"))
        result = await agent.a_receive({"role": "user", "content": "Test message"}, mock_sender)
        assert "Message sent" in str(result)
        mock_executor.send_to_platform.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_slack_agent_async_send(slack_config, mock_sender):
    agent = SlackAgent(
        name="test_slack_agent",
        platform_config=slack_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    with patch.object(agent, "executor_agent") as mock_executor:
        mock_executor.send_to_platform = MagicMock(return_value=("Message sent", "123"))
        result = await agent.a_receive({"role": "user", "content": "Test message"}, mock_sender)
        assert "Message sent" in str(result)
        mock_executor.send_to_platform.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_discord_agent_init(discord_config):
    agent = DiscordAgent(
        name="test_discord_agent",
        platform_config=discord_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    assert agent.name == "test_discord_agent"
    assert isinstance(agent, DiscordAgent)

@pytest.mark.asyncio
async def test_discord_agent_send_message(discord_config, mock_sender):
    agent = DiscordAgent(
        name="test_discord_agent",
        platform_config=discord_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    with patch.object(agent, "executor_agent") as mock_executor:
        mock_executor.send_to_platform = MagicMock(return_value=("Message sent", "123"))
        result = await agent.a_receive({"role": "user", "content": "Test message"}, mock_sender)
        assert "Message sent" in str(result)
        mock_executor.send_to_platform.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_discord_agent_async_send(discord_config, mock_sender):
    agent = DiscordAgent(
        name="test_discord_agent",
        platform_config=discord_config,
        llm_config=DEFAULT_TEST_CONFIG
    )
    with patch.object(agent, "executor_agent") as mock_executor:
        mock_executor.send_to_platform = MagicMock(return_value=("Message sent", "123"))
        result = await agent.a_receive({"role": "user", "content": "Test message"}, mock_sender)
        assert "Message sent" in str(result)
        mock_executor.send_to_platform.assert_called_once_with("Test message")
