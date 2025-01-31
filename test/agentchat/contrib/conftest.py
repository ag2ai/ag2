import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, List, Optional, Union, Tuple

from autogen import Agent
from autogen.agentchat.contrib.comms.discord_agent import DiscordConfig
from autogen.agentchat.contrib.comms.slack_agent import SlackConfig
from autogen.agentchat.contrib.comms.telegram_agent import TelegramConfig, TelegramHandler
from autogen.agentchat.contrib.comms.platform_configs import ReplyMonitorConfig

@pytest.fixture(scope="function", autouse=True)
async def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture
def mock_telegram_handler(mocker):
    mock = mocker.patch("autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler")
    handler = mock.return_value
    handler.start = AsyncMock(return_value=True)
    handler.initialize = AsyncMock()
    handler.send_message = AsyncMock(return_value=("Message sent successfully", "123456789"))
    handler.wait_for_replies = AsyncMock(return_value=[])
    handler.cleanup = AsyncMock()
    handler._message_replies = {}
    return mock

@pytest.fixture
def discord_config() -> DiscordConfig:
    return DiscordConfig(bot_token="mock-token", guild_name="mock-guild", channel_name="mock-channel")

@pytest.fixture
def slack_config() -> SlackConfig:
    return SlackConfig(bot_token="mock-token", channel_id="mock-channel", signing_secret="mock-secret")

@pytest.fixture
def telegram_config() -> TelegramConfig:
    return TelegramConfig(bot_token="mock-token", destination_id="mock-destination")

@pytest.fixture
def reply_monitor_config() -> ReplyMonitorConfig:
    return ReplyMonitorConfig(timeout_minutes=1)

@pytest.fixture
def mock_message_response() -> Tuple[str, str]:
    return ("Message sent successfully", "123456789")

@pytest.fixture
def mock_sender() -> Agent:
    mock = MagicMock(spec=Agent)
    mock.name = "mock_sender"
    return mock

@pytest.fixture
def mock_slack_handler(mocker):
    mock = mocker.patch("autogen.agentchat.contrib.comms.slack_agent.SlackHandler")
    handler = mock.return_value
    handler.start = AsyncMock(return_value=True)
    handler.initialize = AsyncMock()
    handler.send_message = AsyncMock(return_value=("Message sent successfully", "123456789"))
    handler.wait_for_replies = AsyncMock(return_value=[])
    handler.cleanup = AsyncMock()
    handler._message_replies = {}
    return mock

@pytest.fixture
def mock_discord_handler(mocker):
    mock = mocker.patch("autogen.agentchat.contrib.comms.discord_agent.DiscordHandler")
    handler = mock.return_value
    handler.start = AsyncMock(return_value=True)
    handler.initialize = AsyncMock()
    handler.send_message = AsyncMock(return_value=("Message sent successfully", "123456789"))
    handler.wait_for_replies = AsyncMock(return_value=[])
    handler.cleanup = AsyncMock()
    handler._message_replies = {}
    return mock

@pytest.fixture
def mock_telegram_wait_handler(mocker):
    mock = mocker.patch("autogen.agentchat.contrib.comms.telegram_agent.TelegramHandler.wait_for_replies")
    mock.return_value = [
        {"content": "Hi", "author": "TestUser", "timestamp": "2023-01-01T12:00:00", "id": "123.456"}
    ]
    return mock

@pytest.fixture
def mock_slack_wait_handler(mocker):
    mock = mocker.patch("autogen.agentchat.contrib.comms.slack_agent.SlackHandler.wait_for_replies")
    mock.return_value = [
        {"content": "Hi", "author": "TestUser", "timestamp": "2023-01-01T12:00:00", "id": "123.456"}
    ]
    return mock

@pytest.fixture
def mock_discord_wait_handler(mocker):
    mock = mocker.patch("autogen.agentchat.contrib.comms.discord_agent.DiscordHandler.wait_for_replies")
    mock.return_value = [
        {"content": "Hi", "author": "TestUser", "timestamp": "2023-01-01T12:00:00", "id": "123.456"}
    ]
    return mock

@pytest.fixture
def mock_platform_agent(mocker, mock_message_response):
    mock = mocker.patch("autogen.agentchat.contrib.comms.comms_platform_agent.CommsPlatformAgent.send_message")
    mock.return_value = mock_message_response
    return mock

@pytest.fixture
def chat_messages() -> List[Dict[str, str]]:
    return [{"role": "user", "content": "Send this message to the platform"}]
