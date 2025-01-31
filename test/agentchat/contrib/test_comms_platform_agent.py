import pytest
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, List, Optional, Union, Tuple
import asyncio

from autogen import Agent
from pydantic import BaseModel, Field
from autogen.agentchat.contrib.comms.comms_platform_agent import CommsPlatformAgent
from autogen.agentchat.contrib.comms.platform_configs import ReplyMonitorConfig, BaseCommsPlatformConfig
from autogen.agentchat.contrib.comms.platform_errors import (
    PlatformError,
    PlatformAuthenticationError,
    PlatformConnectionError,
    PlatformRateLimitError,
    PlatformTimeoutError,
)

class MockPlatformAgent(CommsPlatformAgent):
    def __init__(self, name: str, **kwargs):
        llm_config = {
            "config_list": [{"model": "gpt-4", "api_key": "fake-key"}]
        }
        class MockConfig(BaseCommsPlatformConfig):
            timeout_minutes: int = 1
            max_reply_messages: int = 1

            def validate_config(self) -> bool:
                return True

            def __str__(self) -> str:
                return "MockConfig"

            class Config:
                extra = "allow"

        platform_config = MockConfig()
        self.mock_send = MagicMock(return_value=("Message sent", "123"))
        self.mock_wait = MagicMock(return_value=["Reply 1"])
        self._executor_agent = MagicMock()
        super().__init__(
            name=name,
            llm_config=llm_config,
            platform_config=platform_config,
            executor_agent=self._executor_agent,
            **kwargs
        )

    async def a_receive(self, message: dict, sender: Agent) -> tuple:
        await super().a_receive(message, sender)
        result = self.send_to_platform(message["content"])
        return result

    def send_to_platform(self, message: str) -> Tuple[str, Optional[str]]:
        return self.mock_send(message)

    def wait_for_reply(self, message_id: str) -> List[str]:
        return self.mock_wait(message_id)

    def send_message(self, message: str) -> Tuple[str, Optional[str]]:
        return self.send_to_platform(message)

    async def a_send_message(self, message: str) -> Tuple[str, Optional[str]]:
        return self.send_to_platform(message)

    def cleanup_monitoring(self, message_id: str) -> None:
        pass

@pytest.mark.asyncio
async def test_comms_platform_agent_init():
    agent = MockPlatformAgent(name="test_agent")
    assert agent.name == "test_agent"
    assert isinstance(agent, CommsPlatformAgent)

@pytest.mark.asyncio
async def test_comms_platform_agent_send_message():
    agent = MockPlatformAgent(name="test_agent")
    response = agent.send_message("Test message")
    assert response == ("Message sent", "123")
    agent.mock_send.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_comms_platform_agent_async_send():
    agent = MockPlatformAgent(name="test_agent")
    response = await agent.a_send_message("Test message")
    assert response == ("Message sent", "123")
    agent.mock_send.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_comms_platform_agent_async_receive():
    agent = MockPlatformAgent(name="test_agent")
    mock_sender = MagicMock(spec=Agent)
    mock_sender.name = "mock_sender"
    mock_sender.silent = None

    test_message = {"role": "user", "content": "Test message"}
    response = await agent.a_receive(test_message, mock_sender)
    
    assert response == ("Message sent", "123")
    agent.mock_send.assert_called_once_with("Test message")

@pytest.mark.asyncio
async def test_comms_platform_agent_error_handling():
    agent = MockPlatformAgent(name="test_agent")
    agent.mock_send.side_effect = PlatformError(
        message="Test error",
        platform_name="Mock"
    )

    with pytest.raises(PlatformError) as exc_info:
        agent.send_message("Test message")
    assert "Test error" in str(exc_info.value)
    assert exc_info.value.platform_name == "Mock"

@pytest.mark.asyncio
async def test_comms_platform_agent_auth_error():
    agent = MockPlatformAgent(name="test_agent")
    agent.mock_send.side_effect = PlatformAuthenticationError(
        message="Auth failed",
        platform_name="Mock"
    )

    with pytest.raises(PlatformAuthenticationError) as exc_info:
        agent.send_message("Test message")
    assert "Auth failed" in str(exc_info.value)

@pytest.mark.asyncio
async def test_comms_platform_agent_connection_error():
    agent = MockPlatformAgent(name="test_agent")
    agent.mock_send.side_effect = PlatformConnectionError(
        message="Connection failed",
        platform_name="Mock"
    )

    with pytest.raises(PlatformConnectionError) as exc_info:
        agent.send_message("Test message")
    assert "Connection failed" in str(exc_info.value)

@pytest.mark.asyncio
async def test_comms_platform_agent_rate_limit():
    agent = MockPlatformAgent(name="test_agent")
    agent.mock_send.side_effect = PlatformRateLimitError(
        message="Rate limit exceeded",
        platform_name="Mock",
        retry_after=5
    )

    with pytest.raises(PlatformRateLimitError) as exc_info:
        agent.send_message("Test message")
    assert "Rate limit exceeded" in str(exc_info.value)
    assert exc_info.value.retry_after == 5

@pytest.mark.asyncio
async def test_comms_platform_agent_timeout():
    agent = MockPlatformAgent(name="test_agent")
    agent.mock_send.side_effect = PlatformTimeoutError(
        message="Operation timed out",
        platform_name="Mock"
    )

    with pytest.raises(PlatformTimeoutError) as exc_info:
        agent.send_message("Test message")
    assert "Operation timed out" in str(exc_info.value)

@pytest.mark.asyncio
async def test_comms_platform_agent_cleanup():
    agent = MockPlatformAgent(name="test_agent")
    agent.cleanup_monitoring("123")
    assert True  # Base class cleanup is a no-op
