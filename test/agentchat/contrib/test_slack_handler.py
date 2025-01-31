import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from slack_sdk.errors import SlackApiError
import asyncio
from typing import Dict, List, Optional, Union, Tuple

from autogen.agentchat.contrib.comms.slack_agent import SlackHandler, SlackConfig
from autogen.agentchat.contrib.comms.platform_errors import (
    PlatformAuthenticationError,
    PlatformConnectionError,
    PlatformRateLimitError,
    PlatformTimeoutError,
)

@pytest.fixture
def slack_handler():
    config = SlackConfig(bot_token="mock_test_token", channel_id="mock_channel", signing_secret="mock_secret")
    handler = SlackHandler(config=config)
    with patch.object(handler, '_slack_client', new=MagicMock()) as mock_client:
        mock_client.chat_postMessage = AsyncMock()
        with patch.object(handler, '_ready', new=asyncio.Event()) as ready_event:
            ready_event.set()
            return handler

@pytest.mark.asyncio
async def test_slack_handler_start(slack_handler):
    with patch.object(slack_handler, '_client') as mock_client:
        mock_client.auth_test = AsyncMock(return_value={"ok": True})
        assert await slack_handler.start() is True
        mock_client.auth_test.assert_called_once()

@pytest.mark.asyncio
async def test_slack_handler_start_auth_error(slack_handler):
    slack_handler._client.auth_test = AsyncMock(side_effect=SlackApiError("invalid_auth", {"error": "invalid_auth"}))
    with pytest.raises(PlatformAuthenticationError):
        await slack_handler.start()

@pytest.mark.asyncio
async def test_slack_handler_send_message(slack_handler):
    slack_handler._client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "123.456"})
    result = await slack_handler.send_message("Test message")
    assert result == ("Message sent successfully", "123.456")
    slack_handler._client.chat_postMessage.assert_called_once_with(channel="ABC123", text="Test message")

@pytest.mark.asyncio
async def test_slack_handler_send_long_message(slack_handler):
    long_message = "x" * 4001  # Slack's limit is 4000
    slack_handler._client.chat_postMessage = AsyncMock(side_effect=[{"ok": True, "ts": "123.456"}, {"ok": True, "ts": "123.457"}])
    result = await slack_handler.send_message(long_message)
    assert result[0] == "Message sent successfully"
    assert slack_handler._client.chat_postMessage.call_count == 2

@pytest.mark.asyncio
async def test_slack_handler_send_message_rate_limit(slack_handler):
    slack_handler._client.chat_postMessage = AsyncMock(side_effect=SlackApiError("ratelimited", {"error": "ratelimited", "retry_after": 5}))
    with pytest.raises(PlatformRateLimitError) as exc_info:
        await slack_handler.send_message("Test message")
    assert exc_info.value.retry_after == 5

@pytest.mark.asyncio
async def test_slack_handler_send_message_network_error(slack_handler):
    slack_handler._client.chat_postMessage = AsyncMock(side_effect=SlackApiError("connection_error", {"error": "connection_error"}))
    with pytest.raises(PlatformConnectionError):
        await slack_handler.send_message("Test message")

@pytest.mark.asyncio
async def test_slack_handler_wait_for_replies(slack_handler):
    slack_handler._message_replies["123.456"] = [{"content": "Reply 1", "author": "U123", "timestamp": "123.457"}, {"content": "Reply 2", "author": "U124", "timestamp": "123.458"}]
    replies = await slack_handler.wait_for_replies("123.456", timeout_minutes=1)
    assert len(replies) == 2
    assert replies[0]["content"] == "Reply 1"
    assert replies[1]["content"] == "Reply 2"

@pytest.mark.asyncio
async def test_slack_handler_wait_for_replies_timeout(slack_handler):
    with pytest.raises(PlatformTimeoutError):
        await slack_handler.wait_for_replies("123.456", timeout_minutes=0.001)

@pytest.mark.asyncio
async def test_slack_handler_cleanup_monitoring(slack_handler):
    slack_handler._message_replies["123.456"] = [{"content": "Reply"}]
    slack_handler.cleanup_reply_monitoring("123.456")
    assert "123.456" not in slack_handler._message_replies
