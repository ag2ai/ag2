import pytest
import asyncio
from unittest.mock import MagicMock, patch
from typing import Dict, List, Optional, Union

import discord
from autogen import Agent
from autogen.agentchat.contrib.comms.discord_agent import DiscordAgent, DiscordConfig, DiscordHandler
from autogen.agentchat.contrib.comms.platform_configs import ReplyMonitorConfig
from autogen.agentchat.contrib.comms.platform_errors import (
    PlatformAuthenticationError,
    PlatformConnectionError,
    PlatformError,
    PlatformRateLimitError,
    PlatformTimeoutError,
)

@pytest.mark.asyncio
async def test_discord_config_validation():
    # Test valid configuration
    config = DiscordConfig(bot_token="fake_token", guild_name="TestGuild", channel_name="TestChannel")
    assert config.validate_config() is True

    # Test missing required fields
    with pytest.raises(ValueError, match="bot_token, guild_name, and channel_name are required"):
        DiscordConfig(bot_token="", guild_name="", channel_name="").validate_config()

@pytest.mark.asyncio
async def test_discord_handler_authentication(mocker):
    # Mock Discord client
    mock_client = MagicMock()
    mock_client.login.side_effect = discord.LoginFailure("Invalid token")
    
    with patch("discord.Client", return_value=mock_client):
        handler = DiscordHandler(bot_token="invalid_token", guild_name="TestGuild", channel_name="TestChannel")
        with pytest.raises(PlatformAuthenticationError, match="Failed to login with provided token"):
            handler.start()

@pytest.mark.asyncio
async def test_discord_handler_connection(mocker):
    # Mock Discord client
    mock_client = MagicMock()
    mock_client.guilds = []
    
    with patch("discord.Client", return_value=mock_client):
        handler = DiscordHandler(bot_token="valid_token", guild_name="NonexistentGuild", channel_name="TestChannel")
        with pytest.raises(PlatformConnectionError, match="Could not find guild: NonexistentGuild"):
            handler.start()

@pytest.mark.asyncio
async def test_discord_handler_rate_limit(mocker):
    mock_client = MagicMock()
    mock_channel = MagicMock()
    mock_channel.send.side_effect = discord.HTTPException(response=MagicMock(status=429), message="Rate limit exceeded")
    
    with patch("discord.Client", return_value=mock_client):
        handler = DiscordHandler(bot_token="valid_token", guild_name="TestGuild", channel_name="TestChannel")
        handler._channel = mock_channel
        handler._ready.set()
        
        with pytest.raises(PlatformRateLimitError, match="Rate limit exceeded"):
            await handler.send_message("Test message")

@pytest.mark.asyncio
async def test_discord_handler_timeout(mocker):
    mock_client = MagicMock()
    
    with patch("discord.Client", return_value=mock_client):
        handler = DiscordHandler(bot_token="valid_token", guild_name="TestGuild", channel_name="TestChannel")
        
        with pytest.raises(PlatformTimeoutError, match="Timeout waiting for Discord client to be ready"):
            await handler.send_message("Test message")

@pytest.mark.asyncio
async def test_discord_agent_initialization():
    config = DiscordConfig(bot_token="fake_token", guild_name="TestGuild", channel_name="TestChannel")
    agent = DiscordAgent(name="test_discord_agent", platform_config=config)
    assert agent.name == "test_discord_agent"
    assert isinstance(agent.platform_config, DiscordConfig)
    assert agent.executor_agent is not None

@pytest.mark.asyncio
async def test_discord_handler_message_chunking(mocker):
    mock_client = MagicMock()
    mock_channel = MagicMock()
    mock_channel.send.side_effect = [
        MagicMock(id="123456789"),  # First chunk
        MagicMock(id="987654321"),  # Second chunk
    ]
    
    with patch("discord.Client", return_value=mock_client):
        handler = DiscordHandler(bot_token="valid_token", guild_name="TestGuild", channel_name="TestChannel")
        handler._channel = mock_channel
        handler._ready.set()
        
        long_message = "x" * 2500  # Message longer than Discord's 2000 char limit
        status, msg_id = await handler.send_message(long_message)
        
        assert status == "Message sent (split into chunks)"
        assert msg_id == "123456789"  # Should return first chunk's ID
        assert mock_channel.send.call_count == 2

@pytest.mark.asyncio
async def test_discord_handler_forbidden_error(mocker):
    mock_client = MagicMock()
    mock_channel = MagicMock()
    mock_channel.send.side_effect = discord.Forbidden(response=MagicMock(), message="Missing permissions")
    
    with patch("discord.Client", return_value=mock_client):
        handler = DiscordHandler(bot_token="valid_token", guild_name="TestGuild", channel_name="TestChannel")
        handler._channel = mock_channel
        handler._ready.set()
        
        with pytest.raises(PlatformAuthenticationError, match="Bot lacks permission to send messages"):
            await handler.send_message("Test message")

@pytest.mark.asyncio
async def test_discord_agent_send_message_sync_and_async(mocker, discord_config, mock_message_response):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.discord_agent.DiscordHandler.send_message")
    mock_handler.return_value = mock_message_response

    agent = DiscordAgent(name="test_discord_agent", platform_config=discord_config)

    # Test synchronous operation through executor agent
    response = agent.executor_agent.send_to_platform("Hello Discord!")
    assert response[0] == mock_message_response[0]
    assert response[1] == mock_message_response[1]

    # Test asynchronous operation through nested chat sequence
    mock_sender = MagicMock(spec=Agent)
    mock_sender.name = "mock_sender"
    test_message: Dict[str, str] = {"role": "user", "content": "Send this message to Discord"}
    async_response = await agent.a_receive(test_message, mock_sender)
    assert isinstance(async_response, str)
    assert "Message sent successfully" in async_response

@pytest.mark.asyncio
async def test_discord_agent_send_message_with_attachments(mocker):
    mock_message = MagicMock()
    mock_message.attachments = [
        MagicMock(filename="test.txt", url="http://test.com/test.txt"),
        MagicMock(filename="image.png", url="http://test.com/image.png")
    ]
    mock_message.embeds = [MagicMock(to_dict=lambda: {"title": "Test Embed"})]
    
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.discord_agent.DiscordHandler.wait_for_replies")
    mock_handler.return_value = [{
        "content": "Test message",
        "author": "TestUser",
        "timestamp": "2023-01-01T12:00:00",
        "id": "123456789",
        "attachments": [
            {"filename": "test.txt", "url": "http://test.com/test.txt"},
            {"filename": "image.png", "url": "http://test.com/image.png"}
        ],
        "embeds": [{"title": "Test Embed"}]
    }]

    config = DiscordConfig(bot_token="fake_token", guild_name="TestGuild", channel_name="TestChannel")
    agent = DiscordAgent(
        name="test_discord_agent",
        platform_config=config,
        reply_monitor_config=ReplyMonitorConfig(timeout_minutes=1)
    )
    
    replies = agent.executor_agent.wait_for_reply("msg_id")
    assert len(replies) == 1
    assert "Test message" in replies[0]
    assert "(Attachment: test.txt, URL: http://test.com/test.txt)" in replies[0]
    assert "(Attachment: image.png, URL: http://test.com/image.png)" in replies[0]

@pytest.mark.asyncio
async def test_discord_agent_send_message_with_markdown(mocker):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.discord_agent.DiscordHandler.send_message")
    mock_handler.return_value = ("Message sent successfully", "123456789")

    config = DiscordConfig(bot_token="fake_token", guild_name="TestGuild", channel_name="TestChannel")
    agent = DiscordAgent(name="test_discord_agent", platform_config=config)

    markdown_message = "**Bold** *Italic* ```Code Block```"
    response = agent.executor_agent.send_to_platform(markdown_message)
    assert response[0] == "Message sent successfully"
    assert response[1] == "123456789"
    mock_handler.assert_called_once_with(markdown_message)

@pytest.mark.asyncio
async def test_discord_agent_wait_for_replies_with_timeout(mocker, discord_config, reply_monitor_config):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.discord_agent.DiscordHandler.wait_for_replies")
    mock_handler.side_effect = PlatformTimeoutError(
        message="Timeout waiting for replies after 1 minutes",
        platform_name="Discord"
    )

    agent = DiscordAgent(
        name="test_discord_agent",
        platform_config=discord_config,
        reply_monitor_config=reply_monitor_config
    )
    replies = agent.executor_agent.wait_for_reply("msg_id")
    assert len(replies) == 0

@pytest.mark.asyncio
async def test_discord_agent_wait_for_replies_max_replies(mocker):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.discord_agent.DiscordHandler.wait_for_replies")
    mock_handler.return_value = [
        {"content": f"Reply {i}", "author": "TestUser", "timestamp": "2023-01-01T12:00:00", "id": str(i)}
        for i in range(5)
    ]

    config = DiscordConfig(bot_token="fake_token", guild_name="TestGuild", channel_name="TestChannel")
    agent = DiscordAgent(
        name="test_discord_agent",
        platform_config=config,
        reply_monitor_config=ReplyMonitorConfig(timeout_minutes=1, max_reply_messages=3)
    )
    replies = agent.executor_agent.wait_for_reply("msg_id")
    assert len(replies) == 5  # Should get all replies even if max_replies is hit
    assert all(f"Reply {i}" in reply for i, reply in enumerate(replies))

@pytest.mark.asyncio
async def test_discord_agent_wait_for_replies_filter_author(mocker):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.discord_agent.DiscordHandler.wait_for_replies")
    mock_handler.return_value = [
        {"content": "Hi from bot", "author": "BotUser", "timestamp": "2023-01-01T12:00:00", "id": "1"},
        {"content": "Hi from human", "author": "HumanUser", "timestamp": "2023-01-01T12:00:01", "id": "2"},
        {"content": "Another bot msg", "author": "BotUser", "timestamp": "2023-01-01T12:00:02", "id": "3"}
    ]

    config = DiscordConfig(bot_token="fake_token", guild_name="TestGuild", channel_name="TestChannel")
    agent = DiscordAgent(
        name="test_discord_agent",
        platform_config=config,
        reply_monitor_config=ReplyMonitorConfig(timeout_minutes=1, allowed_authors=["HumanUser"])
    )
    
    replies = agent.executor_agent.wait_for_reply("msg_id")
    assert len(replies) == 1
    assert "Hi from human" in replies[0]
    assert "Hi from bot" not in replies[0]
    assert "Another bot msg" not in replies[0]

@pytest.mark.asyncio
async def test_discord_agent_cleanup_monitoring(mocker):
    mock_cleanup = mocker.patch("autogen.agentchat.contrib.comms.discord_agent.DiscordHandler.cleanup_reply_monitoring")
    
    config = DiscordConfig(bot_token="fake_token", guild_name="TestGuild", channel_name="TestChannel")
    agent = DiscordAgent(
        name="test_discord_agent",
        platform_config=config,
        reply_monitor_config=ReplyMonitorConfig(timeout_minutes=1)
    )
    
    agent.executor_agent.cleanup_monitoring("msg_id")
    mock_cleanup.assert_called_once_with("msg_id")

@pytest.mark.asyncio
async def test_discord_agent_send_message_rate_limit(mocker):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.discord_agent.DiscordHandler.send_message")
    mock_handler.side_effect = PlatformRateLimitError(
        message="Rate limit exceeded",
        platform_name="Discord",
        retry_after=5
    )

    config = DiscordConfig(bot_token="fake_token", guild_name="TestGuild", channel_name="TestChannel")
    agent = DiscordAgent(name="test_discord_agent", platform_config=config)

    with pytest.raises(PlatformRateLimitError) as exc_info:
        agent.executor_agent.send_to_platform("Test message")
    assert "Rate limit exceeded" in str(exc_info.value)
    assert exc_info.value.retry_after == 5

@pytest.mark.asyncio
async def test_discord_agent_invalid_token(mocker):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.discord_agent.DiscordHandler.start")
    mock_handler.side_effect = PlatformAuthenticationError(
        message="Invalid token provided",
        platform_name="Discord"
    )

    config = DiscordConfig(bot_token="invalid_token", guild_name="TestGuild", channel_name="TestChannel")
    agent = DiscordAgent(name="test_discord_agent", platform_config=config)

    with pytest.raises(PlatformAuthenticationError) as exc_info:
        await agent.executor_agent.initialize()
    assert "Invalid token provided" in str(exc_info.value)

@pytest.mark.asyncio
async def test_discord_agent_channel_not_found(mocker):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.discord_agent.DiscordHandler.start")
    mock_handler.side_effect = PlatformConnectionError(
        message="Channel 'nonexistent' not found in guild 'TestGuild'",
        platform_name="Discord"
    )

    config = DiscordConfig(bot_token="fake_token", guild_name="TestGuild", channel_name="nonexistent")
    agent = DiscordAgent(name="test_discord_agent", platform_config=config)

    with pytest.raises(PlatformConnectionError) as exc_info:
        await agent.executor_agent.initialize()
    assert "Channel 'nonexistent' not found" in str(exc_info.value)

@pytest.mark.asyncio
async def test_discord_agent_network_error(mocker):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.discord_agent.DiscordHandler.send_message")
    mock_handler.side_effect = PlatformConnectionError(
        message="Network connection failed",
        platform_name="Discord"
    )

    config = DiscordConfig(bot_token="fake_token", guild_name="TestGuild", channel_name="TestChannel")
    agent = DiscordAgent(name="test_discord_agent", platform_config=config)

    with pytest.raises(PlatformConnectionError) as exc_info:
        agent.executor_agent.send_to_platform("Test message")
    assert "Network connection failed" in str(exc_info.value)

@pytest.mark.asyncio
async def test_discord_agent_wait_for_replies_no_monitor_config(mocker):
    mock_handler = mocker.patch("autogen.agentchat.contrib.comms.discord_agent.DiscordHandler.wait_for_replies")
    
    config = DiscordConfig(bot_token="fake_token", guild_name="TestGuild", channel_name="TestChannel")
    agent = DiscordAgent(
        name="test_discord_agent",
        platform_config=config,
        reply_monitor_config=None  # No monitoring config
    )
    
    replies = agent.executor_agent.wait_for_reply("msg_id")
    assert len(replies) == 0
    mock_handler.assert_not_called()
