# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
from typing import Callable
from unittest.mock import AsyncMock, Mock

import pytest

from autogen.import_utils import run_for_optional_imports
from autogen.tools.experimental.messageplatform import DiscordRetrieveTool, DiscordSendTool


@run_for_optional_imports("discord", "commsagent-discord")
class TestDiscordSendTool:
    @pytest.fixture(autouse=True)
    def mock_discord_client(self, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
        """Create a mock for the Discord Client."""
        # Create mock instance with required attributes
        mock_instance = AsyncMock()
        mock_instance.close = AsyncMock()

        # Mock start to trigger on_ready event immediately
        async def mock_start(token: str) -> None:
            # Get the on_ready handler that was registered and call it
            if hasattr(mock_instance, "_event_handlers"):
                on_ready = mock_instance._event_handlers.get("on_ready")
                if on_ready:
                    await on_ready()
            return None

        mock_instance.start = AsyncMock(side_effect=mock_start)
        mock_instance._event_handlers = {}

        # Mock the event registration
        def mock_event(func: Callable) -> Callable:  # type: ignore[type-arg]
            mock_instance._event_handlers["on_ready"] = func
            return func

        mock_instance.event = mock_event

        # Create the mock class that returns our instance
        mock_client_cls = Mock(return_value=mock_instance)

        # Patch both possible import paths
        monkeypatch.setattr("discord.Client", mock_client_cls)
        monkeypatch.setattr("autogen.tools.experimental.messageplatform.discord.discord.Client", mock_client_cls)

        return mock_instance

    @pytest.fixture
    def tool(self) -> DiscordSendTool:
        """Create a DiscordSendTool instance for testing."""
        return DiscordSendTool(bot_token="test-token", guild_name="test-guild", channel_name="test-channel")

    def test_discord_send_tool_init(self) -> None:
        """Test initialization of DiscordSendTool."""
        discord_send_tool = DiscordSendTool(
            bot_token="test-token", guild_name="test-guild", channel_name="test-channel"
        )

        assert discord_send_tool.name == "discord_send"
        assert discord_send_tool.description == "Sends a message to a Discord channel."
        assert isinstance(discord_send_tool.func, Callable)  # type: ignore[arg-type]

        expected_schema = {
            "description": "Sends a message to a Discord channel.",
            "name": "discord_send",
            "parameters": {
                "properties": {"message": {"description": "Message to send to the channel.", "type": "string"}},
                "required": ["message"],
                "type": "object",
            },
        }
        assert discord_send_tool.function_schema == expected_schema

    @pytest.mark.asyncio
    async def test_successful_message_send(self, tool: DiscordSendTool, mock_discord_client: AsyncMock) -> None:
        """Test successful message sending."""
        # Setup mock channel and guild
        mock_message = AsyncMock()
        mock_message.id = "123456789"

        mock_channel = AsyncMock()
        mock_channel.send = AsyncMock(return_value=mock_message)
        mock_channel.name = "test-channel"

        mock_guild = AsyncMock()
        mock_guild.name = "test-guild"
        mock_guild.text_channels = [mock_channel]

        # Attach guild to client
        mock_discord_client.guilds = [mock_guild]

        # Call the function with our test message
        message = "Test message"

        # Run the async function with a timeout to prevent infinite loops
        result = await asyncio.wait_for(
            tool.func(message=message, bot_token="test-token", guild_name="test-guild", channel_name="test-channel"),
            timeout=5,  # 5 second timeout
        )

        # Verify the discord client was started
        mock_discord_client.start.assert_called_once_with("test-token")

        # Verify the message was sent
        mock_channel.send.assert_called_once_with(message)

        # Verify the client was closed
        mock_discord_client.close.assert_called_once()

        # Check the result contains success message and message ID
        assert "Message sent successfully" in result
        assert "123456789" in result
        assert message in result

    @pytest.mark.asyncio
    async def test_long_message_chunking(self, tool: DiscordSendTool, mock_discord_client: AsyncMock) -> None:
        """Test that long messages are correctly chunked."""
        # Setup mock channel and guild with multiple message returns
        mock_messages = [
            AsyncMock(id="123456789"),  # First chunk
            AsyncMock(id="123456790"),  # Second chunk
            AsyncMock(id="123456791"),  # Third chunk
        ]

        mock_channel = AsyncMock()
        mock_channel.name = "test-channel"
        mock_channel.send = AsyncMock(side_effect=mock_messages)

        mock_guild = AsyncMock()
        mock_guild.name = "test-guild"
        mock_guild.text_channels = [mock_channel]

        # Attach guild to client
        mock_discord_client.guilds = [mock_guild]

        # Create a message longer than MAX_MESSAGE_LENGTH (2000)
        long_message = "x" * 4500  # Will need to be split into 3 chunks

        # Run the async function with a timeout
        result = await asyncio.wait_for(
            tool.func(
                message=long_message, bot_token="test-token", guild_name="test-guild", channel_name="test-channel"
            ),
            timeout=5,
        )

        # Verify the message was split and sent in chunks
        assert mock_channel.send.call_count == 3

        # Verify each chunk is within Discord's limit
        for call in mock_channel.send.call_args_list:
            assert len(call.args[0]) <= 1999  # MAX_MESSAGE_LENGTH - 1

        # Verify all chunks together make up the original message
        sent_message = ""
        for call in mock_channel.send.call_args_list:
            sent_message += call.args[0]
        assert sent_message == long_message

        # Check the result contains success message and first chunk ID
        assert "Message sent successfully" in result
        assert "chunks" in result
        assert "123456789" in result  # First chunk ID
        assert long_message in result

    @pytest.mark.asyncio
    async def test_guild_not_found(self, tool: DiscordSendTool, mock_discord_client: AsyncMock) -> None:
        """Test handling when guild is not found."""
        # Setup client with empty guilds list
        mock_discord_client.guilds = []

        test_guild_name = "test-guild"

        # Call the function with our test message
        message = "Test message"

        # Run the async function with a timeout
        result = await asyncio.wait_for(
            tool.func(message=message, bot_token="test-token", guild_name=test_guild_name, channel_name="test-channel"),
            timeout=5,
        )

        # Verify the client was started and closed
        mock_discord_client.start.assert_called_once_with("test-token")
        mock_discord_client.close.assert_called_once()

        # Check error message
        assert "Message send failed" in result
        expected_error = f"could not find guild: {test_guild_name}"
        assert expected_error in result.lower()

        # Verify no messages were attempted to be sent (guilds list is empty)
        assert len(mock_discord_client.guilds) == 0

    @pytest.mark.asyncio
    async def test_channel_not_found(self, tool: DiscordSendTool, mock_discord_client: AsyncMock) -> None:
        """Test handling when channel is not found in the guild."""
        # Setup mock guild with no matching channel
        mock_wrong_channel = AsyncMock()
        mock_wrong_channel.name = "wrong-channel"

        mock_guild = AsyncMock()
        mock_guild.name = "test-guild"
        mock_guild.text_channels = [mock_wrong_channel]  # Only contains wrong channel

        # Attach guild to client
        mock_discord_client.guilds = [mock_guild]

        test_channel_name = "test-channel"

        # Call the function with our test message
        message = "Test message"

        # Run the async function with a timeout
        result = await asyncio.wait_for(
            tool.func(message=message, bot_token="test-token", guild_name="test-guild", channel_name=test_channel_name),
            timeout=5,
        )

        # Verify the client was started and closed
        mock_discord_client.start.assert_called_once_with("test-token")
        mock_discord_client.close.assert_called_once()

        # Check error message
        assert "Message send failed" in result
        expected_error = f"could not find channel: {test_channel_name}"
        assert expected_error in result.lower()

        # Verify no message was attempted to be sent
        mock_wrong_channel.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_client_start_failure(self, tool: DiscordSendTool, mock_discord_client: AsyncMock) -> None:
        """Test handling of Discord client start failure."""
        # Setup client to raise an exception on start
        mock_discord_client.start.side_effect = Exception("Failed to connect")

        # Explicitly set empty guilds list
        mock_discord_client.guilds = []

        # Call the function with our test message
        message = "Test message"

        # Run the async function with a timeout and expect an exception
        with pytest.raises(Exception) as exc_info:
            await asyncio.wait_for(
                tool.func(
                    message=message, bot_token="test-token", guild_name="test-guild", channel_name="test-channel"
                ),
                timeout=5,
            )

        # Verify the exception message format
        assert str(exc_info.value) == "Failed to start Discord client: Failed to connect"

        # Verify the client was started but not closed (due to start failure)
        mock_discord_client.start.assert_called_once_with("test-token")
        mock_discord_client.close.assert_not_called()

        # Verify no guild interaction occurred
        assert len(mock_discord_client.guilds) == 0  # More explicit check


@run_for_optional_imports("discord", "commsagent-discord")
class TestDiscordRetrieveTool:
    @pytest.fixture(autouse=True)
    def mock_discord_client(self, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
        """Create a mock for the Discord Client."""
        # Create mock instance with required attributes
        mock_instance = AsyncMock()
        mock_instance.close = AsyncMock()

        # Mock start to trigger on_ready event immediately
        async def mock_start(token: str) -> None:
            # Get the on_ready handler that was registered and call it
            if hasattr(mock_instance, "_event_handlers"):
                on_ready = mock_instance._event_handlers.get("on_ready")
                if on_ready:
                    await on_ready()
            return None

        mock_instance.start = AsyncMock(side_effect=mock_start)
        mock_instance._event_handlers = {}

        # Mock the event registration
        def mock_event(func: Callable) -> Callable:  # type: ignore[type-arg]
            mock_instance._event_handlers["on_ready"] = func
            return func

        mock_instance.event = mock_event

        # Create the mock class that returns our instance
        mock_client_cls = Mock(return_value=mock_instance)

        # Patch both possible import paths
        monkeypatch.setattr("discord.Client", mock_client_cls)
        monkeypatch.setattr("autogen.tools.experimental.messageplatform.discord.discord.Client", mock_client_cls)

        return mock_instance

    @pytest.fixture
    def tool(self) -> DiscordRetrieveTool:
        """Create a DiscordRetrieveTool instance for testing."""
        return DiscordRetrieveTool(bot_token="test-token", guild_name="test-guild", channel_name="test-channel")

    def test_discord_retrieve_tool_init(self) -> None:
        discord_retrieve_tool = DiscordRetrieveTool(
            bot_token="my_bot_token", guild_name="my_guild", channel_name="my_channel"
        )
        assert discord_retrieve_tool.name == "discord_retrieve"
        assert (
            discord_retrieve_tool.description
            == "Retrieves messages from a Discord channel based datetime/message ID and/or number of latest messages."
        )
        assert isinstance(discord_retrieve_tool.func, Callable)  # type: ignore[arg-type]

        expected_schema = {
            "description": "Retrieves messages from a Discord channel based datetime/message ID and/or number of latest messages.",
            "name": "discord_retrieve",
            "parameters": {
                "type": "object",
                "properties": {
                    "messages_since": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": "Date to retrieve messages from (ISO format) OR Discord snowflake ID. If None, retrieves latest messages.",
                    },
                    "maximum_messages": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                        "description": "Maximum number of messages to retrieve. If None, retrieves all messages since date.",
                    },
                },
                "required": [],
            },
        }

        assert discord_retrieve_tool.function_schema == expected_schema

    @pytest.mark.asyncio
    async def test_successful_message_retrieval(
        self, tool: DiscordRetrieveTool, mock_discord_client: AsyncMock
    ) -> None:
        """Test successful message retrieval."""
        # Setup mock messages
        mock_messages = [
            AsyncMock(
                id="123456789",
                content="Test message 1",
                author=AsyncMock(spec_set=["__str__"]),
                created_at=AsyncMock(isoformat=lambda: "2024-01-01T10:00:00+00:00"),
            ),
            AsyncMock(
                id="123456790",
                content="Test message 2",
                author=AsyncMock(spec_set=["__str__"]),
                created_at=AsyncMock(isoformat=lambda: "2024-01-01T10:01:00+00:00"),
            ),
        ]

        # Set author string representations
        mock_messages[0].author.__str__.return_value = "TestUser1"
        mock_messages[1].author.__str__.return_value = "TestUser2"

        # Setup mock channel with history
        mock_channel = AsyncMock()
        mock_channel.name = "test-channel"
        mock_channel.history.return_value = mock_messages

        # Setup mock guild
        mock_guild = AsyncMock()
        mock_guild.name = "test-guild"
        mock_guild.text_channels = [mock_channel]

        # Attach guild to client
        mock_discord_client.guilds = [mock_guild]

        # Create a future to store the result
        result_future: asyncio.Future[list[dict[str, str]]] = asyncio.Future()

        # Override the on_ready event to set the result
        @mock_discord_client.event  # type: ignore[misc]
        async def on_ready() -> None:
            try:
                messages = []
                for msg in mock_messages:
                    messages.append({
                        "id": str(msg.id),
                        "content": msg.content,
                        "author": str(msg.author),
                        "timestamp": msg.created_at.isoformat(),
                    })
                result_future.set_result(messages)
            except Exception as e:
                result_future.set_exception(e)
            finally:
                await mock_discord_client.close()

        # Start the client and get the result
        await mock_discord_client.start("test-token")
        result = await result_future

        # Verify the discord client was started and closed
        mock_discord_client.start.assert_called_once_with("test-token")
        mock_discord_client.close.assert_called_once()

        # Check the result format and content
        assert isinstance(result, list)
        assert len(result) == 2

        # Verify first message
        assert result[0]["id"] == "123456789"
        assert result[0]["content"] == "Test message 1"
        assert result[0]["author"] == "TestUser1"
        assert result[0]["timestamp"] == "2024-01-01T10:00:00+00:00"

        # Verify second message
        assert result[1]["id"] == "123456790"
        assert result[1]["content"] == "Test message 2"
        assert result[1]["author"] == "TestUser2"
        assert result[1]["timestamp"] == "2024-01-01T10:01:00+00:00"

    @pytest.mark.asyncio
    async def test_guild_not_found(self, tool: DiscordRetrieveTool, mock_discord_client: AsyncMock) -> None:
        """Test handling when guild is not found."""
        # Setup client with empty guilds list
        mock_discord_client.guilds = []

        test_guild_name = "test-guild"

        # Call the function and get results
        result = await tool.func(bot_token="test-token", guild_name=test_guild_name, channel_name="test-channel")

        # Verify the discord client was started and closed
        mock_discord_client.start.assert_called_once_with("test-token")
        mock_discord_client.close.assert_called_once()

        # Check error message
        assert isinstance(result, list)
        assert len(result) == 1
        assert "error" in result[0]
        expected_error = f"Could not find guild: {test_guild_name}"
        assert result[0]["error"] == expected_error

    @pytest.mark.asyncio
    async def test_channel_not_found(self, tool: DiscordRetrieveTool, mock_discord_client: AsyncMock) -> None:
        """Test handling when channel is not found in the guild."""
        # Setup mock guild with no matching channel
        mock_wrong_channel = AsyncMock()
        mock_wrong_channel.name = "wrong-channel"

        mock_guild = AsyncMock()
        mock_guild.name = "test-guild"
        mock_guild.text_channels = [mock_wrong_channel]  # Only contains wrong channel

        # Attach guild to client
        mock_discord_client.guilds = [mock_guild]

        test_channel_name = "test-channel"

        # Call the function and get results
        result = await tool.func(bot_token="test-token", guild_name="test-guild", channel_name=test_channel_name)

        # Verify the discord client was started and closed
        mock_discord_client.start.assert_called_once_with("test-token")
        mock_discord_client.close.assert_called_once()

        # Check error message
        assert isinstance(result, list)
        assert len(result) == 1
        assert "error" in result[0]
        expected_error = f"Could not find channel: {test_channel_name}"
        assert result[0]["error"] == expected_error
