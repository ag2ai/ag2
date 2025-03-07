# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Callable
from unittest.mock import AsyncMock, Mock

import pytest

from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.tools.experimental.messageplatform import TelegramRetrieveTool, TelegramSendTool

with optional_import_block():
    from telethon import TelegramClient
    from telethon.tl.types import Message


@run_for_optional_imports(["telethon"], "commsagent-telegram")
class TestTelegramSendTool:
    @pytest.fixture(autouse=True)
    def mock_telegram_client(self, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
        """Create a mock for the TelegramClient."""
        # Create mock instance with required attributes
        mock_instance = AsyncMock(spec=TelegramClient)

        # Mock the context manager methods
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.__aexit__.return_value = None

        # Mock entity-related methods
        mock_entity = AsyncMock()
        mock_instance.get_entity = AsyncMock(return_value=mock_entity)
        mock_instance.iter_dialogs = AsyncMock(return_value=[])

        # Create mock message response
        mock_message = Mock(spec=Message)
        mock_message.id = 123456789
        mock_instance.send_message.return_value = mock_message

        # Create the mock class that returns our instance
        mock_client_cls = Mock(return_value=mock_instance)

        # Mock the TelegramClient in BaseTelegramTool
        monkeypatch.setattr(
            "autogen.tools.experimental.messageplatform.telegram.telegram.TelegramClient",
            mock_client_cls,
        )

        return mock_instance

    @pytest.fixture
    def tool(self) -> TelegramSendTool:
        """Create a TelegramSendTool instance for testing."""
        return TelegramSendTool(api_id="test-api-id", api_hash="test-api-hash", chat_id="-1001234567890")

    def test_telegram_send_tool_init(self) -> None:
        telegram_send_tool = TelegramSendTool(api_id="-100", api_hash="my_api_hash", chat_id="my_chat")
        assert telegram_send_tool.name == "telegram_send"
        assert (
            telegram_send_tool.description == "Sends a message to a personal channel, bot channel, group, or channel."
        )
        assert isinstance(telegram_send_tool.func, Callable)  # type: ignore[arg-type]

        expected_schema = {
            "description": "Sends a message to a personal channel, bot channel, group, or channel.",
            "name": "telegram_send",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string", "description": "Message to send to the chat."}},
                "required": ["message"],
            },
        }

        assert telegram_send_tool.function_schema == expected_schema

    @pytest.mark.asyncio
    async def test_successful_message_send(self, tool: TelegramSendTool, mock_telegram_client: AsyncMock) -> None:
        """Test successful message sending."""
        # Call the function with our test message
        message = "Test message"
        result = await tool.func(message=message, chat_id="-1001234567890")

        # Verify client was used correctly
        mock_telegram_client.__aenter__.assert_called_once()
        mock_telegram_client.__aexit__.assert_called_once()

        # Verify the entity initialization attempted to get entity
        assert mock_telegram_client.get_entity.called

        # Verify message was sent
        mock_telegram_client.send_message.assert_called_once()
        call_args = mock_telegram_client.send_message.call_args
        assert call_args.kwargs["message"] == message
        assert call_args.kwargs["parse_mode"] == "html"
        assert call_args.kwargs["entity"] == mock_telegram_client.get_entity.return_value

        # Check the result
        assert "Message sent successfully" in result
        assert "123456789" in result  # Message ID should be in result
        assert message in result

    @pytest.mark.asyncio
    async def test_long_message_chunking(self, tool: TelegramSendTool, mock_telegram_client: AsyncMock) -> None:
        """Test that long messages are correctly chunked."""
        # Setup mock messages for each chunk
        mock_messages = [
            Mock(spec=Message, id=123456789),  # First chunk
            Mock(spec=Message, id=123456790),  # Second chunk
            Mock(spec=Message, id=123456791),  # Third chunk
        ]

        # Configure send_message to return different messages for each call
        mock_telegram_client.send_message = AsyncMock(side_effect=mock_messages)

        # Create a message longer than MAX_MESSAGE_LENGTH (4096)
        long_message = "x" * 8500  # Will need to be split into 3 chunks

        # Call the function with our long message
        result = await tool.func(message=long_message, chat_id="-1001234567890")

        # Verify client was used correctly
        mock_telegram_client.__aenter__.assert_called_once()
        mock_telegram_client.__aexit__.assert_called_once()

        # Verify the entity initialization attempted to get entity
        assert mock_telegram_client.get_entity.called

        # Verify message was split and sent in chunks
        assert mock_telegram_client.send_message.call_count == 3

        # Verify each chunk is within Telegram's limit
        for call in mock_telegram_client.send_message.call_args_list:
            assert len(call.kwargs["message"]) <= 4095  # MAX_MESSAGE_LENGTH - 1

        # Verify all chunks together make up the original message
        sent_message = ""
        for call in mock_telegram_client.send_message.call_args_list:
            sent_message += call.kwargs["message"]
        assert sent_message == long_message

        # Verify chain replies (reply_to parameter)
        for i, call in enumerate(mock_telegram_client.send_message.call_args_list[1:], 1):
            assert call.kwargs["reply_to"] == mock_messages[0].id

        # Check the result
        assert "Message sent successfully" in result
        assert "chunks" in result
        assert "123456789" in result  # First chunk ID
        assert long_message in result

    @pytest.mark.asyncio
    async def test_entity_not_found(self, tool: TelegramSendTool, mock_telegram_client: AsyncMock) -> None:
        """Test handling when entity (channel/chat/user) is not found."""
        # Setup client to fail entity retrieval
        mock_telegram_client.get_entity.side_effect = ValueError("Failed to get the input entity")
        mock_telegram_client.iter_dialogs = AsyncMock(return_value=[])

        message = "Test message"
        result = await tool.func(message=message, chat_id="-1001234567890")

        # Verify client was used correctly
        mock_telegram_client.__aenter__.assert_called_once()
        mock_telegram_client.__aexit__.assert_called_once()

        # Verify entity initialization was attempted
        assert mock_telegram_client.get_entity.called
        assert mock_telegram_client.iter_dialogs.called

        # Verify no message was sent
        mock_telegram_client.send_message.assert_not_called()

        # Check error message in result
        assert "Message send failed" in result
        assert "Could not initialize entity" in result

    @pytest.mark.asyncio
    async def test_general_exception(self, tool: TelegramSendTool, mock_telegram_client: AsyncMock) -> None:
        """Test handling of general exceptions."""
        # Setup client to raise an unexpected exception
        mock_telegram_client.send_message.side_effect = Exception("Unexpected error")

        message = "Test message"
        result = await tool.func(message=message, chat_id="-1001234567890")

        # Verify client was used correctly
        mock_telegram_client.__aenter__.assert_called_once()
        mock_telegram_client.__aexit__.assert_called_once()

        # Verify entity initialization was attempted
        assert mock_telegram_client.get_entity.called

        # Check error message in result
        assert "Message send failed" in result
        assert "Unexpected error" in result

    @pytest.mark.asyncio
    async def test_client_start_failure(self, tool: TelegramSendTool, mock_telegram_client: AsyncMock) -> None:
        """Test handling of Telegram client initialization failure."""
        # Setup client to fail on context manager enter
        mock_telegram_client.__aenter__.side_effect = Exception("Failed to initialize client")

        message = "Test message"
        result = await tool.func(message=message, chat_id="-1001234567890")

        # Verify client initialization was attempted
        mock_telegram_client.__aenter__.assert_called_once()
        mock_telegram_client.__aexit__.assert_not_called()  # Should not be called if enter fails

        # Check error message in result
        assert "Message send failed" in result
        assert "Failed to initialize client" in result

        # Verify no message was sent
        mock_telegram_client.send_message.assert_not_called()


@run_for_optional_imports(["telethon"], "commsagent-telegram")
class TestTelegramRetrieveTool:
    @pytest.fixture(autouse=True)
    def mock_telegram_client(self, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
        """Create a mock for the TelegramClient."""
        # Create mock instance with required attributes
        mock_instance = AsyncMock(spec=TelegramClient)

        # Mock the context manager methods
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.__aexit__.return_value = None

        # Mock entity-related methods (from base tool tests)
        mock_entity = AsyncMock()
        mock_instance.get_entity = AsyncMock(return_value=mock_entity)
        mock_instance.iter_dialogs = AsyncMock(return_value=[])
        mock_instance.iter_messages = AsyncMock(return_value=[])  # Add this here!

        # Create the mock class that returns our instance
        mock_client_cls = Mock(return_value=mock_instance)

        # Mock the TelegramClient
        monkeypatch.setattr(
            "autogen.tools.experimental.messageplatform.telegram.telegram.TelegramClient",
            mock_client_cls,
        )

        return mock_instance

    @pytest.fixture
    def tool(self) -> TelegramRetrieveTool:
        """Create a TelegramRetrieveTool instance for testing."""
        return TelegramRetrieveTool(api_id="test-api-id", api_hash="test-api-hash", chat_id="-1001234567890")

    def test_telegram_retrieve_tool_init(self) -> None:
        telegram_retrieve_tool = TelegramRetrieveTool(api_id="-100", api_hash="my_api_hash", chat_id="my_chat")
        assert telegram_retrieve_tool.name == "telegram_retrieve"
        assert (
            telegram_retrieve_tool.description
            == "Retrieves messages from a Telegram chat based on datetime/message ID and/or number of latest messages."
        )
        assert isinstance(telegram_retrieve_tool.func, Callable)  # type: ignore[arg-type]

        expected_schema = {
            "description": "Retrieves messages from a Telegram chat based on datetime/message ID and/or number of latest messages.",
            "name": "telegram_retrieve",
            "parameters": {
                "type": "object",
                "properties": {
                    "messages_since": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": "Date to retrieve messages from (ISO format) OR message ID. If None, retrieves latest messages.",
                    },
                    "maximum_messages": {
                        "anyOf": [{"type": "integer"}, {"type": "null"}],
                        "default": None,
                        "description": "Maximum number of messages to retrieve. If None, retrieves all messages since date.",
                    },
                    "search": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": "Optional string to search for in messages.",
                    },
                },
                "required": [],
            },
        }

        assert telegram_retrieve_tool.function_schema == expected_schema

    @pytest.mark.asyncio
    async def test_message_retrieval_with_search(
        self, tool: TelegramRetrieveTool, mock_telegram_client: AsyncMock
    ) -> None:
        """Test message retrieval with search parameter."""
        mock_messages = [
            Mock(
                id=123456789,
                date=datetime(2024, 1, 1, 10, 0),
                from_id="user123",
                text="Target message",
                reply_to_msg_id=None,
                forward=None,
                edit_date=None,
                media=False,
                entities=[],
            )
        ]

        mock_telegram_client.iter_messages = AsyncMock(return_value=iter(mock_messages))

        # Call the function with search parameter
        _ = await tool.func(chat_id="-1001234567890", search="Target")

        # Verify search parameter was passed
        call_args = mock_telegram_client.iter_messages.call_args
        assert call_args.kwargs["search"] == "Target"

        # Verify the entity initialization attempted to get entity
        assert mock_telegram_client.get_entity.called

    @pytest.mark.asyncio
    async def test_message_retrieval_with_limit(
        self, tool: TelegramRetrieveTool, mock_telegram_client: AsyncMock
    ) -> None:
        """Test message retrieval with maximum_messages parameter."""
        mock_messages = [
            Mock(
                id=idx,
                date=datetime(2024, 1, 1, 10, 0),
                from_id="user123",
                text=f"Message {idx}",
                reply_to_msg_id=None,
                forward=None,
                edit_date=None,
                media=False,
                entities=[],
            )
            for idx in range(1, 6)
        ]  # 5 messages

        mock_telegram_client.iter_messages = AsyncMock(return_value=iter(mock_messages))

        # Call with limit
        await tool.func(chat_id="-1001234567890", maximum_messages=3)

        # Verify limit parameter was passed
        call_args = mock_telegram_client.iter_messages.call_args
        assert call_args.kwargs["limit"] == 3
