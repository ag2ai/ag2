# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
from unittest.mock import MagicMock, Mock

import pytest

from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.tools.experimental.messageplatform import SlackRetrieveRepliesTool, SlackRetrieveTool, SlackSendTool

with optional_import_block():
    from slack_sdk.errors import SlackApiError


@run_for_optional_imports("slack_sdk", "commsagent-slack")
class TestSlackSendTool:
    @pytest.fixture(autouse=True)
    def mock_webclient(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        """Create a mock for the WebClient constructor."""
        # Create the mock instance with the desired return value
        mock_instance = Mock()
        mock_instance.chat_postMessage.return_value = {"ok": True, "ts": "1234567890.123456"}

        # Create the mock class
        mock_webclient_cls = Mock(return_value=mock_instance)

        # Patch at both the original import location and where it might be cached
        monkeypatch.setattr("slack_sdk.WebClient", mock_webclient_cls)
        monkeypatch.setattr("autogen.tools.experimental.messageplatform.slack.slack.WebClient", mock_webclient_cls)

        return mock_webclient_cls

    @pytest.fixture
    def tool(self) -> SlackSendTool:
        return SlackSendTool(bot_token="xoxb-test-token", channel_id="test-channel")

    def test_slack_send_tool_init(self) -> None:
        slack_send_tool = SlackSendTool(bot_token="my_bot_token", channel_id="my_channel")
        assert slack_send_tool.name == "slack_send"
        assert slack_send_tool.description == "Sends a message to a Slack channel."
        assert isinstance(slack_send_tool.func, Callable)  # type: ignore[arg-type]

        expected_schema = {
            "description": "Sends a message to a Slack channel.",
            "name": "slack_send",
            "parameters": {
                "properties": {"message": {"description": "Message to send to the channel.", "type": "string"}},
                "required": ["message"],
                "type": "object",
            },
        }
        assert slack_send_tool.function_schema == expected_schema

    @pytest.mark.asyncio
    async def test_successful_message_send(self, tool: SlackSendTool, mock_webclient: MagicMock) -> None:
        """Test successful message sending."""

        # Get the mock instance that will be returned when WebClient is constructed
        mock_instance = mock_webclient.return_value

        # Call the function with our message
        message = "Test message"
        result = await tool.func(message=message, bot_token="xoxb-test-token", channel_id="test-channel")

        # Verify the call and result
        mock_instance.chat_postMessage.assert_called_once_with(channel="test-channel", text=message)
        assert "Message sent successfully" in result
        assert message in result

    @pytest.mark.asyncio
    async def test_long_message_chunking(self, tool: SlackSendTool, mock_webclient: MagicMock) -> None:
        """Test that long messages are correctly chunked."""
        mock_instance = mock_webclient.return_value

        # Create a message longer than MAX_MESSAGE_LENGTH (40000)
        long_message = "x" * 45000
        result = await tool.func(message=long_message, bot_token="xoxb-test-token", channel_id="test-channel")

        # Should be called twice due to message length
        assert mock_instance.chat_postMessage.call_count == 2

        # Verify each chunk is within the MAX_MESSAGE_LENGTH limit
        calls = mock_instance.chat_postMessage.call_args_list
        for call in calls:
            assert len(call.kwargs["text"]) <= 40000

        assert "Message sent successfully" in result
        assert "chunks" in result
        assert long_message in result

    @pytest.mark.asyncio
    async def test_slack_api_error(self, tool: SlackSendTool, mock_webclient: MagicMock) -> None:
        """Test handling of SlackApiError."""
        mock_instance = mock_webclient.return_value
        mock_instance.chat_postMessage.side_effect = SlackApiError(  # type: ignore[no-untyped-call]
            message="", response={"ok": False, "error": "channel_not_found"}
        )

        result = await tool.func(message="Test message", bot_token="xoxb-test-token", channel_id="test-channel")
        assert "Message send failed" in result
        assert "channel_not_found" in result

    @pytest.mark.asyncio
    async def test_general_exception(self, tool: SlackSendTool, mock_webclient: MagicMock) -> None:
        """Test handling of general exceptions."""
        mock_instance = mock_webclient.return_value
        mock_instance.chat_postMessage.side_effect = Exception("Unexpected error")

        result = await tool.func(message="Test message", bot_token="xoxb-test-token", channel_id="test-channel")
        assert "Message send failed" in result
        assert "Unexpected error" in result

    @pytest.mark.asyncio
    async def test_failed_message_response(self, tool: SlackSendTool, mock_webclient: MagicMock) -> None:
        """Test handling of failed message response from Slack."""
        mock_instance = mock_webclient.return_value
        mock_instance.chat_postMessage.return_value = {"ok": False, "error": "invalid_auth"}

        result = await tool.func(message="Test message", bot_token="xoxb-test-token", channel_id="test-channel")
        assert "Message send failed" in result
        assert "invalid_auth" in result

    @pytest.mark.asyncio
    async def test_chunked_message_failure(self, tool: SlackSendTool, mock_webclient: MagicMock) -> None:
        """Test handling of failure during chunked message sending."""
        mock_instance = mock_webclient.return_value

        # Set up the mock to succeed on first chunk but fail on second
        mock_instance.chat_postMessage.side_effect = [
            {"ok": True, "ts": "1234567890.123456"},  # First chunk succeeds
            {"ok": False, "error": "rate_limited"},  # Second chunk fails
        ]

        # Create a message that will be split into chunks
        long_message = "x" * 45000
        result = await tool.func(message=long_message, bot_token="xoxb-test-token", channel_id="test-channel")

        assert mock_instance.chat_postMessage.call_count == 2
        assert "Message send failed on chunk 2" in result
        assert "rate_limited" in result


@run_for_optional_imports("slack_sdk", "commsagent-slack")
class TestSlackRetrieveTool:
    @pytest.fixture(autouse=True)
    def mock_webclient(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        """Create a mock for the WebClient constructor."""
        # Create the mock instance with the desired return value
        mock_instance = Mock()
        mock_instance.conversations_history.return_value = {
            "ok": True,
            "messages": [{"text": "Test message", "ts": "1234567890.123456"}],
            "has_more": False,
            "response_metadata": {"next_cursor": None},
        }

        # Create the mock class
        mock_webclient_cls = Mock(return_value=mock_instance)

        # Patch at both the original import location and where it might be cached
        monkeypatch.setattr("slack_sdk.WebClient", mock_webclient_cls)
        monkeypatch.setattr("autogen.tools.experimental.messageplatform.slack.slack.WebClient", mock_webclient_cls)

        return mock_webclient_cls

    @pytest.fixture
    def tool(self) -> SlackRetrieveTool:
        return SlackRetrieveTool(bot_token="xoxb-test-token", channel_id="test-channel")

    def test_slack_retrieve_tool_init(self) -> None:
        slack_retrieve_tool = SlackRetrieveTool(bot_token="my_bot_token", channel_id="my_channel")
        assert slack_retrieve_tool.name == "slack_retrieve"
        assert (
            slack_retrieve_tool.description
            == "Retrieves messages from a Slack channel based datetime/message ID and/or number of latest messages."
        )
        assert isinstance(slack_retrieve_tool.func, Callable)  # type: ignore[arg-type]

        expected_schema = {
            "description": "Retrieves messages from a Slack channel based datetime/message ID and/or number of latest messages.",
            "name": "slack_retrieve",
            "parameters": {
                "type": "object",
                "properties": {
                    "messages_since": {
                        "anyOf": [
                            {
                                "type": "string",
                            },
                            {
                                "type": "null",
                            },
                        ],
                        "default": None,
                        "description": "Date to retrieve messages from (ISO format) OR Slack message ID. If None, retrieves latest messages.",
                    },
                    "maximum_messages": {
                        "anyOf": [
                            {
                                "type": "integer",
                            },
                            {
                                "type": "null",
                            },
                        ],
                        "default": None,
                        "description": "Maximum number of messages to retrieve. If None, retrieves all messages since date.",
                    },
                },
                "required": [],
            },
        }

        assert slack_retrieve_tool.function_schema == expected_schema

    @pytest.mark.asyncio
    async def test_successful_message_retrieval(self, tool: SlackRetrieveTool, mock_webclient: MagicMock) -> None:
        """Test successful message retrieval without any filters."""
        mock_instance = mock_webclient.return_value

        result = await tool.func(bot_token="xoxb-test-token", channel_id="test-channel")

        # Verify the call and result
        mock_instance.conversations_history.assert_called_once_with(channel="test-channel", limit=1000)
        assert isinstance(result, dict)
        assert result["message_count"] == 1
        assert len(result["messages"]) == 1
        assert result["messages"][0]["text"] == "Test message"

    @pytest.mark.asyncio
    async def test_message_retrieval_with_date(self, tool: SlackRetrieveTool, mock_webclient: MagicMock) -> None:
        """Test message retrieval with ISO date filter."""
        mock_instance = mock_webclient.return_value

        _ = await tool.func(
            bot_token="xoxb-test-token", channel_id="test-channel", messages_since="2025-01-25T00:00:00Z"
        )

        # Verify timestamp conversion and API call
        mock_instance.conversations_history.assert_called_once()
        call_args = mock_instance.conversations_history.call_args[1]
        assert "oldest" in call_args
        assert float(call_args["oldest"]) > 0  # Verify timestamp conversion

    @pytest.mark.asyncio
    async def test_message_retrieval_with_message_id(self, tool: SlackRetrieveTool, mock_webclient: MagicMock) -> None:
        """Test message retrieval with Slack message ID filter."""
        mock_instance = mock_webclient.return_value

        _ = await tool.func(bot_token="xoxb-test-token", channel_id="test-channel", messages_since="1234567890.123456")

        # Verify direct message ID usage
        mock_instance.conversations_history.assert_called_once_with(
            channel="test-channel", limit=1000, oldest="1234567890.123456"
        )

    @pytest.mark.asyncio
    async def test_message_retrieval_with_pagination(self, tool: SlackRetrieveTool, mock_webclient: MagicMock) -> None:
        """Test message retrieval with pagination."""
        mock_instance = mock_webclient.return_value

        # Set up mock to return multiple pages
        mock_instance.conversations_history.side_effect = [
            {
                "ok": True,
                "messages": [{"text": "Message 1", "ts": "1234567890.123456"}],
                "has_more": True,
                "response_metadata": {"next_cursor": "cursor123"},
            },
            {
                "ok": True,
                "messages": [{"text": "Message 2", "ts": "1234567890.123457"}],
                "has_more": False,
                "response_metadata": {"next_cursor": None},
            },
        ]

        result = await tool.func(bot_token="xoxb-test-token", channel_id="test-channel")

        # Verify pagination handling
        assert mock_instance.conversations_history.call_count == 2
        assert result["message_count"] == 2
        assert len(result["messages"]) == 2

    @pytest.mark.asyncio
    async def test_message_retrieval_with_maximum(self, tool: SlackRetrieveTool, mock_webclient: MagicMock) -> None:
        """Test message retrieval with maximum messages limit."""
        mock_instance = mock_webclient.return_value

        await tool.func(bot_token="xoxb-test-token", channel_id="test-channel", maximum_messages=500)

        # Verify limit parameter
        mock_instance.conversations_history.assert_called_once_with(channel="test-channel", limit=500)

    @pytest.mark.asyncio
    async def test_invalid_date_format(self, tool: SlackRetrieveTool, mock_webclient: MagicMock) -> None:
        """Test handling of invalid date format."""
        result = await tool.func(bot_token="xoxb-test-token", channel_id="test-channel", messages_since="invalid-date")

        assert "Invalid date format" in result
        assert mock_webclient.return_value.conversations_history.call_count == 0

    @pytest.mark.asyncio
    async def test_slack_api_error(self, tool: SlackRetrieveTool, mock_webclient: MagicMock) -> None:
        """Test handling of SlackApiError."""
        mock_instance = mock_webclient.return_value
        mock_instance.conversations_history.side_effect = SlackApiError(  # type: ignore[no-untyped-call]
            message="", response={"ok": False, "error": "channel_not_found"}
        )

        result = await tool.func(bot_token="xoxb-test-token", channel_id="test-channel")

        assert "Message retrieval failed" in result
        assert "channel_not_found" in result

    @pytest.mark.asyncio
    async def test_general_exception(self, tool: SlackRetrieveTool, mock_webclient: MagicMock) -> None:
        """Test handling of general exceptions."""
        mock_instance = mock_webclient.return_value
        mock_instance.conversations_history.side_effect = Exception("Unexpected error")

        result = await tool.func(bot_token="xoxb-test-token", channel_id="test-channel")

        assert "Message retrieval failed" in result
        assert "Unexpected error" in result

    @pytest.mark.asyncio
    async def test_failed_message_response(self, tool: SlackRetrieveTool, mock_webclient: MagicMock) -> None:
        """Test handling of failed message response from Slack."""
        mock_instance = mock_webclient.return_value
        mock_instance.conversations_history.return_value = {"ok": False, "error": "invalid_auth"}

        result = await tool.func(bot_token="xoxb-test-token", channel_id="test-channel")

        assert "Message retrieval failed" in result
        assert "invalid_auth" in result


@run_for_optional_imports("slack_sdk", "commsagent-slack")
class TestSlackRetrieveRepliesTool:
    @pytest.fixture(autouse=True)
    def mock_webclient(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        """Create a mock for the WebClient constructor."""
        # Create the mock instance with the desired return value
        mock_instance = Mock()
        mock_instance.conversations_history.return_value = {
            "ok": True,
            "messages": [{"text": "Test message", "ts": "1234567890.123456"}],
            "has_more": False,
            "response_metadata": {"next_cursor": None},
        }

        # Create the mock class
        mock_webclient_cls = Mock(return_value=mock_instance)

        # Patch at both the original import location and where it might be cached
        monkeypatch.setattr("slack_sdk.WebClient", mock_webclient_cls)
        monkeypatch.setattr("autogen.tools.experimental.messageplatform.slack.slack.WebClient", mock_webclient_cls)

        return mock_webclient_cls

    @pytest.fixture
    def tool(self) -> SlackRetrieveRepliesTool:
        return SlackRetrieveRepliesTool(bot_token="xoxb-test-token", channel_id="test-channel")

    def test_slack_retrieve_replies_tool_init(self) -> None:
        slack_retrieve_replies_tool = SlackRetrieveRepliesTool(bot_token="my_bot_token", channel_id="my_channel")
        assert slack_retrieve_replies_tool.name == "slack_retrieve_replies"
        assert (
            slack_retrieve_replies_tool.description
            == "Retrieves replies to a specific Slack message from both threads and the channel."
        )
        assert isinstance(slack_retrieve_replies_tool.func, Callable)  # type: ignore[arg-type]

        expected_schema = {
            "description": "Retrieves replies to a specific Slack message from both threads and the channel.",
            "name": "slack_retrieve_replies",
            "parameters": {
                "type": "object",
                "properties": {
                    "mesage_ts": {
                        "type": "string",
                        "default": None,
                        "description": "Timestamp (ts) of the parent message to retrieve replies for.",
                    },
                    "min_replies": {
                        "anyOf": [
                            {
                                "type": "integer",
                            },
                            {
                                "type": "null",
                            },
                        ],
                        "default": None,
                        "description": "Minimum number of replies to wait for before returning (thread + channel). If None, returns immediately.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "default": 60,
                        "description": "Maximum time in seconds to wait for the requested number of replies.",
                    },
                    "poll_interval": {
                        "type": "integer",
                        "default": 5,
                        "description": "Time in seconds between polling attempts when waiting for replies.",
                    },
                    "include_channel_messages": {
                        "type": "boolean",
                        "default": True,
                        "description": "Whether to include messages in the channel after the original message.",
                    },
                },
                "required": [],
            },
        }

        assert slack_retrieve_replies_tool.function_schema == expected_schema

    @pytest.mark.asyncio
    async def test_successful_message_reply_retrieval(
        self, tool: SlackRetrieveRepliesTool, mock_webclient: MagicMock
    ) -> None:
        """Test successful message reply retrieval without any filters."""
        mock_instance = mock_webclient.return_value

        result = await tool.func(bot_token="xoxb-test-token", channel_id="test-channel")

        # Verify the call and result
        mock_instance.conversations_history.assert_called_once_with(channel="test-channel", limit=1000)
        assert isinstance(result, dict)
        assert result["message_count"] == 1
        assert len(result["messages"]) == 1
        assert result["messages"][0]["text"] == "Test message"

    @pytest.mark.asyncio
    async def test_message_reply_retrieval_with_ts(
        self, tool: SlackRetrieveRepliesTool, mock_webclient: MagicMock
    ) -> None:
        """Test message reply retrieval with timestamp filter."""
        mock_instance = mock_webclient.return_value

        result = await tool.func(message_ts="1234567890.123456", bot_token="xoxb-test-token", channel_id="test-channel")

        # Verify timestamp
        mock_instance.conversations_history.assert_called_once_with(
            channel="test-channel", limit=1000, oldest="1234567890.123456"
        )
        assert result["parent_message_ts"] == "1234567890.123456"

    @pytest.mark.asyncio
    async def test_message_reply_retrieval_with_pagination(
        self, tool: SlackRetrieveRepliesTool, mock_webclient: MagicMock
    ) -> None:
        """Test message reply retrieval with pagination."""
        mock_instance = mock_webclient.return_value

        # Set up mock to return multiple pages
        mock_instance.conversations_history.side_effect = [
            {
                "ok": True,
                "messages": [{"text": "Message 1", "ts": "1234567890.123456"}],
                "has_more": True,
                "response_metadata": {"next_cursor": "cursor123"},
            },
            {
                "ok": True,
                "messages": [{"text": "Message 2", "ts": "1234567890.123457"}],
                "has_more": False,
                "response_metadata": {"next_cursor": None},
            },
        ]

        result = await tool.func(bot_token="xoxb-test-token", channel_id="test-channel")

        # Verify pagination handling
        assert mock_instance.conversations_history.call_count == 2
        assert result["total_reply_count"] == 2

    @pytest.mark.asyncio
    async def test_message_reply_retrieval_with_minimum(
        self, tool: SlackRetrieveRepliesTool, mock_webclient: MagicMock
    ) -> None:
        """Test message reply retrieval with minimum reply requirement."""
        mock_instance = mock_webclient.return_value

        mock_instance.conversations_history.side_effect = [
            {
                "ok": True,
                "messages": [{"text": "Message 1", "ts": "1234567890.123456"}],
                "has_more": True,
                "response_metadata": {"next_cursor": "cursor123"},
            },
            {
                "ok": True,
                "messages": [
                    {"text": "Message 2", "ts": "1234567890.123457"},
                    {"text": "Message 3", "ts": "1234567890.123458"},
                    {"text": "Message 4", "ts": "1234567890.123459"},
                    {"text": "Message 5", "ts": "1234567890.123460"},
                ],
                "has_more": False,
                "response_metadata": {"next_cursor": None},
            },
        ]

        result = await tool.func(bot_token="xoxb-test-token", channel_id="test-channel", min_replies=5)

        # Verify that the minimum number of replies is met
        assert result["total_reply_count"] >= 5
        assert result["thread_reply_count"] + result["channel_message_count"] >= 5
        assert len(result["thread_replies"]) + len(result["channel_messages"]) >= 5

    @pytest.mark.asyncio
    async def test_message_reply_retrieval_with_timeout(
        self, tool: SlackRetrieveRepliesTool, mock_webclient: MagicMock
    ) -> None:
        """Test if tool waits for the requested amount of time before sending next message."""
        mock_instance = mock_webclient.return_value

        mock_instance.conversations_history.side_effect = [
            {
                "ok": True,
                "messages": [{"text": "Message 1", "ts": "1234567890.123456"}],
                "has_more": True,
                "response_metadata": {"next_cursor": "cursor123"},
            },
            {
                "ok": True,
                "messages": [{"text": "Message 2", "ts": "1234567890.123457"}],
                "has_more": True,
                "response_metadata": {"next_cursor": "cursor456"},
            },
        ]

        result = await tool.func(bot_token="xoxb-test-token", channel_id="test-channel", timeout_seconds=5)

        # Verify the timeout rule is met
        assert result["waited_seconds"] >= 5

    @pytest.mark.asyncio
    async def test_slack_api_error(self, tool: SlackRetrieveRepliesTool, mock_webclient: MagicMock) -> None:
        """Test handling of SlackApiError."""
        mock_instance = mock_webclient.return_value
        mock_instance.conversations_history.side_effect = SlackApiError(  # type: ignore[no-untyped-call]
            message="", response={"ok": False, "error": "channel_not_found"}
        )

        result = await tool.func(bot_token="xoxb-test-token", channel_id="test-channel")

        assert "Reply retrieval failed" in result
        assert "message retrieval failed" in result
        assert "channel_not_found" in result

    @pytest.mark.asyncio
    async def test_general_exception(self, tool: SlackRetrieveRepliesTool, mock_webclient: MagicMock) -> None:
        """Test handling of general exceptions."""
        mock_instance = mock_webclient.return_value
        mock_instance.conversations_history.side_effect = Exception("Unexpected error")

        result = await tool.func(bot_token="xoxb-test-token", channel_id="test-channel")

        assert "Reply retrieval failed" in result
        assert "message retrieval failed" in result
        assert "Unexpected error" in result

    @pytest.mark.asyncio
    async def test_failed_message_response(self, tool: SlackRetrieveRepliesTool, mock_webclient: MagicMock) -> None:
        """Test handling of failed message response from Slack."""
        mock_instance = mock_webclient.return_value
        mock_instance.conversations_history.return_value = {"ok": False, "error": "invalid_auth"}

        result = await tool.func(bot_token="xoxb-test-token", channel_id="test-channel")

        assert "Reply retrieval failed" in result
        assert "message retrieval failed" in result
        assert "invalid_auth" in result
