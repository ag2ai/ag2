# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

"""Tests for OpenAI client streaming with structured output fixes."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from autogen import OpenAIWrapper
from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.oai.client import OpenAIClient
from test.credentials import Credentials

with optional_import_block() as result:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    from openai.types.chat.chat_completion import ChatCompletionMessage, Choice
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
    from openai.types.chat.chat_completion_chunk import ChoiceDelta
    from openai.types.completion_usage import CompletionUsage


class TestAddStreamingUsageToParams:
    """Tests for the _add_streaming_usage_to_params static method."""

    def test_adds_stream_options_when_streaming(self) -> None:
        """Verify stream_options with include_usage is added when stream=True."""
        params: dict[str, Any] = {"stream": True, "messages": []}
        OpenAIClient._add_streaming_usage_to_params(params)

        assert "stream_options" in params
        assert params["stream_options"]["include_usage"] is True

    def test_does_not_modify_when_not_streaming(self) -> None:
        """Verify params unchanged when stream=False."""
        params: dict[str, Any] = {"stream": False, "messages": []}
        OpenAIClient._add_streaming_usage_to_params(params)

        assert "stream_options" not in params

    def test_does_not_modify_when_stream_not_present(self) -> None:
        """Verify params unchanged when stream key is absent."""
        params: dict[str, Any] = {"messages": []}
        OpenAIClient._add_streaming_usage_to_params(params)

        assert "stream_options" not in params

    def test_preserves_existing_stream_options(self) -> None:
        """Verify existing stream_options are preserved."""
        params: dict[str, Any] = {
            "stream": True,
            "stream_options": {"some_other_option": "value"},
            "messages": [],
        }
        OpenAIClient._add_streaming_usage_to_params(params)

        assert params["stream_options"]["some_other_option"] == "value"
        assert params["stream_options"]["include_usage"] is True

    def test_does_not_override_existing_include_usage(self) -> None:
        """Verify existing include_usage is not overridden."""
        params: dict[str, Any] = {
            "stream": True,
            "stream_options": {"include_usage": False},
            "messages": [],
        }
        OpenAIClient._add_streaming_usage_to_params(params)

        # setdefault should not override existing value
        assert params["stream_options"]["include_usage"] is False


@run_for_optional_imports(["openai"], "openai")
class TestStructuredOutputDisablesStreaming:
    """Test 1: Structured output with stream=True should disable streaming."""

    @pytest.fixture
    def mock_oai_client(self) -> MagicMock:
        """Create a mock OpenAI client."""
        mock_client = MagicMock(spec=OpenAI)
        mock_completions = MagicMock()
        mock_client.chat.completions = mock_completions

        # Mock the create method to return a ChatCompletion
        mock_completions.create.return_value = ChatCompletion(
            id="chatcmpl-test",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content='{"result": "test"}',
                        role="assistant",
                    ),
                    logprobs=None,
                )
            ],
            created=1234567890,
            model="gpt-4o-mini",
            object="chat.completion",
            usage=CompletionUsage(completion_tokens=10, prompt_tokens=5, total_tokens=15),
        )
        return mock_client

    def test_structured_output_removes_stream_param(self, mock_oai_client: MagicMock) -> None:
        """Verify that streaming is disabled when using structured output via response_format param."""
        # Use response_format in params (dict form) to trigger structured output path
        client = OpenAIClient(mock_oai_client, response_format=None)

        params: dict[str, Any] = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            # Use a Pydantic-style schema dict that triggers structured output
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "TestResponse",
                    "schema": {"type": "object", "properties": {"result": {"type": "string"}}},
                },
            },
        }

        client.create(params)

        # Check that create was called without stream and stream_options
        call_kwargs = mock_oai_client.chat.completions.create.call_args.kwargs
        assert "stream" not in call_kwargs
        assert "stream_options" not in call_kwargs

    def test_structured_output_via_params_removes_stream(self, mock_oai_client: MagicMock) -> None:
        """Verify streaming disabled when response_format is in params."""
        client = OpenAIClient(mock_oai_client, response_format=None)

        params: dict[str, Any] = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "response_format": {"type": "json_object"},
        }

        client.create(params)

        # Check that create was called without stream
        call_kwargs = mock_oai_client.chat.completions.create.call_args.kwargs
        assert "stream" not in call_kwargs
        assert "stream_options" not in call_kwargs


@run_for_optional_imports(["openai"], "openai")
class TestStreamingCapturesUsage:
    """Test 2: Streaming without structured output captures usage from last chunk."""

    @pytest.fixture
    def mock_oai_client(self) -> MagicMock:
        """Create a mock OpenAI client that returns streaming chunks."""
        mock_client = MagicMock(spec=OpenAI)
        mock_completions = MagicMock()
        mock_client.chat.completions = mock_completions
        return mock_client

    def _create_content_chunk(
        self, chunk_id: str, model: str, created: int, content: str, index: int = 0
    ) -> ChatCompletionChunk:
        """Create a ChatCompletionChunk with content."""
        return ChatCompletionChunk(
            id=chunk_id,
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content=content, role="assistant"),
                    index=index,
                    finish_reason=None,
                )
            ],
            created=created,
            model=model,
            object="chat.completion.chunk",
        )

    def _create_final_chunk_with_finish_reason(
        self, chunk_id: str, model: str, created: int, index: int = 0
    ) -> ChatCompletionChunk:
        """Create a final ChatCompletionChunk with finish_reason."""
        return ChatCompletionChunk(
            id=chunk_id,
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content=None, role=None),
                    index=index,
                    finish_reason="stop",
                )
            ],
            created=created,
            model=model,
            object="chat.completion.chunk",
        )

    def _create_usage_chunk(
        self, chunk_id: str, model: str, created: int, prompt_tokens: int, completion_tokens: int
    ) -> ChatCompletionChunk:
        """Create a ChatCompletionChunk with usage information (no choices)."""
        return ChatCompletionChunk(
            id=chunk_id,
            choices=[],
            created=created,
            model=model,
            object="chat.completion.chunk",
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def test_usage_captured_from_last_chunk(self, mock_oai_client: MagicMock) -> None:
        """Verify usage tokens are correctly accumulated from the last chunk."""
        chunk_id = "chatcmpl-stream-test"
        model = "gpt-4o-mini"
        created = 1234567890

        # Create a sequence of chunks that mimics real OpenAI streaming
        chunks = [
            self._create_content_chunk(chunk_id, model, created, "Hello"),
            self._create_content_chunk(chunk_id, model, created, " world"),
            self._create_content_chunk(chunk_id, model, created, "!"),
            self._create_final_chunk_with_finish_reason(chunk_id, model, created),
            self._create_usage_chunk(chunk_id, model, created, prompt_tokens=25, completion_tokens=3),
        ]

        mock_oai_client.chat.completions.create.return_value = iter(chunks)

        client = OpenAIClient(mock_oai_client, response_format=None)

        params: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True,
        }

        response = client.create(params)

        # Verify usage was captured from the last chunk
        assert response.usage is not None
        assert response.usage.prompt_tokens == 25
        assert response.usage.completion_tokens == 3
        assert response.usage.total_tokens == 28

        # Verify content was accumulated
        assert response.choices[0].message.content == "Hello world!"

    def test_stream_options_added_to_params(self, mock_oai_client: MagicMock) -> None:
        """Verify that stream_options with include_usage is added to params."""
        chunk_id = "chatcmpl-stream-test"
        model = "gpt-4o-mini"
        created = 1234567890

        chunks = [
            self._create_content_chunk(chunk_id, model, created, "Hi"),
            self._create_final_chunk_with_finish_reason(chunk_id, model, created),
            self._create_usage_chunk(chunk_id, model, created, prompt_tokens=10, completion_tokens=1),
        ]

        mock_oai_client.chat.completions.create.return_value = iter(chunks)

        client = OpenAIClient(mock_oai_client, response_format=None)

        params: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }

        client.create(params)

        # Check that stream_options was added to the call
        call_kwargs = mock_oai_client.chat.completions.create.call_args.kwargs
        assert "stream_options" in call_kwargs
        assert call_kwargs["stream_options"]["include_usage"] is True


@run_for_optional_imports(["openai"], "openai")
class TestStreamingHandlesInvalidChunks:
    """Test 3: Non-ChatCompletionChunk objects are handled gracefully."""

    @pytest.fixture
    def mock_oai_client(self) -> MagicMock:
        """Create a mock OpenAI client."""
        mock_client = MagicMock(spec=OpenAI)
        mock_completions = MagicMock()
        mock_client.chat.completions = mock_completions
        return mock_client

    def _create_content_chunk(self, chunk_id: str, model: str, created: int, content: str) -> ChatCompletionChunk:
        """Create a ChatCompletionChunk with content."""
        return ChatCompletionChunk(
            id=chunk_id,
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content=content, role="assistant"),
                    index=0,
                    finish_reason=None,
                )
            ],
            created=created,
            model=model,
            object="chat.completion.chunk",
        )

    def _create_final_chunk_with_finish_reason(self, chunk_id: str, model: str, created: int) -> ChatCompletionChunk:
        """Create a final chunk with finish_reason."""
        return ChatCompletionChunk(
            id=chunk_id,
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(content=None, role=None),
                    index=0,
                    finish_reason="stop",
                )
            ],
            created=created,
            model=model,
            object="chat.completion.chunk",
        )

    def _create_usage_chunk(self, chunk_id: str, model: str, created: int) -> ChatCompletionChunk:
        """Create a chunk with usage information (no choices)."""
        return ChatCompletionChunk(
            id=chunk_id,
            choices=[],
            created=created,
            model=model,
            object="chat.completion.chunk",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )

    def test_skips_non_chunk_objects(self, mock_oai_client: MagicMock) -> None:
        """Verify graceful handling of unexpected chunk types."""
        chunk_id = "chatcmpl-test"
        model = "gpt-4o-mini"
        created = 1234567890

        # Create a mix of valid chunks and invalid objects
        valid_chunk_1 = self._create_content_chunk(chunk_id, model, created, "Hello")
        invalid_object = {"some": "dict"}  # Not a ChatCompletionChunk
        valid_chunk_2 = self._create_content_chunk(chunk_id, model, created, " there")
        invalid_string = "not a chunk"  # Also not valid
        finish_chunk = self._create_final_chunk_with_finish_reason(chunk_id, model, created)
        usage_chunk = self._create_usage_chunk(chunk_id, model, created)

        chunks = [valid_chunk_1, invalid_object, valid_chunk_2, invalid_string, finish_chunk, usage_chunk]

        mock_oai_client.chat.completions.create.return_value = iter(chunks)

        client = OpenAIClient(mock_oai_client, response_format=None)

        params: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Test"}],
            "stream": True,
        }

        # Should not raise an exception
        response = client.create(params)

        # Valid content should still be accumulated
        assert response.choices[0].message.content == "Hello there"
        # Usage should be captured
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 2

    def test_logs_debug_for_invalid_chunks(self, mock_oai_client: MagicMock) -> None:
        """Verify that debug logging is called for invalid chunk types."""
        chunk_id = "chatcmpl-test"
        model = "gpt-4o-mini"
        created = 1234567890

        valid_chunk = self._create_content_chunk(chunk_id, model, created, "Hi")
        invalid_object = MagicMock()  # Not a ChatCompletionChunk
        finish_chunk = self._create_final_chunk_with_finish_reason(chunk_id, model, created)
        usage_chunk = self._create_usage_chunk(chunk_id, model, created)

        chunks = [valid_chunk, invalid_object, finish_chunk, usage_chunk]
        mock_oai_client.chat.completions.create.return_value = iter(chunks)

        client = OpenAIClient(mock_oai_client, response_format=None)

        params: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Test"}],
            "stream": True,
        }

        with patch("autogen.oai.client.logger") as mock_logger:
            client.create(params)

            # Verify debug was called for the invalid chunk
            mock_logger.debug.assert_called()
            debug_calls = [str(call) for call in mock_logger.debug.call_args_list]
            assert any("Skipping unexpected chunk type" in call for call in debug_calls)


# Pydantic model for integration tests
class SimpleResponse(BaseModel):
    """Simple response model for testing structured output."""

    answer: str
    confidence: float


@run_for_optional_imports(["openai"], "openai")
class TestStreamingStructuredOutputIntegration:
    """Integration tests for streaming with structured output using real OpenAI API calls.

    These tests verify the fix that automatically disables streaming when structured
    output is requested, and verifies usage metrics are captured for regular streaming.
    """

    @pytest.mark.openai
    def test_streaming_with_structured_output_pydantic(self, credentials_gpt_4o_mini: Credentials) -> None:
        """Test that streaming with Pydantic structured output works correctly.

        When stream=True and response_format is a Pydantic model, streaming should
        be automatically disabled internally and a valid structured response returned.
        """
        config_list = credentials_gpt_4o_mini.config_list

        # Add stream=True and response_format to config
        for config in config_list:
            config["stream"] = True
            config["response_format"] = SimpleResponse

        client = OpenAIWrapper(config_list=config_list, cache_seed=None)

        response = client.create(
            messages=[{"role": "user", "content": "What is 2+2? Reply with answer and confidence (0-1)."}],
        )

        # Verify response is valid
        assert response is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

        # Verify the response can be parsed as the structured output model
        content = response.choices[0].message.content
        parsed = SimpleResponse.model_validate_json(content)
        assert parsed.answer is not None
        assert 0 <= parsed.confidence <= 1

    @pytest.mark.openai
    def test_streaming_with_structured_output_json_schema(self, credentials_gpt_4o_mini: Credentials) -> None:
        """Test that streaming with JSON schema structured output works correctly.

        When stream=True and response_format is a JSON schema dict, streaming should
        be automatically disabled internally and a valid JSON response returned.
        """
        config_list = credentials_gpt_4o_mini.config_list

        # Use Pydantic's model_json_schema() for the JSON schema format
        # This matches the pattern used in test_structured_output.py
        json_schema = SimpleResponse.model_json_schema()

        # Add stream=True and response_format to config
        for config in config_list:
            config["stream"] = True
            config["response_format"] = json_schema

        client = OpenAIWrapper(config_list=config_list, cache_seed=None)

        response = client.create(
            messages=[{"role": "user", "content": "What is 2+2? Reply with answer and confidence (0-1)."}],
        )

        # Verify response is valid
        assert response is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

        # Verify the response is valid JSON matching schema
        content = response.choices[0].message.content
        parsed = json.loads(content)
        assert "answer" in parsed
        assert "confidence" in parsed
        assert isinstance(parsed["confidence"], (int, float))

    @pytest.mark.openai
    def test_streaming_without_structured_output_captures_usage(self, credentials_gpt_4o_mini: Credentials) -> None:
        """Test that streaming without structured output correctly captures usage metrics.

        When stream=True without response_format, streaming should work normally
        and usage metrics should be captured from the final chunk.
        """
        config_list = credentials_gpt_4o_mini.config_list

        # Add stream=True without response_format
        for config in config_list:
            config["stream"] = True

        client = OpenAIWrapper(config_list=config_list, cache_seed=None)

        response = client.create(
            messages=[{"role": "user", "content": "Say hello in exactly 5 words."}],
        )

        # Verify response is valid
        assert response is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

        # Verify usage metrics were captured
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens
