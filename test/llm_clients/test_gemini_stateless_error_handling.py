# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for GeminiStatelessClient error handling and edge cases.

These tests verify that the client properly handles:
1. Empty responses from Gemini API
2. Missing content in response messages
3. Malformed response structures
4. Image/audio generation model edge cases
"""

from unittest.mock import Mock

import pytest

from autogen.llm_clients.gemini_stateless_client import GeminiStatelessClient
from autogen.llm_clients.models.content_blocks import ImageContent, TextContent
from autogen.llm_clients.models.unified_message import UnifiedMessage
from autogen.llm_clients.models.unified_response import UnifiedResponse


class TestGeminiStatelessErrorHandling:
    """Test error handling for Gemini client edge cases."""

    def test_message_retrieval_raises_on_empty_response(self):
        """message_retrieval() raises ValueError when response has no messages."""
        client = GeminiStatelessClient(api_key="test_key")

        # Create response with no messages
        response = UnifiedResponse(
            id="test_id",
            model="gemini-2.5-flash",
            provider="gemini",
            messages=[],  # Empty messages list
            usage={"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
            finish_reason="stop",
            status="completed",
            cost=0.0,
        )

        # Should raise ValueError with helpful message
        with pytest.raises(ValueError, match="Gemini response contains no messages"):
            client.message_retrieval(response)

    def test_message_retrieval_raises_on_empty_content(self):
        """message_retrieval() raises ValueError when all messages have empty content."""
        client = GeminiStatelessClient(api_key="test_key")

        # Create response with message but empty content
        response = UnifiedResponse(
            id="test_id",
            model="gemini-2.5-flash-image",
            provider="gemini",
            messages=[
                UnifiedMessage(role="assistant", content=[])  # Empty content list
            ],
            usage={"prompt_tokens": 25, "completion_tokens": 0, "total_tokens": 25},
            finish_reason="stop",
            status="completed",
            cost=0.0,
            provider_metadata={"model_version": "gemini-2.5-flash-image"},
        )

        # Should raise ValueError with diagnostic info
        with pytest.raises(ValueError) as exc_info:
            client.message_retrieval(response)

        error_msg = str(exc_info.value)
        assert "empty messages" in error_msg
        assert "gemini-2.5-flash-image" in error_msg
        assert "Image/audio generation model" in error_msg or "unsupported format" in error_msg

    def test_message_retrieval_handles_text_only_response(self):
        """message_retrieval() correctly extracts text from text-only response."""
        client = GeminiStatelessClient(api_key="test_key")

        response = UnifiedResponse(
            id="test_id",
            model="gemini-2.5-flash",
            provider="gemini",
            messages=[
                UnifiedMessage(
                    role="assistant",
                    content=[TextContent(type="text", text="Hello, world!")],
                )
            ],
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            finish_reason="stop",
            status="completed",
            cost=0.0,
        )

        result = client.message_retrieval(response)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "Hello, world!"

    def test_message_retrieval_handles_multimodal_response(self):
        """message_retrieval() correctly extracts multimodal content."""
        client = GeminiStatelessClient(api_key="test_key")

        response = UnifiedResponse(
            id="test_id",
            model="gemini-2.5-flash",
            provider="gemini",
            messages=[
                UnifiedMessage(
                    role="assistant",
                    content=[
                        TextContent(type="text", text="Here's an image:"),
                        ImageContent(type="image", image_url="data:image/png;base64,iVBORw0KGgo="),
                    ],
                )
            ],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            finish_reason="stop",
            status="completed",
            cost=0.0,
        )

        result = client.message_retrieval(response)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert result[0]["role"] == "assistant"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image_url"


class TestGeminiMediaGenerationModels:
    """Test media generation model special handling."""

    def test_system_instruction_skipped_for_tts_models(self):
        """TTS models skip system_instruction to avoid 500 errors."""
        client = GeminiStatelessClient(api_key="test_key")

        # Build config with system_instruction for TTS model
        params = {
            "model": "gemini-2.5-flash-preview-tts",
            "messages": [
                {"role": "system", "content": "You are a TTS generator"},
                {"role": "user", "content": "Say hello"},
            ],
            "response_modalities": ["AUDIO"],
            "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}},
        }

        # Extract system instruction
        system_instruction = client._extract_system_instruction(params["messages"])
        assert system_instruction == "You are a TTS generator"

        # Build config - should skip system_instruction for TTS model
        config = client._build_generation_config(params, system_instruction)

        # Verify system_instruction is NOT in config (would cause 500 error)
        assert config.system_instruction is None, "TTS models should not have system_instruction (causes 500 errors)"

    def test_system_instruction_preserved_for_image_models(self):
        """Image generation models DO support system_instruction."""
        client = GeminiStatelessClient(api_key="test_key")

        params = {
            "model": "gemini-2.5-flash-image",
            "messages": [
                {"role": "system", "content": "You are an image generator"},
                {"role": "user", "content": "Create a red circle"},
            ],
            "response_modalities": ["Image"],
            "image_config": {"aspect_ratio": "1:1"},
        }

        system_instruction = client._extract_system_instruction(params["messages"])
        config = client._build_generation_config(params, system_instruction)

        # Verify system_instruction IS preserved for image models
        assert config.system_instruction == "You are an image generator", (
            "Image generation models DO support system_instruction"
        )

    def test_system_instruction_preserved_for_text_models(self):
        """Standard text models preserve system_instruction."""
        client = GeminiStatelessClient(api_key="test_key")

        params = {
            "model": "gemini-2.5-flash",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
            ],
        }

        system_instruction = client._extract_system_instruction(params["messages"])
        config = client._build_generation_config(params, system_instruction)

        # Verify system_instruction IS in config for normal models
        assert config.system_instruction == "You are a helpful assistant"

    def test_tts_model_extracts_last_user_message(self):
        """TTS models automatically extract only the last user message from multi-turn conversations."""
        from autogen.llm_clients.gemini_stateless_client import GeminiStatelessClient

        client = GeminiStatelessClient(api_key="test_key")

        # Multi-turn conversation (2 user messages)
        messages = [
            {"role": "user", "content": "Say hello"},
            {"role": "assistant", "content": "<audio>"},
            {"role": "user", "content": "Say goodbye"},  # Last message - should be extracted
        ]

        # Should extract only the last user message
        result = client._extract_last_user_message_for_tts(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Say goodbye"

    def test_tts_model_preserves_system_message(self):
        """TTS models preserve system message along with last user message."""
        from autogen.llm_clients.gemini_stateless_client import GeminiStatelessClient

        client = GeminiStatelessClient(api_key="test_key")

        # Conversation with system message
        messages = [
            {"role": "system", "content": "You are a TTS generator"},
            {"role": "user", "content": "Say hello"},
            {"role": "assistant", "content": "<audio>"},
            {"role": "user", "content": "Say goodbye"},
        ]

        # Should extract system + last user message
        result = client._extract_last_user_message_for_tts(messages)

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a TTS generator"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Say goodbye"

    def test_tts_model_single_turn_unchanged(self):
        """TTS models with single-turn conversations remain unchanged."""
        from autogen.llm_clients.gemini_stateless_client import GeminiStatelessClient

        client = GeminiStatelessClient(api_key="test_key")

        # Single-turn conversation (1 user message)
        messages = [
            {"role": "user", "content": "Say hello"},
        ]

        # Should return same message list
        result = client._extract_last_user_message_for_tts(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Say hello"


class TestOpenAIWrapperErrorHandling:
    """Test OpenAIWrapper's extract_text_or_completion_object error handling."""

    def test_extract_raises_on_empty_message_retrieval_result(self):
        """extract_text_or_completion_object() raises ValueError on empty result."""
        from autogen.oai.client import OpenAIWrapper

        wrapper = OpenAIWrapper()

        # Create mock client that returns empty list
        mock_client = Mock()
        mock_client.message_retrieval = Mock(return_value=[])

        # Create mock response WITHOUT message_retrieval_function to force Option 2 path
        mock_response = Mock(spec=["id", "model"])  # Only spec specific attrs
        mock_response.id = "test_response_id"
        mock_response.model = "gemini-2.5-flash-image"

        # Register response in metadata
        wrapper._response_metadata[mock_response.id] = {"client": mock_client}

        # Should raise ValueError with diagnostic info
        with pytest.raises(ValueError, match="returned empty result"):
            wrapper.extract_text_or_completion_object(mock_response)

    def test_extract_reraises_with_context_on_client_error(self):
        """extract_text_or_completion_object() re-raises errors with context."""
        from autogen.oai.client import OpenAIWrapper

        wrapper = OpenAIWrapper()

        # Create mock client that raises ValueError
        mock_client = Mock()
        original_error = ValueError("Gemini response contains only empty messages")
        mock_client.message_retrieval = Mock(side_effect=original_error)

        # Create mock response WITHOUT message_retrieval_function to force Option 2 path
        mock_response = Mock(spec=["id", "model"])  # Only spec specific attrs
        mock_response.id = "test_response_id"
        mock_response.model = "gemini-2.5-flash-image"

        # Register response in metadata
        wrapper._response_metadata[mock_response.id] = {"client": mock_client}

        # Should re-raise ValueError as-is (line 1577-1579 in client.py)
        with pytest.raises(ValueError, match="Gemini response contains only empty messages"):
            wrapper.extract_text_or_completion_object(mock_response)
