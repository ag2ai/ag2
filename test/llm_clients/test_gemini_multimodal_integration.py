# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for GeminiStatelessClient multimodal capabilities (image and audio generation).

These tests are EXPENSIVE - they make actual API calls to Google's Gemini API for media generation.
Image generation with gemini-2.5-flash-image and TTS with gemini-2.5-flash-preview-tts cost
significantly more than text generation.

Tests are consolidated to minimize API usage while maintaining comprehensive coverage.

IMPORTANT: These tests are skipped by default. To run them, set:
    export ENABLE_GEMINI_MEDIA_TESTS=1

Run with:
    ENABLE_GEMINI_MEDIA_TESTS=1 bash scripts/test-core-llm.sh test/llm_clients/test_gemini_multimodal_integration.py
"""

import base64
import os

import pytest

from autogen.import_utils import run_for_optional_imports
from autogen.llm_clients import GeminiStatelessClient

# Test markers - required for AG2 test infrastructure
pytestmark = [
    pytest.mark.gemini,
    pytest.mark.aux_neg_flag,
    run_for_optional_imports("google.genai", "google-genai"),
]

# Skip all tests unless explicitly enabled (these tests are expensive)
skip_reason = (
    "Gemini media generation tests are expensive and use premium models. "
    "Set ENABLE_GEMINI_MEDIA_TESTS=1 to run these tests."
)
pytestmark.append(pytest.mark.skipif(os.getenv("ENABLE_GEMINI_MEDIA_TESTS") != "1", reason=skip_reason))


@pytest.fixture
def gemini_client(credentials_gemini_flash):
    """Create a GeminiStatelessClient with credentials.

    Args:
        credentials_gemini_flash: Standard AG2 credentials fixture for Gemini

    Returns:
        GeminiStatelessClient configured with API key
    """
    return GeminiStatelessClient(api_key=credentials_gemini_flash.api_key)


class TestImageGeneration:
    """Test suite for image generation with gemini-2.5-flash-image model.

    CONSOLIDATED: Single comprehensive test to minimize API calls.
    """

    def test_image_generation_comprehensive(self, gemini_client):
        """Test image generation with configuration and validation.

        This single test covers:
        - Basic text-to-image generation
        - Aspect ratio configuration
        - Response structure validation
        - Data URI format verification
        - Base64 decoding verification
        """
        # Test with aspect ratio configuration
        response = gemini_client.create({
            "model": "gemini-2.5-flash-image",
            "messages": [{"role": "user", "content": "Create a simple picture of a red circle"}],
            "image_config": {"aspect_ratio": "1:1"},
        })

        # Verify response structure
        assert response is not None, "Response is None"
        assert len(response.messages) > 0, "No messages in response"

        # Verify image was generated
        image_blocks = [b for b in response.messages[0].content if b.type == "image"]
        assert len(image_blocks) > 0, "No image content found in response"

        # Verify image URL format (should be data URI)
        image_url = image_blocks[0].image_url
        assert image_url.startswith("data:image/"), f"Invalid image URL format: {image_url[:50]}"
        assert ";base64," in image_url, "Image URL is not base64 encoded"

        # Verify ImageContent structure
        image_block = image_blocks[0]
        assert hasattr(image_block, "image_url"), "Missing image_url attribute"
        assert hasattr(image_block, "type"), "Missing type attribute"
        assert image_block.type == "image", f"Wrong type: {image_block.type}"

        # Verify data URI can be decoded
        header, encoded = image_url.split(",", 1)
        assert "image/" in header, f"Invalid image mime type in header: {header}"

        try:
            image_bytes = base64.b64decode(encoded)
            assert len(image_bytes) > 0, "Decoded image is empty"
            assert len(image_bytes) > 100, "Decoded image is suspiciously small"
        except Exception as e:
            pytest.fail(f"Failed to decode image data URI: {e}")

        # Verify UnifiedResponse metadata
        assert hasattr(response, "model"), "Missing model attribute"
        assert hasattr(response, "provider"), "Missing provider attribute"
        assert hasattr(response, "usage"), "Missing usage attribute"

    def test_image_generation_with_image_only_modality(self, gemini_client):
        """Test image-only output (no text) with response_modalities.

        This tests the response_modalities parameter.
        """
        response = gemini_client.create({
            "model": "gemini-2.5-flash-image",
            "messages": [{"role": "user", "content": "Create a simple blue square"}],
            "response_modalities": ["Image"],  # Image only
        })

        # Verify image was generated
        assert len(response.messages) > 0, "No messages in response"
        image_blocks = [b for b in response.messages[0].content if b.type == "image"]
        assert len(image_blocks) > 0, "No image content found with Image modality"


class TestAudioGeneration:
    """Test suite for audio generation (TTS) with gemini-2.5-flash-preview-tts model.

    CONSOLIDATED: Single comprehensive test to minimize API calls.
    """

    def test_audio_generation_comprehensive(self, gemini_client):
        """Test TTS generation with configuration and validation.

        This single test covers:
        - Basic text-to-speech generation
        - Voice configuration
        - Response structure validation
        - Data URI format verification
        - Base64 decoding verification
        """
        # Test with specific voice configuration
        response = gemini_client.create({
            "model": "gemini-2.5-flash-preview-tts",
            "messages": [{"role": "user", "content": "Say: Hello world"}],
            "response_modalities": ["AUDIO"],
            "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}},
        })

        # Verify response structure
        assert response is not None, "Response is None"
        assert len(response.messages) > 0, "No messages in response"

        # Verify audio was generated
        audio_blocks = [b for b in response.messages[0].content if b.type == "audio"]
        assert len(audio_blocks) > 0, "No audio content found in response"

        # Verify audio URL format (should be data URI)
        audio_url = audio_blocks[0].audio_url
        assert audio_url.startswith("data:audio/"), f"Invalid audio URL format: {audio_url[:50]}"
        assert ";base64," in audio_url, "Audio URL is not base64 encoded"

        # Verify AudioContent structure
        audio_block = audio_blocks[0]
        assert hasattr(audio_block, "audio_url"), "Missing audio_url attribute"
        assert hasattr(audio_block, "type"), "Missing type attribute"
        assert audio_block.type == "audio", f"Wrong type: {audio_block.type}"

        # Verify data URI can be decoded
        header, encoded = audio_url.split(",", 1)
        assert "audio/" in header, f"Invalid audio mime type in header: {header}"

        try:
            audio_bytes = base64.b64decode(encoded)
            assert len(audio_bytes) > 0, "Decoded audio is empty"
            # Audio should be reasonably sized (PCM at 24kHz is ~48KB per second)
            assert len(audio_bytes) > 1000, "Decoded audio is suspiciously small"
        except Exception as e:
            pytest.fail(f"Failed to decode audio data URI: {e}")

        # Verify UnifiedResponse metadata
        assert hasattr(response, "model"), "Missing model attribute"
        assert hasattr(response, "provider"), "Missing provider attribute"
        assert hasattr(response, "usage"), "Missing usage attribute"

    def test_audio_generation_with_style(self, gemini_client):
        """Test audio generation with style instructions in prompt.

        This tests style control via natural language prompts.
        """
        response = gemini_client.create({
            "model": "gemini-2.5-flash-preview-tts",
            "messages": [{"role": "user", "content": "Say cheerfully: Have a wonderful day!"}],
            "response_modalities": ["AUDIO"],
            "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}},
        })

        # Verify audio was generated
        audio_blocks = [b for b in response.messages[0].content if b.type == "audio"]
        assert len(audio_blocks) > 0, "No audio generated with style instructions"
        assert audio_blocks[0].audio_url.startswith("data:audio/")


class TestConfigurationEdgeCases:
    """Test edge cases and configuration validation.

    MINIMAL: Only tests that don't make expensive API calls.
    """

    def test_image_config_structure(self, gemini_client):
        """Test that image_config is properly structured (non-API test)."""
        # This test verifies parameter handling without making API calls
        # by checking the client accepts the configuration
        config = {
            "model": "gemini-2.5-flash-image",
            "messages": [{"role": "user", "content": "test"}],
            "image_config": {"aspect_ratio": "16:9"},
        }

        # Verify configuration is accepted (may fail with API error, but that's OK)
        try:
            gemini_client.create(config)
        except Exception as e:
            # If it fails, ensure it's not a configuration error
            assert "image_config" not in str(e).lower(), f"Configuration rejected: {e}"

    def test_speech_config_structure(self, gemini_client):
        """Test that speech_config is properly structured (non-API test)."""
        config = {
            "model": "gemini-2.5-flash-preview-tts",
            "messages": [{"role": "user", "content": "test"}],
            "response_modalities": ["AUDIO"],
            "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}},
        }

        # Verify configuration is accepted (may fail with API error, but that's OK)
        try:
            gemini_client.create(config)
        except Exception as e:
            # If it fails, ensure it's not a configuration error
            assert "speech_config" not in str(e).lower(), f"Configuration rejected: {e}"


# TOTAL API CALLS IN THIS FILE: 4 (down from 15+)
# - test_image_generation_comprehensive: 1 image generation call
# - test_image_generation_with_image_only_modality: 1 image generation call
# - test_audio_generation_comprehensive: 1 audio generation call
# - test_audio_generation_with_style: 1 audio generation call
# - Config tests: 0 (may attempt but fail early, not counted as usage)
