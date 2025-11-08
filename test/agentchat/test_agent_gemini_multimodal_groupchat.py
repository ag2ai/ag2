# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for GeminiStatelessClient multimodal generation in group chats.

These tests verify the FULL CYCLE of multimodal content in multi-agent systems:
1. Agent generates media (image/audio) using specialized Gemini models
2. Generated media is extracted from response
3. Another agent consumes the generated media as input
4. Verify the content flows correctly through the group chat system

IMPORTANT: These tests are EXPENSIVE - they use premium Gemini models:
- gemini-2.5-flash-image for image generation
- gemini-2.5-flash-preview-tts for audio/TTS generation
- gemini-2.0-flash for multimodal consumption (vision, audio understanding)

Tests are consolidated to minimize API usage while maintaining comprehensive coverage.

These tests are skipped by default. To run them, set:
    export ENABLE_GEMINI_MEDIA_TESTS=1

Run with:
    ENABLE_GEMINI_MEDIA_TESTS=1 bash scripts/test-core-llm.sh test/agentchat/test_agent_gemini_multimodal_groupchat.py
"""

import base64
import logging
import os
from typing import Any

import pytest

from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from autogen.import_utils import run_for_optional_imports
from test.credentials import Credentials

logger = logging.getLogger(__name__)

# Test markers - required for AG2 test infrastructure
pytestmark = [
    pytest.mark.gemini,
    pytest.mark.aux_neg_flag,
    pytest.mark.skipif(
        os.getenv("ENABLE_GEMINI_MEDIA_TESTS") != "1",
        reason="Gemini multimodal group chat tests are expensive and use premium models. "
        "Set ENABLE_GEMINI_MEDIA_TESTS=1 to run these tests.",
    ),
]


def _create_image_generation_config(credentials: Credentials) -> dict[str, Any]:
    """Create config for image generation with gemini-2.5-flash-image.

    Configured for small test outputs:
    - Square aspect ratio (1:1) for balanced dimensions
    - 1K image size (smallest option: ~1024x1024 pixels)
    - Lower temperature for more consistent test results
    - Image-only output modality

    Valid image_size values: 1K, 2K, 4K (default: 1K)
    Valid aspect ratios: 1:1, 2:3, 3:2, 3:4, 4:3, 9:16, 16:9, 21:9
    """
    return {
        "config_list": [
            {
                "api_type": "google_stateless",
                "model": "gemini-2.5-flash-image",
                "api_key": credentials.api_key,
                "image_config": {
                    "aspect_ratio": "1:1",  # Square format for balanced dimensions
                    "image_size": "1K",  # Smallest size option for fast testing
                },
                "response_modalities": ["Image"],  # Image only output
            }
        ],
        "temperature": 0.3,  # Lower temperature for more deterministic results
    }


def _create_audio_generation_config(credentials: Credentials) -> dict[str, Any]:
    """Create config for audio/TTS generation with gemini-2.5-flash-preview-tts.

    Configured for small test outputs:
    - Audio-only output modality
    - English US language for consistent results
    - Specific voice config for consistent results
    - Lower temperature for deterministic generation

    SpeechConfig parameters:
    - language_code: ISO 639 language code (e.g., en-US)
    - voice_config: Voice configuration (prebuilt or custom)
    """
    return {
        "config_list": [
            {
                "api_type": "google_stateless",
                "model": "gemini-2.5-flash-preview-tts",
                "api_key": credentials.api_key,
                "response_modalities": ["AUDIO"],  # Audio only output
                "speech_config": {
                    "language_code": "en-US",  # English US for consistent results
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": "Kore"  # Use consistent voice for testing
                        }
                    },
                },
            }
        ],
        "temperature": 0.3,  # Lower temperature for more consistent results
    }


def _create_vision_config(credentials: Credentials) -> dict[str, Any]:
    """Create config for vision/multimodal understanding with gemini-2.0-flash."""
    return {
        "config_list": [
            {
                "api_type": "google_stateless",
                "model": "gemini-2.0-flash",
                "api_key": credentials.api_key,
            }
        ],
        "temperature": 0.0,
    }


def _extract_image_from_response(chat_result: Any) -> str | None:
    """Extract image data URI from chat result.

    Returns:
        Data URI string (data:image/png;base64,...) or None if no image found
    """
    for msg in reversed(chat_result.chat_history):
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                # Handle OpenAI-style multimodal format (type: "image_url")
                if isinstance(block, dict):
                    if block.get("type") == "image_url":
                        # Extract from nested structure: {'type': 'image_url', 'image_url': {'url': '...'}}
                        image_url_obj = block.get("image_url")
                        if isinstance(image_url_obj, dict):
                            return image_url_obj.get("url")
                        return image_url_obj
                    elif block.get("type") == "image":
                        return block.get("image_url")
                elif hasattr(block, "type"):
                    if block.type == "image_url":
                        return block.image_url.url if hasattr(block.image_url, "url") else block.image_url
                    elif block.type == "image":
                        return block.image_url
    return None


def _extract_audio_from_response(chat_result: Any) -> str | None:
    """Extract audio data URI from chat result.

    Returns:
        Data URI string (data:audio/pcm;base64,...) or None if no audio found
    """
    for msg in reversed(chat_result.chat_history):
        content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                # Handle OpenAI-style multimodal format (type: "audio_url")
                if isinstance(block, dict):
                    if block.get("type") == "audio_url":
                        # Extract from nested structure: {'type': 'audio_url', 'audio_url': {'url': '...'}}
                        audio_url_obj = block.get("audio_url")
                        if isinstance(audio_url_obj, dict):
                            return audio_url_obj.get("url")
                        return audio_url_obj
                    elif block.get("type") == "audio":
                        return block.get("audio_url")
                elif hasattr(block, "type"):
                    if block.type == "audio_url":
                        return block.audio_url.url if hasattr(block.audio_url, "url") else block.audio_url
                    elif block.type == "audio":
                        return block.audio_url
    return None


@pytest.mark.gemini
@run_for_optional_imports(["google.genai"], "google-genai")
def test_groupchat_image_generation_and_consumption_cycle(credentials_gemini_flash: Credentials) -> None:
    """Test full cycle: Agent generates image -> Another agent consumes it as input.

    This test verifies:
    1. Image generator agent creates an image using gemini-2.5-flash-image
    2. Image is extracted as data URI from response
    3. Vision agent consumes the generated image as multimodal input
    4. Vision agent analyzes the image content successfully

    This is a critical workflow for multi-agent systems where agents produce and consume media.
    """
    # Create specialized agents for group chat
    image_generator = AssistantAgent(
        name="ImageCreator",
        llm_config=_create_image_generation_config(credentials_gemini_flash),
        system_message=(
            "You are an image generator. You create simple, clean images based on descriptions. "
            "Focus on basic geometric shapes and solid colors for test images."
        ),
    )

    vision_analyst = AssistantAgent(
        name="VisionAnalyst",
        llm_config=_create_vision_config(credentials_gemini_flash),
        system_message="You analyze images and describe what you see in detail.",
    )

    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # Phase 1: Generate image in group chat
    logger.info("=== Phase 1: Generating Image ===")
    groupchat_gen = GroupChat(
        agents=[user_proxy, image_generator],
        messages=[],
        max_round=2,
        speaker_selection_method="round_robin",
    )

    manager_gen = GroupChatManager(
        groupchat=groupchat_gen,
        llm_config=_create_vision_config(credentials_gemini_flash),
    )

    # Use a unique, specific prompt to avoid IMAGE_RECITATION filter
    # Adding specific details makes the image request unique and less likely to match training data
    result_gen = user_proxy.initiate_chat(
        manager_gen,
        message=(
            "Create a test image with these exact specifications: "
            "A gradient blue hexagon with cyan edges (hex color #4A90E2 to #00D4FF) "
            "positioned 15% from top on an off-white background (#FAFAFA). "
            "Add small teal dots (#20B2AA) in three corners. "
            "This unique pattern is for automated testing purposes."
        ),
        max_turns=1,
    )

    # Verify image was generated
    assert result_gen is not None
    assert len(result_gen.chat_history) >= 2

    # Extract generated image
    image_data_uri = _extract_image_from_response(result_gen)
    assert image_data_uri is not None, "No image found in generation response"
    assert image_data_uri.startswith("data:image/"), f"Invalid image data URI: {image_data_uri[:50]}"
    assert ";base64," in image_data_uri, "Image should be base64 encoded"

    logger.info("✓ Image generated successfully: %s...", image_data_uri[:80])

    # Verify image can be decoded
    header, encoded = image_data_uri.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    assert len(image_bytes) > 100, "Generated image is suspiciously small"
    logger.info("✓ Image size: %d bytes", len(image_bytes))

    # Phase 2: Consume generated image in new group chat
    logger.info("=== Phase 2: Consuming Generated Image ===")
    groupchat_consume = GroupChat(
        agents=[user_proxy, vision_analyst],
        messages=[],
        max_round=2,
        speaker_selection_method="round_robin",
    )

    manager_consume = GroupChatManager(
        groupchat=groupchat_consume,
        llm_config=_create_vision_config(credentials_gemini_flash),
    )

    # Create multimodal message with generated image
    multimodal_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see in this image? Describe the shape and color."},
            {"type": "image_url", "image_url": {"url": image_data_uri}},
        ],
    }

    result_consume = user_proxy.initiate_chat(
        manager_consume,
        message=multimodal_message,
        max_turns=1,
    )

    # Verify vision agent processed the image
    assert result_consume is not None
    assert len(result_consume.chat_history) >= 2

    # Verify analysis mentions expected content (hexagon or blue/cyan colors)
    summary_lower = result_consume.summary.lower()
    assert any(word in summary_lower for word in ["hexagon", "blue", "cyan", "teal", "dot", "geometric"]), (
        f"Vision agent didn't detect hexagon or colors in image: {result_consume.summary}"
    )

    logger.info("✓ Vision agent successfully analyzed generated image")
    logger.info("✓ Full cycle complete: Generate -> Extract -> Consume -> Analyze")


@pytest.mark.gemini
@run_for_optional_imports(["google.genai"], "google-genai")
def test_groupchat_audio_generation_and_metadata_validation(credentials_gemini_flash: Credentials) -> None:
    """Test audio generation in group chat and validate audio metadata.

    This test verifies:
    1. TTS agent generates audio using gemini-2.5-flash-preview-tts
    2. Audio is extracted as data URI from response
    3. Audio data URI format is valid
    4. Audio can be decoded to PCM format
    5. Audio metadata (size, format) is reasonable

    Note: Full audio consumption (speech-to-text) would require additional models/APIs,
    so this test focuses on generation and format validation.
    """
    # Create specialized agents for audio generation
    tts_generator = AssistantAgent(
        name="TTSGenerator",
        llm_config=_create_audio_generation_config(credentials_gemini_flash),
        system_message=(
            "You are a text-to-speech generator. You create short, clear audio clips from text. "
            "Focus on simple, brief messages for testing purposes."
        ),
    )

    coordinator = AssistantAgent(
        name="Coordinator",
        llm_config=_create_vision_config(credentials_gemini_flash),
        system_message="You coordinate tasks and verify outputs.",
    )

    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # Phase 1: Generate audio in group chat
    logger.info("=== Phase 1: Generating Audio ===")
    groupchat = GroupChat(
        agents=[user_proxy, tts_generator, coordinator],
        messages=[],
        max_round=3,
        speaker_selection_method="round_robin",
    )

    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=_create_vision_config(credentials_gemini_flash),
    )

    # Use a short, simple phrase to minimize audio generation time/size
    result = user_proxy.initiate_chat(
        manager,
        message="Generate a short test audio clip saying: 'Testing one two three'",
        max_turns=2,
    )

    # Verify audio was generated
    assert result is not None
    assert len(result.chat_history) >= 2

    # Extract generated audio
    audio_data_uri = _extract_audio_from_response(result)
    assert audio_data_uri is not None, "No audio found in generation response"
    assert audio_data_uri.startswith("data:audio/"), f"Invalid audio data URI: {audio_data_uri[:50]}"
    assert ";base64," in audio_data_uri, "Audio should be base64 encoded"

    logger.info("✓ Audio generated successfully: %s...", audio_data_uri[:80])

    # Phase 2: Validate audio format and metadata
    logger.info("=== Phase 2: Validating Audio Metadata ===")
    header, encoded = audio_data_uri.split(",", 1)

    # Verify audio MIME type
    assert "audio/" in header, f"Invalid audio MIME type: {header}"
    logger.info("✓ Audio MIME type: %s", header)

    # Decode and validate audio data
    audio_bytes = base64.b64decode(encoded)
    assert len(audio_bytes) > 1000, "Generated audio is suspiciously small"

    # PCM audio at 24kHz is ~48KB per second
    # A sentence like "Hello, this is a test..." should be ~3-4 seconds = ~150KB
    logger.info("✓ Audio size: %d bytes (~%.1f seconds at 24kHz PCM)", len(audio_bytes), len(audio_bytes) / 48000)

    # Phase 3: Verify audio data URI can be used in multimodal message format
    logger.info("=== Phase 3: Format Validation ===")
    # Create a multimodal message structure with audio (even if not consumed)
    multimodal_audio_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Audio validation test"},
            {"type": "audio_url", "audio_url": {"url": audio_data_uri}},
        ],
    }

    # Verify structure is valid
    assert isinstance(multimodal_audio_message["content"], list)
    assert len(multimodal_audio_message["content"]) == 2
    assert multimodal_audio_message["content"][1]["type"] == "audio_url"

    logger.info("✓ Audio data URI is properly formatted for multimodal messages")
    logger.info("✓ Audio generation cycle complete: Generate -> Extract -> Validate")


@pytest.mark.gemini
@run_for_optional_imports(["google.genai"], "google-genai")
def test_groupchat_mixed_multimodal_generation_workflow(credentials_gemini_flash: Credentials) -> None:
    """Test complete multimodal workflow with image, vision analysis, and audio generation.

    This test verifies a multimodal group chat workflow where:
    1. ImageCreator generates an image
    2. VisionAnalyst analyzes and describes the generated image
    3. TTSSpeaker creates audio narration from the description
    4. All three agents participate in a single RoundRobinPattern conversation

    The test demonstrates that TTS models can now be used in group chats by automatically
    extracting only the last user message for stateless TTS conversion.
    """
    from autogen.agentchat import initiate_group_chat
    from autogen.agentchat.group.patterns import RoundRobinPattern

    # Create specialized agents for image generation and analysis
    image_generator = AssistantAgent(
        name="ImageCreator",
        llm_config=_create_image_generation_config(credentials_gemini_flash),
        description="Creates images based on text descriptions.",
        system_message="You create images based on descriptions provided.",
    )

    vision_analyst = AssistantAgent(
        name="VisionAnalyst",
        llm_config=_create_vision_config(credentials_gemini_flash),
        description="Analyzes and describes images in detail.",
        system_message="You analyze images. When you see an image, describe it clearly in 1-2 sentences.",
    )

    tts_speaker = AssistantAgent(
        name="TTSSpeaker",
        llm_config=_create_audio_generation_config(credentials_gemini_flash),
        description="Creates audio narration from text descriptions.",
        system_message="You generate audio narration from text descriptions provided to you.",
    )

    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # Create RoundRobinPattern for sequential execution: ImageCreator -> VisionAnalyst -> TTSSpeaker
    logger.info("=== Starting Multimodal Group Chat with RoundRobinPattern ===")
    pattern = RoundRobinPattern(
        initial_agent=image_generator,  # Start with image generation
        agents=[image_generator, vision_analyst, tts_speaker],
        user_agent=user_proxy,
    )

    # User initiates workflow: create image, analyze it, then create audio
    result, _, last_agent = initiate_group_chat(
        pattern=pattern,
        messages="Create an abstract geometric pattern with colorful shapes.",
        max_rounds=4,  # User + ImageCreator + VisionAnalyst + TTSSpeaker
    )

    # Verify the workflow completed
    assert result is not None
    assert result.chat_history is not None
    assert len(result.chat_history) >= 3  # At least User + ImageCreator + VisionAnalyst

    # Extract image from chat history
    image_data_uri = _extract_image_from_response(result)

    # Verify image was generated
    assert image_data_uri is not None, "ImageCreator should have generated an image"
    assert image_data_uri.startswith("data:image/"), "Image should be a valid data URI"

    # Verify image can be decoded
    _, image_encoded = image_data_uri.split(",", 1)
    image_bytes = base64.b64decode(image_encoded)
    assert len(image_bytes) > 100, "Image should contain actual data"

    # Extract audio from chat history
    audio_data_uri = _extract_audio_from_response(result)

    # Verify audio was generated
    assert audio_data_uri is not None, "TTSSpeaker should have generated audio"
    assert audio_data_uri.startswith("data:audio/"), "Audio should be a valid data URI"

    # Verify audio can be decoded
    _, audio_encoded = audio_data_uri.split(",", 1)
    audio_bytes = base64.b64decode(audio_encoded)
    assert len(audio_bytes) > 100, "Audio should contain actual data"

    # Verify all three agents participated
    participant_names = {msg.get("name") for msg in result.chat_history if msg.get("name")}
    assert "ImageCreator" in participant_names, "ImageCreator should have participated"
    assert "VisionAnalyst" in participant_names, "VisionAnalyst should have participated and analyzed the image"
    assert "TTSSpeaker" in participant_names, "TTSSpeaker should have participated and created audio"

    # Verify VisionAnalyst provided image description
    vision_messages = [msg for msg in result.chat_history if msg.get("name") == "VisionAnalyst"]
    assert len(vision_messages) > 0, "VisionAnalyst should have provided analysis"
    vision_content = vision_messages[0].get("content", "")
    assert isinstance(vision_content, str) and len(vision_content) > 10, "VisionAnalyst should have described the image"

    logger.info("✓ Complete multimodal group chat workflow successful:")
    logger.info("  - Agents participated: %s", participant_names)
    logger.info("  - Image: %d bytes", len(image_bytes))
    logger.info("  - Audio: %d bytes", len(audio_bytes))
    logger.info("  - Image description: %s", vision_content[:100])
    logger.info("  - Total conversation turns: %d", len(result.chat_history))
    logger.info("✓ All three modalities working together: Image → Vision → Audio")
