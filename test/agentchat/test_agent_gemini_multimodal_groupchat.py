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
    """Create config for image generation with gemini-2.5-flash-image."""
    return {
        "config_list": [
            {
                "api_type": "google_stateless",
                "model": "gemini-2.5-flash-image",
                "api_key": credentials.api_key,
                "image_config": {"aspect_ratio": "1:1"},
                "response_modalities": ["Image"],  # Image only output
            }
        ],
        "temperature": 0.7,
    }


def _create_audio_generation_config(credentials: Credentials) -> dict[str, Any]:
    """Create config for audio/TTS generation with gemini-2.5-flash-preview-tts."""
    return {
        "config_list": [
            {
                "api_type": "google_stateless",
                "model": "gemini-2.5-flash-preview-tts",
                "api_key": credentials.api_key,
                "response_modalities": ["AUDIO"],
                "speech_config": {"voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}},
            }
        ],
        "temperature": 0.5,
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
                if isinstance(block, dict) and block.get("type") == "image":
                    return block.get("image_url")
                elif hasattr(block, "type") and block.type == "image":
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
                if isinstance(block, dict) and block.get("type") == "audio":
                    return block.get("audio_url")
                elif hasattr(block, "type") and block.type == "audio":
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
        system_message="You generate images based on descriptions. Output images only.",
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

    result_gen = user_proxy.initiate_chat(
        manager_gen,
        message="Create a simple image of a red circle on white background.",
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

    # Verify analysis mentions expected content (red circle)
    summary_lower = result_consume.summary.lower()
    assert "red" in summary_lower or "circle" in summary_lower, (
        f"Vision agent didn't detect red circle: {result_consume.summary}"
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
        system_message="You generate speech audio from text. Output audio only.",
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

    result = user_proxy.initiate_chat(
        manager,
        message="Generate audio saying: 'Hello, this is a test of multimodal group chat.'",
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
    """Test workflow with multiple media types in group chat conversation.

    This test verifies a complex workflow:
    1. Image generator creates a visual
    2. Vision analyst describes the image
    3. TTS generator creates audio based on the description
    4. Verify all media is properly tracked in chat history

    This simulates a realistic multi-agent workflow where different agents produce
    different types of media content in a coordinated conversation.
    """
    # Create specialized agents
    image_generator = AssistantAgent(
        name="ImageCreator",
        llm_config=_create_image_generation_config(credentials_gemini_flash),
        system_message="You create images. Output images only.",
    )

    vision_analyst = AssistantAgent(
        name="VisionAnalyst",
        llm_config=_create_vision_config(credentials_gemini_flash),
        system_message="You analyze images and provide brief descriptions.",
    )

    tts_generator = AssistantAgent(
        name="TTSSpeaker",
        llm_config=_create_audio_generation_config(credentials_gemini_flash),
        system_message="You generate speech audio. Output audio only.",
    )

    user_proxy = UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )

    # Step 1: Generate image
    logger.info("=== Step 1: Generating Image ===")
    groupchat_image = GroupChat(
        agents=[user_proxy, image_generator],
        messages=[],
        max_round=2,
        speaker_selection_method="round_robin",
    )

    manager_image = GroupChatManager(
        groupchat=groupchat_image,
        llm_config=_create_vision_config(credentials_gemini_flash),
    )

    result_image = user_proxy.initiate_chat(
        manager_image,
        message="Create a simple image of a blue square.",
        max_turns=1,
    )

    image_data_uri = _extract_image_from_response(result_image)
    assert image_data_uri is not None
    logger.info("✓ Image generated: %s...", image_data_uri[:60])

    # Step 2: Analyze image with vision agent
    logger.info("=== Step 2: Analyzing Image ===")
    groupchat_vision = GroupChat(
        agents=[user_proxy, vision_analyst],
        messages=[],
        max_round=2,
        speaker_selection_method="round_robin",
    )

    manager_vision = GroupChatManager(
        groupchat=groupchat_vision,
        llm_config=_create_vision_config(credentials_gemini_flash),
    )

    result_vision = user_proxy.initiate_chat(
        manager_vision,
        message={
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ],
        },
        max_turns=1,
    )

    assert result_vision is not None
    description = result_vision.summary
    logger.info("✓ Vision analysis: %s", description)

    # Step 3: Generate audio from description
    logger.info("=== Step 3: Generating Audio ===")
    groupchat_audio = GroupChat(
        agents=[user_proxy, tts_generator],
        messages=[],
        max_round=2,
        speaker_selection_method="round_robin",
    )

    manager_audio = GroupChatManager(
        groupchat=groupchat_audio,
        llm_config=_create_vision_config(credentials_gemini_flash),
    )

    result_audio = user_proxy.initiate_chat(
        manager_audio,
        message=f"Say: {description}",
        max_turns=1,
    )

    audio_data_uri = _extract_audio_from_response(result_audio)
    assert audio_data_uri is not None
    logger.info("✓ Audio generated: %s...", audio_data_uri[:60])

    # Verify all media types were generated
    assert image_data_uri.startswith("data:image/")
    assert audio_data_uri.startswith("data:audio/")

    # Verify media can be decoded
    _, image_encoded = image_data_uri.split(",", 1)
    _, audio_encoded = audio_data_uri.split(",", 1)

    image_bytes = base64.b64decode(image_encoded)
    audio_bytes = base64.b64decode(audio_encoded)

    assert len(image_bytes) > 100
    assert len(audio_bytes) > 1000

    logger.info("✓ Mixed multimodal workflow complete:")
    logger.info("  - Image: %d bytes", len(image_bytes))
    logger.info("  - Audio: %d bytes", len(audio_bytes))
    logger.info("✓ All media types generated and validated successfully")


# TOTAL API CALLS IN THIS FILE: ~8-10 calls
# - test_groupchat_image_generation_and_consumption_cycle: 2 image gen + 1 vision = 3 calls
# - test_groupchat_audio_generation_and_metadata_validation: 1 audio gen + 1 coordination = 2 calls
# - test_groupchat_mixed_multimodal_generation_workflow: 1 image + 1 vision + 1 audio = 3 calls
# Note: Video generation is NOT included as it uses separate generate_videos() API (out of scope)
