# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Capture real OpenAI API responses for unit test fixtures.

This test module makes actual API calls to OpenAI and captures the raw responses
to create realistic test fixtures for unit testing without needing API keys.

Run with:
    bash scripts/test-core-llm.sh test/llm_clients/test_openai_v2_response_capture.py
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

from autogen.import_utils import run_for_optional_imports
from test.credentials import Credentials

try:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion
except ImportError:
    OpenAI = None
    ChatCompletion = None

logger = logging.getLogger(__name__)


def _get_api_key_from_credentials(credentials: Credentials) -> str:
    """Extract API key from credentials or environment."""
    api_key = credentials.api_key if hasattr(credentials, "api_key") else os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OpenAI API key available")
    return api_key


def _serialize_openai_response(response: Any) -> dict[str, Any]:
    """
    Serialize OpenAI response to JSON-compatible dict.

    Args:
        response: OpenAI ChatCompletion response object

    Returns:
        Dictionary representation that can be saved to JSON
    """
    if hasattr(response, "model_dump"):
        # Pydantic v2 models
        return response.model_dump()
    elif hasattr(response, "dict"):
        # Pydantic v1 models
        return response.dict()
    else:
        # Fallback: convert to dict manually
        return {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                        "refusal": getattr(choice.message, "refusal", None),
                    },
                    "finish_reason": choice.finish_reason,
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "system_fingerprint": getattr(response, "system_fingerprint", None),
        }


def _save_response_fixture(response: Any, fixture_name: str, output_dir: Path) -> None:
    """
    Save OpenAI response as a JSON fixture file.

    Args:
        response: OpenAI response object
        fixture_name: Name for the fixture file (without extension)
        output_dir: Directory to save fixture files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = output_dir / f"{fixture_name}.json"

    response_dict = _serialize_openai_response(response)

    with open(fixture_path, "w") as f:
        json.dump(response_dict, f, indent=2)

    logger.info(f"✓ Saved fixture: {fixture_path}")
    logger.info(f"  - Model: {response_dict['model']}")
    logger.info(f"  - Tokens: {response_dict['usage']['total_tokens']}")


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_capture_simple_text_response(credentials_gpt_4o_mini: Credentials) -> None:
    """Capture a simple text-only response for unit testing."""
    if OpenAI is None:
        pytest.skip("OpenAI not installed")

    api_key = _get_api_key_from_credentials(credentials_gpt_4o_mini)
    client = OpenAI(api_key=api_key)

    # Make simple API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2 + 2? Answer with just the number."}],
        temperature=0,
    )

    # Save fixture
    output_dir = Path(__file__).parent / "fixtures" / "openai_responses"
    _save_response_fixture(response, "simple_text_response", output_dir)

    # Verify response structure
    assert response.choices[0].message.content is not None
    assert "4" in response.choices[0].message.content


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_capture_multimodal_vision_response(credentials_gpt_4o_mini: Credentials) -> None:
    """Capture a multimodal vision response with image for unit testing."""
    if OpenAI is None:
        pytest.skip("OpenAI not installed")

    api_key = _get_api_key_from_credentials(credentials_gpt_4o_mini)
    client = OpenAI(api_key=api_key)

    # Make vision API call with image
    image_url = "https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/test/test_files/test_image.png"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this image? Answer in one word."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        temperature=0,
    )

    # Save fixture
    output_dir = Path(__file__).parent / "fixtures" / "openai_responses"
    _save_response_fixture(response, "multimodal_vision_response", output_dir)

    # Verify response
    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content) > 0


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_capture_tool_call_response(credentials_gpt_4o_mini: Credentials) -> None:
    """Capture a response with tool calls for unit testing."""
    if OpenAI is None:
        pytest.skip("OpenAI not installed")

    api_key = _get_api_key_from_credentials(credentials_gpt_4o_mini)
    client = OpenAI(api_key=api_key)

    # Define a simple tool
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    # Make API call that triggers tool call
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "What's the weather in San Francisco?"}], tools=tools
    )

    # Save fixture
    output_dir = Path(__file__).parent / "fixtures" / "openai_responses"
    _save_response_fixture(response, "tool_call_response", output_dir)

    # Verify tool call structure
    assert response.choices[0].message.tool_calls is not None
    assert len(response.choices[0].message.tool_calls) > 0


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_capture_multi_turn_context_response(credentials_gpt_4o_mini: Credentials) -> None:
    """Capture a multi-turn conversation response for unit testing."""
    if OpenAI is None:
        pytest.skip("OpenAI not installed")

    api_key = _get_api_key_from_credentials(credentials_gpt_4o_mini)
    client = OpenAI(api_key=api_key)

    # Multi-turn conversation
    messages = [
        {"role": "user", "content": "My favorite color is blue."},
        {"role": "assistant", "content": "I'll remember that your favorite color is blue."},
        {"role": "user", "content": "What is my favorite color?"},
    ]

    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0)

    # Save fixture
    output_dir = Path(__file__).parent / "fixtures" / "openai_responses"
    _save_response_fixture(response, "multi_turn_context_response", output_dir)

    # Verify context was maintained
    assert response.choices[0].message.content is not None
    assert "blue" in response.choices[0].message.content.lower()


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_capture_system_message_response(credentials_gpt_4o_mini: Credentials) -> None:
    """Capture a response with system message for unit testing."""
    if OpenAI is None:
        pytest.skip("OpenAI not installed")

    api_key = _get_api_key_from_credentials(credentials_gpt_4o_mini)
    client = OpenAI(api_key=api_key)

    # Call with system message
    messages = [
        {"role": "system", "content": "You are a math tutor. Always show your work step by step."},
        {"role": "user", "content": "What is 15 + 27?"},
    ]

    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0)

    # Save fixture
    output_dir = Path(__file__).parent / "fixtures" / "openai_responses"
    _save_response_fixture(response, "system_message_response", output_dir)

    # Verify response has answer
    assert response.choices[0].message.content is not None
    assert "42" in response.choices[0].message.content


@pytest.mark.openai
@run_for_optional_imports("openai", "openai")
def test_capture_multiple_images_response(credentials_gpt_4o_mini: Credentials) -> None:
    """Capture a response with multiple images using Base64 encoding for unit testing."""
    if OpenAI is None:
        pytest.skip("OpenAI not installed")

    api_key = _get_api_key_from_credentials(credentials_gpt_4o_mini)
    client = OpenAI(api_key=api_key)

    # Two simple Base64 encoded images (1x1 pixel red and blue PNGs)
    # Red 1x1 pixel PNG
    base64_image_1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    # Blue 1x1 pixel PNG
    base64_image_2 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAEBgIApD5fRAAAAABJRU5ErkJggg=="

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these two images. What colors do you see?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_1}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_2}"}},
                ],
            }
        ],
        temperature=0,
    )

    # Save fixture
    output_dir = Path(__file__).parent / "fixtures" / "openai_responses"
    _save_response_fixture(response, "multiple_images_response", output_dir)

    # Verify response
    assert response.choices[0].message.content is not None


def test_fixture_summary() -> None:
    """Print summary of all captured fixtures."""
    fixture_dir = Path(__file__).parent / "fixtures" / "openai_responses"

    if not fixture_dir.exists():
        logger.info("No fixtures directory found yet. Run API capture tests first.")
        return

    fixture_files = list(fixture_dir.glob("*.json"))

    logger.info("\n" + "=" * 60)
    logger.info("CAPTURED OPENAI API RESPONSE FIXTURES")
    logger.info("=" * 60)

    for fixture_file in sorted(fixture_files):
        with open(fixture_file) as f:
            data = json.load(f)

        logger.info(f"\n{fixture_file.name}:")
        logger.info(f"  Model: {data['model']}")
        logger.info(f"  Tokens: {data['usage']['total_tokens']}")
        logger.info(f"  Choices: {len(data['choices'])}")

        if data["choices"]:
            choice = data["choices"][0]
            message = choice["message"]

            if message.get("tool_calls"):
                logger.info(f"  Tool Calls: {len(message['tool_calls'])}")
            if message.get("content"):
                content_preview = str(message["content"])[:60]
                logger.info(f"  Content: {content_preview}...")

    logger.info("\n" + "=" * 60)
    logger.info(f"Total fixtures: {len(fixture_files)}")
    logger.info(f"Location: {fixture_dir}")
    logger.info("=" * 60)
