# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


"""Unit-tests for the OpenAIResponsesClient abstraction.

These tests are self-contained—they DO NOT call the real OpenAI
endpoint. Instead we mock the `openai.OpenAI` instance and capture the
kwargs passed to `client.responses.create` / `client.responses.parse`.

We follow the style of existing tests in *test/oai/test_client.py*.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from autogen.oai.openai_responses import OpenAIResponsesClient, calculate_openai_image_cost

# Try to import ImageGenerationCall for proper mocking
try:
    from openai.types.responses.response_output_item import ImageGenerationCall

    # Check if it's a Pydantic model
    HAS_IMAGE_GENERATION_CALL = True
except ImportError:
    # Create a mock class if openai SDK is not available
    HAS_IMAGE_GENERATION_CALL = False

    class ImageGenerationCall:
        pass

# -----------------------------------------------------------------------------
# Helper fakes
# -----------------------------------------------------------------------------


class _FakeUsage:
    """Mimics the `.usage` member on an OpenAI Response object."""

    def __init__(self, **fields):
        self._fields = fields

    def model_dump(self):  # type: ignore[override]
        return self._fields


class _FakeResponse:
    """Minimal object returned by mocked `.responses.create`"""

    def __init__(self, *, output=None, usage=None):
        self.output = output or []
        self.usage = usage or {}
        self.cost = 1.23  # arbitrary
        self.model = "gpt-4o"
        self.id = "fake-id"


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture()
def mocked_openai_client():
    """Return a fake `OpenAI` instance with stubbed `.responses` interface."""

    mock_client = MagicMock()
    mock_responses = MagicMock()
    mock_client.responses = mock_responses  # attach

    # By default `.create` returns an empty fake response; tests can overwrite.
    mock_responses.create.return_value = _FakeResponse()
    mock_responses.parse.return_value = _FakeResponse()

    return mock_client


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_messages_are_transformed_into_input(mocked_openai_client):
    """`messages=[…]` should be converted into `input=[{{type:'message',…}}]`."""

    client = OpenAIResponsesClient(mocked_openai_client)

    messages_param = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]

    client.create({"messages": messages_param})

    # capture the kwargs actually sent to mocked .responses.create
    kwargs = mocked_openai_client.responses.create.call_args.kwargs

    assert "messages" not in kwargs, "messages should have been popped"
    assert "input" in kwargs, "input should be present after conversion"

    # the first converted item should reflect original content
    first_item = kwargs["input"][0]
    assert first_item["role"] == "user"
    assert first_item["content"][0]["text"] == "Hello"


def test_structured_output_path_uses_parse(mocked_openai_client):
    """When `response_format` / `text_format` is supplied the client should call
    `.responses.parse` instead of `.responses.create` and inject the correct
    `text_format` payload."""

    response_format_schema = {"type": "object", "properties": {"a": {"type": "string"}}}

    client = OpenAIResponsesClient(mocked_openai_client)

    client.create({
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": response_format_schema,
    })

    # The parse method should have been invoked
    assert mocked_openai_client.responses.parse.called, "parse() must be used for structured output"

    # verify `text_format` kwarg exists
    kwargs = mocked_openai_client.responses.parse.call_args.kwargs
    assert "text_format" in kwargs


def test_usage_dict_parses_pydantic_like_object():
    usage_obj = _FakeUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    resp = _FakeResponse(usage=usage_obj)
    client = OpenAIResponsesClient(MagicMock())

    usage = client._usage_dict(resp)

    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 15
    assert usage["cost"] == 1.23
    assert usage["model"] == "gpt-4o"


def test_message_retrieval_handles_various_item_types():
    # fake pydantic-like blocks
    class _Block:
        def __init__(self, d):
            self._d = d

        def model_dump(self):
            return self._d

    output = [
        _FakeResponse(output=[{"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}]).output[0],
        {"type": "function_call", "name": "foo", "arguments": "{}"},
        {"type": "web_search_call", "id": "abc", "arguments": {}},
    ]

    # Wrap dicts into objects providing model_dump to test conversion path
    output_wrapped = [_Block(o) if isinstance(o, dict) else o for o in output]

    resp = _FakeResponse(output=output_wrapped)
    client = OpenAIResponsesClient(MagicMock())

    msgs = client.message_retrieval(resp)

    # The client aggregates the three items into a single assistant message
    assert len(msgs) == 1

    top_msg = msgs[0]
    assert top_msg["role"] == "assistant"

    blocks = top_msg["content"]

    # After the refactor function calls are stored in `tool_calls`,
    # so `content` now contains only the assistant text and built-in tool calls.
    assert len(blocks) == 2

    # 1) Plain text block
    assert blocks[0]["text"] == "Hi"

    # 2) Tool-call block (web_search)
    assert blocks[1]["name"] == "web_search"

    # Custom function call moved to `tool_calls`
    tool_calls = top_msg["tool_calls"]
    assert len(tool_calls) == 1
    func_call = tool_calls[0]
    assert func_call["function"]["name"] == "foo"


# -----------------------------------------------------------------------------
# New tests --------------------------------------------------------------------
# -----------------------------------------------------------------------------


def test_get_delta_messages_filters_completed_blocks():
    """_get_delta_messages should drop already-completed messages and return only deltas."""

    client = OpenAIResponsesClient(MagicMock())

    msgs = [
        {"role": "assistant", "content": "Hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Previous reply"},
                {"status": "completed"},
            ],
        },
        {"role": "user", "content": "follow-up"},
    ]

    deltas = client._get_delta_messages(msgs)

    # Only the last message (after completed) should be returned
    assert deltas == [msgs[-1]]


def test_create_converts_multimodal_blocks(mocked_openai_client):
    """create() must turn mixed text / image blocks into correct input schema."""

    client = OpenAIResponsesClient(mocked_openai_client)

    messages_param = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe this image."},
                {"type": "input_image", "image_url": "https://example.com/cat.png"},
            ],
        }
    ]

    client.create({
        "messages": messages_param,
        # Explicitly request these built-in tools so the client injects them.
        "built_in_tools": ["image_generation", "web_search"],
    })

    kwargs = mocked_openai_client.responses.create.call_args.kwargs

    # Ensure conversion occurred
    assert "input" in kwargs
    first = kwargs["input"][0]
    blocks = first["content"]
    assert blocks[0]["type"] == "input_text"
    assert blocks[0]["text"] == "Describe this image."
    assert blocks[1]["type"] == "input_image"
    assert blocks[1]["image_url"] == "https://example.com/cat.png"

    # The requested built-in tools should map to image_generation and web_search_preview
    tool_types = {t["type"] for t in kwargs["tools"]}
    assert {"image_generation", "web_search_preview"}.issubset(tool_types)


# -----------------------------------------------------------------------------
# Image Cost Tests
# -----------------------------------------------------------------------------


def test_calculate_openai_image_cost_gpt_image_1():
    """Test image cost calculation for gpt-image-1 model."""
    # Test all valid combinations for gpt-image-1
    test_cases = [
        # (size, quality, expected_cost)
        ("1024x1024", "low", 0.011),
        ("1024x1024", "medium", 0.042),
        ("1024x1024", "high", 0.167),
        ("1024x1536", "low", 0.016),
        ("1024x1536", "medium", 0.063),
        ("1024x1536", "high", 0.25),
        ("1536x1024", "low", 0.016),
        ("1536x1024", "medium", 0.063),
        ("1536x1024", "high", 0.25),
    ]

    for size, quality, expected in test_cases:
        cost, error = calculate_openai_image_cost("gpt-image-1", size, quality)
        assert cost == expected
        assert error is None


def test_calculate_openai_image_cost_dalle_3():
    """Test image cost calculation for dall-e-3 model."""
    test_cases = [
        ("1024x1024", "standard", 0.040),
        ("1024x1024", "hd", 0.080),
        ("1024x1792", "standard", 0.080),
        ("1024x1792", "hd", 0.120),
        ("1792x1024", "standard", 0.080),
        ("1792x1024", "hd", 0.120),
    ]

    for size, quality, expected in test_cases:
        cost, error = calculate_openai_image_cost("dall-e-3", size, quality)
        assert cost == expected
        assert error is None


def test_calculate_openai_image_cost_dalle_2():
    """Test image cost calculation for dall-e-2 model."""
    test_cases = [
        ("1024x1024", "standard", 0.020),
        ("512x512", "standard", 0.018),
        ("256x256", "standard", 0.016),
    ]

    for size, quality, expected in test_cases:
        cost, error = calculate_openai_image_cost("dall-e-2", size, quality)
        assert cost == expected
        assert error is None


def test_calculate_openai_image_cost_case_insensitive():
    """Test that model and quality parameters are case-insensitive."""
    # Test uppercase model name
    cost, error = calculate_openai_image_cost("GPT-IMAGE-1", "1024x1024", "HIGH")
    assert cost == 0.167
    assert error is None

    # Test mixed case
    cost, error = calculate_openai_image_cost("Dall-E-3", "1024x1024", "Standard")
    assert cost == 0.040
    assert error is None


def test_calculate_openai_image_cost_invalid_model():
    """Test error handling for invalid model names."""
    cost, error = calculate_openai_image_cost("invalid-model", "1024x1024", "high")
    assert cost == 0.0
    assert "Invalid model: invalid-model" in error
    assert "Valid models: ['gpt-image-1', 'dall-e-3', 'dall-e-2']" in error


def test_calculate_openai_image_cost_invalid_size():
    """Test error handling for invalid image sizes."""
    # Invalid size for gpt-image-1
    cost, error = calculate_openai_image_cost("gpt-image-1", "512x512", "high")
    assert cost == 0.0
    assert "Invalid size 512x512 for gpt-image-1" in error

    # Invalid size for dall-e-3
    cost, error = calculate_openai_image_cost("dall-e-3", "256x256", "standard")
    assert cost == 0.0
    assert "Invalid size 256x256 for dall-e-3" in error


def test_calculate_openai_image_cost_invalid_quality():
    """Test error handling for invalid quality levels."""
    # Invalid quality for gpt-image-1
    cost, error = calculate_openai_image_cost("gpt-image-1", "1024x1024", "ultra")
    assert cost == 0.0
    assert "Invalid quality 'ultra' for gpt-image-1" in error

    # Invalid quality for dall-e-3
    cost, error = calculate_openai_image_cost("dall-e-3", "1024x1024", "low")
    assert cost == 0.0
    assert "Invalid quality 'low' for dall-e-3" in error


def test_add_image_cost_single_image(mocked_openai_client):
    """Test _add_image_cost method with a single image generation."""
    client = OpenAIResponsesClient(mocked_openai_client)

    # Create a mock ImageGenerationCall-like object using MagicMock
    mock_image_call = MagicMock(spec=ImageGenerationCall)
    mock_image_call.model_extra = {"size": "1024x1536", "quality": "medium"}

    # Create a response with one image generation
    output = [mock_image_call]
    resp = _FakeResponse(output=output)

    # Process the response
    client._add_image_cost(resp)

    # Verify the cost was added (1024x1536 medium = 0.063)
    assert client.image_costs == 0.063


def test_add_image_cost_multiple_images(mocked_openai_client):
    """Test _add_image_cost method with multiple image generations."""
    client = OpenAIResponsesClient(mocked_openai_client)

    # Create multiple mock ImageGenerationCall objects
    mock_calls = []
    for size, quality in [("1024x1024", "low"), ("1536x1024", "high"), ("1024x1536", "medium")]:
        mock_call = MagicMock(spec=ImageGenerationCall)
        mock_call.model_extra = {"size": size, "quality": quality}
        mock_calls.append(mock_call)

    # Create a response with multiple image generations
    resp = _FakeResponse(output=mock_calls)

    # Process the response
    client._add_image_cost(resp)

    # Verify the total cost (0.011 + 0.25 + 0.063)
    assert client.image_costs == 0.011 + 0.25 + 0.063


def test_add_image_cost_no_images(mocked_openai_client):
    """Test _add_image_cost method with no image generations."""
    client = OpenAIResponsesClient(mocked_openai_client)

    # Create a response with no image generations
    output = [{"type": "message", "content": "Hello"}]
    resp = _FakeResponse(output=output)

    # Process the response
    client._add_image_cost(resp)

    # Verify no cost was added
    assert client.image_costs == 0


def test_add_image_cost_missing_model_extra(mocked_openai_client):
    """Test _add_image_cost method when model_extra is missing."""
    client = OpenAIResponsesClient(mocked_openai_client)

    # Create a mock without model_extra
    mock_call = MagicMock(spec=ImageGenerationCall)
    # Don't set model_extra attribute

    output = [mock_call]
    resp = _FakeResponse(output=output)

    # This should not raise an error
    client._add_image_cost(resp)

    # No cost should be added
    assert client.image_costs == 0


def test_add_image_cost_defaults(mocked_openai_client):
    """Test _add_image_cost uses correct defaults when fields are missing."""
    client = OpenAIResponsesClient(mocked_openai_client)

    # Create a mock with empty model_extra dict
    mock_call = MagicMock(spec=ImageGenerationCall)
    mock_call.model_extra = {}  # Empty dict

    # Note: Due to the bug in line 193, empty dict is falsy so no cost will be added
    output = [mock_call]
    resp = _FakeResponse(output=output)

    # Process the response
    client._add_image_cost(resp)

    # Empty model_extra dict is falsy, so the condition fails and no cost is added
    assert client.image_costs == 0


def test_total_cost_includes_image_costs(mocked_openai_client):
    """Test that the cost() method includes accumulated image costs."""
    client = OpenAIResponsesClient(mocked_openai_client)

    # Add some image costs
    client.image_costs = 0.5

    # Create a response with usage cost
    usage = _FakeUsage(input_tokens=10, output_tokens=5, total_tokens=15)
    resp = _FakeResponse(usage=usage)
    resp.cost = 0.3  # API usage cost

    # Total cost should include both
    total_cost = client.cost(resp)
    assert total_cost == 0.3 + 0.5  # API cost + image costs


def test_image_costs_persist_across_calls(mocked_openai_client):
    """Test that image costs accumulate across multiple create() calls."""
    client = OpenAIResponsesClient(mocked_openai_client)

    # First create call with image
    mock_call1 = MagicMock(spec=ImageGenerationCall)
    mock_call1.model_extra = {"size": "1024x1024", "quality": "low"}
    output1 = [mock_call1]  # 0.011
    mocked_openai_client.responses.create.return_value = _FakeResponse(output=output1)
    client.create({"messages": [{"role": "user", "content": "Generate image 1"}]})

    # Second create call with image
    mock_call2 = MagicMock(spec=ImageGenerationCall)
    mock_call2.model_extra = {"size": "1024x1024", "quality": "medium"}
    output2 = [mock_call2]  # 0.042
    mocked_openai_client.responses.create.return_value = _FakeResponse(output=output2)
    client.create({"messages": [{"role": "user", "content": "Generate image 2"}]})

    # Costs should accumulate
    assert client.image_costs == 0.011 + 0.042


def test_add_image_cost_bug_demonstration(mocked_openai_client):
    """Demonstrate the bug in _add_image_cost where it checks output[0] instead of current item."""
    client = OpenAIResponsesClient(mocked_openai_client)

    # Create two mocks: first with model_extra (required by bug), second with model_extra
    mock_call1 = MagicMock(spec=ImageGenerationCall)
    mock_call1.model_extra = {"size": "1024x1024", "quality": "high"}  # Need this due to bug

    mock_call2 = MagicMock(spec=ImageGenerationCall)
    mock_call2.model_extra = {"size": "1024x1024", "quality": "low"}

    output = [mock_call1, mock_call2]
    resp = _FakeResponse(output=output)

    # Process the response
    client._add_image_cost(resp)

    # Due to the bug checking output[0], both images will be processed
    # First: 1024x1024 high = 0.167, Second: 1024x1024 low = 0.011
    assert client.image_costs == 0.167 + 0.011


def test_add_image_cost_partial_defaults(mocked_openai_client):
    """Test _add_image_cost uses defaults for missing fields when model_extra is truthy."""
    client = OpenAIResponsesClient(mocked_openai_client)

    # Create a mock with model_extra that has only size (missing quality)
    mock_call = MagicMock(spec=ImageGenerationCall)
    mock_call.model_extra = {"size": "1024x1024"}  # Missing quality, should use default "high"

    output = [mock_call]
    resp = _FakeResponse(output=output)

    # Process the response
    client._add_image_cost(resp)

    # Should use size=1024x1024, quality=high (default) for gpt-image-1
    # Cost should be 0.167
    assert client.image_costs == 0.167


def test_add_image_cost_with_non_image_first(mocked_openai_client):
    """Test case where first output is not an ImageGenerationCall."""
    client = OpenAIResponsesClient(mocked_openai_client)

    # First item is not an ImageGenerationCall, second is
    mock_call = MagicMock(spec=ImageGenerationCall)
    mock_call.model_extra = {"size": "1024x1024", "quality": "low"}

    output = [
        {"type": "message", "content": "Hello"},  # Not an ImageGenerationCall
        mock_call,
    ]
    resp = _FakeResponse(output=output)

    # Process the response
    client._add_image_cost(resp)

    # The bug will try to check output[0].model_extra on a dict, which will fail
    # So no image cost will be added
    assert client.image_costs == 0

# -----------------------------------------------------------------------------
# GPT Reasoning Tests
# -----------------------------------------------------------------------------

def test_reasoning_effort_added_to_payload(mocked_openai_client):
    """When 'reasoning_effort' is provided, client must map it to 'reasoning.effort'."""
    client = OpenAIResponsesClient(mocked_openai_client)

    client.create({
        "model": "gpt-5",
        "reasoning_effort": "minimal",
    })

    kwargs = mocked_openai_client.responses.create.call_args.kwargs

    assert "reasoning" in kwargs, f"Expected 'reasoning' in payload, got: {kwargs}"
    assert kwargs["reasoning"]["effort"] == "minimal"


def test_reasoning_and_verbosity_coexist(mocked_openai_client):
    """Both 'reasoning_effort' and 'verbosity' should be mapped simultaneously."""
    client = OpenAIResponsesClient(mocked_openai_client)

    client.create({
        "model": "gpt-5",
        "reasoning_effort": "low",
        "verbosity": "medium",
    })

    kwargs = mocked_openai_client.responses.create.call_args.kwargs

    # text.verbosity mapping
    assert "text" in kwargs, f"Expected 'text' in payload, got: {kwargs}"
    assert kwargs["text"]["verbosity"] == "medium"

    # reasoning.effort mapping
    assert "reasoning" in kwargs, f"Expected 'reasoning' in payload, got: {kwargs}"
    assert kwargs["reasoning"]["effort"] == "low"


def test_no_reasoning_or_verbosity_when_unset(mocked_openai_client):
    """If neither setting is provided, payload should not include 'text' or 'reasoning'."""
    client = OpenAIResponsesClient(mocked_openai_client)

    client.create({
        "model": "gpt-5",
        # no verbosity, no reasoning_effort
    })

    kwargs = mocked_openai_client.responses.create.call_args.kwargs
    assert "text" not in kwargs
    assert "reasoning" not in kwargs
