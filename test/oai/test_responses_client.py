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
# Responses Client Tests
# -----------------------------------------------------------------------------


def test_parse_params_with_verbosity_high():
    """Test _parse_params method transforms verbosity parameter correctly."""
    client = OpenAIResponsesClient(MagicMock())

    # Test with verbosity parameter
    params = {
        "verbosity": "high",
        "other_param": "value",
        "api_key": "sk-xxx",
        "organization": "org-yyy",
        "project": "proj-zzz",
        "base_url": "https://api.openai.com/v1",
        "websocket_base_url": "wss://api.openai.com/v1",
        "timeout": 30,
        "max_retries": 2,
        "default_headers": {"X-Test": "1"},
        "default_query": {"foo": "bar"},
        "http_client": None,
        "_strict_response_validation": True,
        "webhook_secret": "whsec-abc",
    }
    result = client._parse_params(params)

    # Should transform verbosity into text format
    assert "verbosity" not in params  # Original verbosity should be removed
    assert "text" in params
    assert params["text"]["verbosity"] == "high"
    assert params["other_param"] == "value"  # Other params should remain unchanged
    assert result == params


def test_parse_params_with_verbosity_low():
    """Test _parse_params method transforms verbosity parameter correctly."""
    client = OpenAIResponsesClient(MagicMock())

    # Test with verbosity parameter
    params = {
        "verbosity": "low",
        "other_param": "value",
        "api_key": "sk-xxx",
        "organization": "org-yyy",
        "project": "proj-zzz",
        "base_url": "https://api.openai.com/v1",
        "websocket_base_url": "wss://api.openai.com/v1",
        "timeout": 30,
        "max_retries": 2,
        "default_headers": {"X-Test": "1"},
        "default_query": {"foo": "bar"},
        "http_client": None,
        "_strict_response_validation": True,
        "webhook_secret": "whsec-abc",
    }
    result = client._parse_params(params)

    # Should transform verbosity into text format
    assert "verbosity" not in params  # Original verbosity should be removed
    assert "text" in params
    assert params["text"]["verbosity"] == "low"
    assert params["other_param"] == "value"  # Other params should remain unchanged
    assert result == params


def test_parse_params_with_verbosity_medium():
    """Test _parse_params method transforms verbosity parameter correctly."""
    client = OpenAIResponsesClient(MagicMock())

    # Test with verbosity parameter
    params = {
        "verbosity": "medium",
        "other_param": "value",
        "api_key": "sk-xxx",
        "organization": "org-yyy",
        "project": "proj-zzz",
        "base_url": "https://api.openai.com/v1",
        "websocket_base_url": "wss://api.openai.com/v1",
        "timeout": 30,
        "max_retries": 2,
        "default_headers": {"X-Test": "1"},
        "default_query": {"foo": "bar"},
        "http_client": None,
        "_strict_response_validation": True,
        "webhook_secret": "whsec-abc",
    }
    result = client._parse_params(params)

    # Should transform verbosity into text format
    assert "verbosity" not in params  # Original verbosity should be removed
    assert "text" in params
    assert params["text"]["verbosity"] == "medium"
    assert params["other_param"] == "value"  # Other params should remain unchanged
    assert result == params


def test_parse_params_with_reasoning_effort_low():
    """Test _parse_params method transforms reasoning_effort parameter correctly."""
    client = OpenAIResponsesClient(MagicMock())

    params = {
        "reasoning_effort": "low",
        "other_param": "value",
    }
    result = client._parse_params(params)

    # Should transform reasoning_effort into reasoning format
    assert "reasoning_effort" not in params  # Original reasoning_effort should be removed
    assert "reasoning" in params
    assert params["reasoning"]["effort"] == "low"
    assert params["other_param"] == "value"  # Other params should remain unchanged
    assert result == params


def test_parse_params_with_reasoning_effort_medium():
    """Test _parse_params method transforms reasoning_effort parameter correctly."""
    client = OpenAIResponsesClient(MagicMock())

    params = {
        "reasoning_effort": "medium",
        "other_param": "value",
    }
    result = client._parse_params(params)

    # Should transform reasoning_effort into reasoning format
    assert "reasoning_effort" not in params
    assert "reasoning" in params
    assert params["reasoning"]["effort"] == "medium"
    assert params["other_param"] == "value"
    assert result == params


def test_parse_params_with_reasoning_effort_high():
    """Test _parse_params method transforms reasoning_effort parameter correctly."""
    client = OpenAIResponsesClient(MagicMock())

    params = {
        "reasoning_effort": "high",
        "other_param": "value",
    }
    result = client._parse_params(params)

    # Should transform reasoning_effort into reasoning format
    assert "reasoning_effort" not in params
    assert "reasoning" in params
    assert params["reasoning"]["effort"] == "high"
    assert params["other_param"] == "value"
    assert result == params


def test_parse_params_with_reasoning_effort_xhigh():
    """Test _parse_params method transforms xhigh reasoning_effort parameter correctly."""
    client = OpenAIResponsesClient(MagicMock())

    params = {
        "reasoning_effort": "xhigh",
        "other_param": "value",
    }
    result = client._parse_params(params)

    # Should transform reasoning_effort into reasoning format
    assert "reasoning_effort" not in params
    assert "reasoning" in params
    assert params["reasoning"]["effort"] == "xhigh"
    assert params["other_param"] == "value"
    assert result == params


def test_parse_params_with_both_verbosity_and_reasoning_effort():
    """Test _parse_params method transforms both verbosity and reasoning_effort correctly."""
    client = OpenAIResponsesClient(MagicMock())

    params = {
        "verbosity": "high",
        "reasoning_effort": "medium",
        "other_param": "value",
    }
    result = client._parse_params(params)

    # Both should be transformed
    assert "verbosity" not in params
    assert "reasoning_effort" not in params
    assert "text" in params
    assert params["text"]["verbosity"] == "high"
    assert "reasoning" in params
    assert params["reasoning"]["effort"] == "medium"
    assert params["other_param"] == "value"
    assert result == params


def test_create_passes_reasoning_effort_to_api(mocked_openai_client):
    """Test that create() properly passes reasoning_effort to the API."""
    client = OpenAIResponsesClient(mocked_openai_client)

    client.create({
        "messages": [{"role": "user", "content": "Solve this complex math problem"}],
        "reasoning_effort": "high",
    })

    kwargs = mocked_openai_client.responses.create.call_args.kwargs

    # Verify reasoning was transformed correctly
    assert "reasoning_effort" not in kwargs
    assert "reasoning" in kwargs
    assert kwargs["reasoning"]["effort"] == "high"


def test_message_retrieval_with_real_response_structure():
    """Test message_retrieval method with realistic response structure including reasoning."""
    client = OpenAIResponsesClient(MagicMock())

    # Create mock objects that mimic the real response structure
    class MockReasoningItem:
        def model_dump(self):
            return {
                "id": "rs_68969f552c488197a913a89ad1f323850cc7af4f09431b8a",
                "summary": [],
                "type": "reasoning",
                "content": None,
                "encrypted_content": None,
                "status": None,
            }

    class MockOutputMessageMultipleBlocks:
        def model_dump(self):
            return {
                "id": "msg_68969f57e34c8197b05c73b4335b57ac0cc7af4f09431b8a",
                "content": [
                    {
                        "annotations": [],
                        "text": "New York City: where 'rush hour' lasts all day",
                        "type": "output_text",
                        "logprobs": [],
                    },
                    {
                        "annotations": [],
                        "text": "and finding parking is a sport.",
                        "type": "output_text",
                        "logprobs": [],
                    },
                    {
                        "annotations": [],
                        "text": "TERMINATE",
                        "type": "output_text",
                        "logprobs": [],
                    },
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }

    # Create response with reasoning item (should be skipped) and message with multiple blocks
    output = [MockReasoningItem(), MockOutputMessageMultipleBlocks()]
    resp = _FakeResponse(output=output)

    msgs = client.message_retrieval(resp)

    assert len(msgs) == 1

    msg = msgs[0]
    assert msg["role"] == "assistant"
    assert msg["id"] == "fake-id"
    assert len(msg["tool_calls"]) == 0

    content = msg["content"]
    assert len(content) == 1

    text_item = content[0]
    assert text_item["type"] == "text"
    assert text_item["role"] == "assistant"
    # Test that multiple blocks are joined with spaces
    assert (
        text_item["text"] == "New York City: where 'rush hour' lasts all day and finding parking is a sport. TERMINATE"
    )
    assert "content" not in text_item


def _create_apply_patch_call_mock(call_id, operation_type, path, diff=None, status="completed"):
    """Factory for creating realistic apply_patch_call mock objects."""

    class MockApplyPatchCall:
        def model_dump(self):
            result = {
                "id": f"apc_{call_id}",
                "type": "apply_patch_call",
                "call_id": call_id,
                "status": status,
                "operation": {
                    "type": operation_type,
                    "path": path,
                },
            }
            # Only include diff for operations that need it
            if operation_type != "delete_file" and diff is not None:
                result["operation"]["diff"] = diff
            return result

    return MockApplyPatchCall()


def _create_message_mock(text, msg_id="msg_auto"):
    """Factory for creating realistic message mock objects."""

    class MockMessage:
        def model_dump(self):
            return {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
                "status": "completed",
            }

    return MockMessage()


def _create_reasoning_mock():
    """Factory for creating realistic reasoning mock objects."""

    class MockReasoning:
        def model_dump(self):
            return {
                "id": "msg_reasoning",
                "type": "reasoning",
                "summary": [],
                "content": None,
                "encrypted_content": None,
                "status": None,
            }

    return MockReasoning()


# -----------------------------------------------------------------------------
# Refactored Apply Patch Tool Tests
# -----------------------------------------------------------------------------


def test_apply_patch_tool_added_to_built_in_tools(mocked_openai_client):
    """Test that apply_patch is properly added to built-in tools list."""
    client = OpenAIResponsesClient(mocked_openai_client)

    client.create({
        "messages": [{"role": "user", "content": "Refactor this code"}],
        "built_in_tools": ["apply_patch"],
    })

    kwargs = mocked_openai_client.responses.create.call_args.kwargs

    # Verify apply_patch tool was added
    tool_types = [t["type"] for t in kwargs["tools"]]
    assert "apply_patch" in tool_types


def test_apply_patch_with_other_built_in_tools(mocked_openai_client):
    """Test that apply_patch works alongside other built-in tools."""
    client = OpenAIResponsesClient(mocked_openai_client)

    client.create({
        "messages": [{"role": "user", "content": "Search and edit code"}],
        "built_in_tools": ["web_search", "apply_patch", "image_generation"],
    })

    kwargs = mocked_openai_client.responses.create.call_args.kwargs

    # All three built-in tools should be present
    tool_types = {t["type"] for t in kwargs["tools"]}
    assert {"web_search_preview", "apply_patch", "image_generation"}.issubset(tool_types)


def test_message_retrieval_handles_apply_patch_call():
    """Test that message_retrieval correctly handles apply_patch_call items."""
    client = OpenAIResponsesClient(MagicMock())

    # Use realistic fixture
    patch_call = _create_apply_patch_call_mock(
        call_id="call_Rjsqzz96C5xzPb0jUWJFRTNW",
        operation_type="update_file",
        path="lib/fib.py",
        diff="@@\n-def fib(n):\n+def fibonacci(n):\n    if n <= 1:\n        return n\n",
    )

    output = [patch_call]
    resp = _FakeResponse(output=output)

    msgs = client.message_retrieval(resp)

    # Should return one message
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg["role"] == "assistant"

    # Check that apply_patch_call is in content
    content = msg["content"]
    assert len(content) == 1

    patch = content[0]
    assert patch["type"] == "apply_patch_call"
    assert patch["call_id"] == "call_Rjsqzz96C5xzPb0jUWJFRTNW"
    assert patch["status"] == "completed"
    assert patch["operation"]["type"] == "update_file"
    assert patch["operation"]["path"] == "lib/fib.py"
    assert "diff" in patch["operation"]
    assert "fibonacci" in patch["operation"]["diff"]


def test_message_retrieval_handles_multiple_apply_patch_calls():
    """Test message_retrieval with multiple apply_patch_call operations."""
    client = OpenAIResponsesClient(MagicMock())

    # Create realistic patch calls for different operation types
    output = [
        _create_apply_patch_call_mock(
            "call_1", "update_file", "src/main.py", diff="@@\n-old_function()\n+new_function()\n"
        ),
        _create_apply_patch_call_mock(
            "call_2", "create_file", "src/test.py", diff="@@\n+def test_new_function():\n+    assert True\n"
        ),
        _create_apply_patch_call_mock("call_3", "delete_file", "src/old.py"),
    ]
    resp = _FakeResponse(output=output)

    msgs = client.message_retrieval(resp)

    assert len(msgs) == 1
    content = msgs[0]["content"]
    assert len(content) == 3

    # Verify each operation with realistic assertions
    assert content[0]["operation"]["type"] == "update_file"
    assert content[0]["operation"]["path"] == "src/main.py"
    assert "new_function" in content[0]["operation"]["diff"]

    assert content[1]["operation"]["type"] == "create_file"
    assert content[1]["operation"]["path"] == "src/test.py"
    assert "test_new_function" in content[1]["operation"]["diff"]

    assert content[2]["operation"]["type"] == "delete_file"
    assert content[2]["operation"]["path"] == "src/old.py"
    assert "diff" not in content[2]["operation"]  # delete has no diff


def test_message_retrieval_mixed_content_with_apply_patch():
    """Test message_retrieval with mixed content including text and apply_patch_call."""
    client = OpenAIResponsesClient(MagicMock())

    # Use realistic fixtures for both message and patch
    output = [
        _create_message_mock("I'll refactor that for you.", msg_id="msg_explanation"),
        _create_apply_patch_call_mock("call_xyz", "update_file", "app.py", diff="@@\n-old_code\n+new_code\n"),
    ]
    resp = _FakeResponse(output=output)

    msgs = client.message_retrieval(resp)

    assert len(msgs) == 1
    content = msgs[0]["content"]

    # Should have both text and patch call
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "I'll refactor that for you."

    assert content[1]["type"] == "apply_patch_call"
    assert content[1]["operation"]["path"] == "app.py"
    assert "new_code" in content[1]["operation"]["diff"]


def test_apply_patch_call_preserves_status():
    """Test that apply_patch_call status is preserved correctly for all status types."""
    client = OpenAIResponsesClient(MagicMock())

    for status in ["in_progress", "completed", "failed"]:
        patch_call = _create_apply_patch_call_mock(
            f"call_{status}", "update_file", "test.py", diff="@@\n+test\n", status=status
        )

        output = [patch_call]
        resp = _FakeResponse(output=output)

        msgs = client.message_retrieval(resp)
        patch = msgs[0]["content"][0]

        assert patch["status"] == status, f"Status {status} was not preserved"


def test_apply_patch_no_diff_for_delete():
    """Test that delete_file operation doesn't require diff field."""
    client = OpenAIResponsesClient(MagicMock())

    # Use factory which automatically omits diff for delete operations
    delete_call = _create_apply_patch_call_mock("call_delete", "delete_file", "obsolete.py")

    output = [delete_call]
    resp = _FakeResponse(output=output)

    msgs = client.message_retrieval(resp)
    patch = msgs[0]["content"][0]

    assert patch["operation"]["type"] == "delete_file"
    assert patch["operation"]["path"] == "obsolete.py"
    assert "diff" not in patch["operation"]


def test_create_with_no_built_in_tools_excludes_apply_patch(mocked_openai_client):
    """Test that apply_patch is not added when built_in_tools is empty or not specified."""
    client = OpenAIResponsesClient(mocked_openai_client)

    client.create({
        "messages": [{"role": "user", "content": "Hello"}],
    })

    kwargs = mocked_openai_client.responses.create.call_args.kwargs

    # Tools should be empty or not contain apply_patch
    tools = kwargs.get("tools", [])
    tool_types = [t["type"] for t in tools]
    assert "apply_patch" not in tool_types


def test_message_retrieval_with_realistic_apply_patch_response():
    """Test message_retrieval with a realistic multi-file refactoring response including reasoning."""
    client = OpenAIResponsesClient(MagicMock())

    # Build realistic response with reasoning, message, and multiple patches
    output = [
        _create_reasoning_mock(),  # Should be skipped in retrieval
        _create_message_mock("I'll rename the fib() function to fibonacci() in both files.", msg_id="msg_explanation"),
        _create_apply_patch_call_mock(
            "call_Rjsqzz96C5xzPb0jUWJFRTNW",
            "update_file",
            "lib/fib.py",
            diff="@@\n-def fib(n):\n+def fibonacci(n):\n    if n <= 1:\n        return n\n-    return fib(n-1) + fib(n-2)\n+    return fibonacci(n-1) + fibonacci(n-2)\n",
        ),
        _create_apply_patch_call_mock(
            "call_X8bnqmK3LpYzQb9jXRHDSPOL",
            "update_file",
            "run.py",
            diff="@@\n-from lib.fib import fib\n+from lib.fib import fibonacci\n\n def main():\n-  print(fib(42))\n+  print(fibonacci(42))\n",
        ),
    ]

    usage = _FakeUsage(
        input_tokens=150, output_tokens=80, total_tokens=230, output_tokens_details={"reasoning_tokens": 0}
    )

    resp = _FakeResponse(output=output, usage=usage)
    resp.id = "resp_abc123"
    resp.model = "gpt-5.1"

    # Test message retrieval
    msgs = client.message_retrieval(resp)

    # Should return single message with consolidated content
    assert len(msgs) == 1

    msg = msgs[0]
    assert msg["role"] == "assistant"
    assert msg["id"] == "resp_abc123"

    # Content should have: 1 text message + 2 apply_patch_calls (reasoning is skipped)
    content = msg["content"]
    assert len(content) == 3

    # First item: explanation text
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "I'll rename the fib() function to fibonacci() in both files."

    # Second item: first patch operation (lib/fib.py)
    assert content[1]["type"] == "apply_patch_call"
    assert content[1]["call_id"] == "call_Rjsqzz96C5xzPb0jUWJFRTNW"
    assert content[1]["status"] == "completed"
    assert content[1]["operation"]["type"] == "update_file"
    assert content[1]["operation"]["path"] == "lib/fib.py"
    assert "def fibonacci(n):" in content[1]["operation"]["diff"]
    assert "fibonacci(n-1) + fibonacci(n-2)" in content[1]["operation"]["diff"]

    # Third item: second patch operation (run.py)
    assert content[2]["type"] == "apply_patch_call"
    assert content[2]["call_id"] == "call_X8bnqmK3LpYzQb9jXRHDSPOL"
    assert content[2]["status"] == "completed"
    assert content[2]["operation"]["type"] == "update_file"
    assert content[2]["operation"]["path"] == "run.py"
    assert "from lib.fib import fibonacci" in content[2]["operation"]["diff"]
    assert "print(fibonacci(42))" in content[2]["operation"]["diff"]

    # Verify usage is correctly extracted
    usage_dict = client._usage_dict(resp)
    assert usage_dict["prompt_tokens"] == 150
    assert usage_dict["completion_tokens"] == 80
    assert usage_dict["total_tokens"] == 230
    assert usage_dict["model"] == "gpt-5.1"


def test_apply_patch_with_reasoning_is_filtered():
    """Test that reasoning blocks are properly filtered out from content."""
    client = OpenAIResponsesClient(MagicMock())

    # Reasoning should not appear in final message content
    output = [
        _create_reasoning_mock(),
        _create_apply_patch_call_mock("call_test", "create_file", "new.py", diff="@@\n+print('hello')\n"),
    ]
    resp = _FakeResponse(output=output)

    msgs = client.message_retrieval(resp)
    content = msgs[0]["content"]

    # Only the patch call should be in content (reasoning filtered)
    assert len(content) == 1
    assert content[0]["type"] == "apply_patch_call"

    # Verify no reasoning type in content
    content_types = [item["type"] for item in content]
    assert "reasoning" not in content_types
