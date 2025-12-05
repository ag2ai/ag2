# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Fixture-based unit tests for GeminiStatelessClient using real API response structures.

These tests use real API response fixtures captured from integration tests,
providing more realistic testing without requiring API calls.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from autogen.llm_clients import GeminiStatelessClient
from autogen.llm_clients.models import UnifiedResponse

# Load fixtures from integration test captures
FIXTURES_PATH = Path(__file__).parent / "fixtures" / "gemini_sample_responses.json"


@pytest.fixture(scope="module")
def gemini_fixtures():
    """Load Gemini API response fixtures."""
    with open(FIXTURES_PATH) as f:
        return json.load(f)


def create_mock_response_from_fixture(fixture_data: dict) -> Mock:
    """Create a mock Gemini API response from fixture data.

    This reconstructs the Gemini API response structure that would come from
    google.genai.models.generate_content(), based on a serialized UnifiedResponse.
    """
    # Create mock response object
    mock_response = Mock()
    mock_response.model_version = fixture_data["model"]
    mock_response.response_id = fixture_data["id"]  # Set response ID from fixture

    # Create usage metadata
    usage = fixture_data["usage"]
    mock_usage = Mock()
    mock_usage.prompt_token_count = usage["prompt_tokens"]
    mock_usage.candidates_token_count = usage["completion_tokens"]
    mock_usage.total_token_count = usage["total_tokens"]
    mock_usage.thoughts_token_count = usage.get("thinking_tokens", 0)
    mock_response.usage_metadata = mock_usage

    # Create candidates with content
    mock_candidates = []
    for message in fixture_data["messages"]:
        mock_candidate = Mock()
        mock_content = Mock()
        mock_parts = []

        for content_block in message["content"]:
            # Create object-like mock that works properly with hasattr
            class MockPart:
                pass

            mock_part = MockPart()

            if content_block["type"] == "text":
                mock_part.text = content_block["text"]
                mock_part.thought = False
                # Don't set function_call so hasattr returns False
            elif content_block["type"] == "thinking":
                mock_part.text = content_block["thinking"]
                mock_part.thought = True
            elif content_block["type"] == "tool_call":

                class MockFunctionCall:
                    pass

                mock_func_call = MockFunctionCall()
                mock_func_call.name = content_block["name"]
                mock_func_call.args = json.loads(content_block["arguments"])
                mock_part.function_call = mock_func_call
                mock_part.thought = False

            mock_parts.append(mock_part)

        mock_content.parts = mock_parts
        mock_content.role = "model"  # Gemini uses "model" for assistant messages
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"
        mock_candidates.append(mock_candidate)

    mock_response.candidates = mock_candidates

    return mock_response


@pytest.fixture
def mock_genai_client():
    """Create mock Google GenAI client."""
    # Patch at the import location, not the module attribute
    with patch("google.genai.Client") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock the models.generate_content method
        mock_client.models = Mock()
        mock_client.models.generate_content = Mock()

        yield mock_client


class TestGeminiStatelessClientWithFixtures:
    """Test GeminiStatelessClient using real API response fixtures."""

    def test_basic_text_with_real_structure(self, mock_genai_client, gemini_fixtures):
        """Test basic text generation using real API response structure."""
        fixture = gemini_fixtures["basic_text"]

        # Create mock response from fixture
        mock_response = create_mock_response_from_fixture(fixture)
        mock_genai_client.models.generate_content.return_value = mock_response

        # Test client
        client = GeminiStatelessClient(api_key="test-key")
        response = client.create({
            "model": fixture["model"],
            "messages": [{"role": "user", "content": "Say 'Hello, World!' and nothing else."}],
        })

        # Verify response matches fixture structure
        assert isinstance(response, UnifiedResponse)
        assert response.provider == fixture["provider"]
        assert response.model == fixture["model"]

        # Verify usage matches real API response
        assert response.usage["prompt_tokens"] == fixture["usage"]["prompt_tokens"]
        assert response.usage["completion_tokens"] == fixture["usage"]["completion_tokens"]
        assert response.usage["total_tokens"] == fixture["usage"]["total_tokens"]
        assert response.usage["thinking_tokens"] == fixture["usage"]["thinking_tokens"]

        # Verify cost calculation matches
        assert abs(response.cost - fixture["cost"]) < 1e-9  # Allow for floating point precision

        # Verify content structure matches
        # Note: response.messages includes both request and response messages
        assert len(response.messages) >= len(fixture["messages"])

        # Find the assistant message in the response
        assistant_messages = [m for m in response.messages if m.role == "assistant"]
        assert len(assistant_messages) == len(fixture["messages"])

        # Verify text content from assistant message
        text_blocks = [b for b in assistant_messages[0].content if b.type == "text"]
        fixture_text_blocks = [b for b in fixture["messages"][0]["content"] if b["type"] == "text"]
        assert len(text_blocks) == len(fixture_text_blocks)
        assert text_blocks[0].text == fixture_text_blocks[0]["text"]

    def test_thinking_mode_with_real_structure(self, mock_genai_client, gemini_fixtures):
        """Test thinking mode (Gemini 2.5+) using real API response structure."""
        fixture = gemini_fixtures["thinking_mode"]

        # Create mock response from fixture
        mock_response = create_mock_response_from_fixture(fixture)
        mock_genai_client.models.generate_content.return_value = mock_response

        # Test client
        client = GeminiStatelessClient(api_key="test-key")
        response = client.create({
            "model": fixture["model"],
            "messages": [{"role": "user", "content": "What is 15 factorial?"}],
            "thinking_config": {"include_thoughts": True},
        })

        # Verify response structure
        assert response.provider == fixture["provider"]
        assert response.model == fixture["model"]

        # Verify thinking tokens are tracked
        assert response.usage["thinking_tokens"] == fixture["usage"]["thinking_tokens"]
        assert fixture["usage"]["thinking_tokens"] == 100  # From real API

        # Verify both thinking and text content exist
        assistant_messages = [m for m in response.messages if m.role == "assistant"]
        assert len(assistant_messages) > 0

        thinking_blocks = [b for b in assistant_messages[0].content if b.type == "thinking"]
        text_blocks = [b for b in assistant_messages[0].content if b.type == "text"]

        # Should have both thinking and text content
        assert len(thinking_blocks) == 1, "Should have thinking content"
        assert len(text_blocks) == 1, "Should have text content"

        # Verify thinking content matches fixture
        fixture_thinking = [b for b in fixture["messages"][0]["content"] if b["type"] == "thinking"]
        assert thinking_blocks[0].thinking == fixture_thinking[0]["thinking"]

        # Verify text content matches fixture
        fixture_text = [b for b in fixture["messages"][0]["content"] if b["type"] == "text"]
        assert text_blocks[0].text == fixture_text[0]["text"]

        # Verify cost calculation includes thinking token pricing
        assert response.cost == fixture["cost"]
        # Thinking mode should be more expensive due to thinking tokens
        assert response.cost > 0.001  # Should be > $0.001 due to 100 thinking tokens

    def test_structured_output_with_real_structure(self, mock_genai_client, gemini_fixtures):
        """Test structured JSON output using real API response structure."""
        fixture = gemini_fixtures["structured_output"]

        # Create mock response from fixture
        mock_response = create_mock_response_from_fixture(fixture)
        mock_genai_client.models.generate_content.return_value = mock_response

        # Test client
        client = GeminiStatelessClient(api_key="test-key")
        response = client.create({
            "model": fixture["model"],
            "messages": [{"role": "user", "content": "Generate JSON"}],
            "response_mime_type": "application/json",
        })

        # Verify response structure
        assert response.provider == fixture["provider"]
        assert response.model == fixture["model"]

        # Verify JSON content is preserved correctly
        text_content = [b for b in response.messages[0].content if b.type == "text"]
        assert len(text_content) > 0

        # The response should contain valid JSON
        json_text = text_content[0].text
        parsed_json = json.loads(json_text)
        assert "colors" in parsed_json
        assert isinstance(parsed_json["colors"], list)
        assert len(parsed_json["colors"]) == 3  # From fixture

        # Verify each color has expected structure
        for color in parsed_json["colors"]:
            assert "name" in color
            assert "hex" in color

    def test_tool_calling_with_real_structure(self, mock_genai_client, gemini_fixtures):
        """Test tool calling using real API response structure."""
        fixture = gemini_fixtures["tool_calling"]

        # Create mock response from fixture
        mock_response = create_mock_response_from_fixture(fixture)
        mock_genai_client.models.generate_content.return_value = mock_response

        # Test client
        client = GeminiStatelessClient(api_key="test-key")
        response = client.create({
            "model": fixture["model"],
            "messages": [{"role": "user", "content": "Use the calculate function to add 123 and 456."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "Perform a mathematical calculation",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                                "a": {"type": "number"},
                                "b": {"type": "number"},
                            },
                            "required": ["operation", "a", "b"],
                        },
                    },
                }
            ],
        })

        # Verify response structure
        assert response.provider == fixture["provider"]
        assert response.model == fixture["model"]

        # Verify usage tracking
        assert response.usage["prompt_tokens"] == fixture["usage"]["prompt_tokens"]
        assert response.usage["completion_tokens"] == fixture["usage"]["completion_tokens"]

        # Find assistant messages (tool calls come from assistant)
        assistant_messages = [m for m in response.messages if m.role == "assistant"]
        assert len(assistant_messages) > 0

        # Verify tool call content exists
        tool_calls = [b for b in assistant_messages[0].content if b.type == "tool_call"]
        assert len(tool_calls) > 0, "Should have tool call in response"

        # Verify tool call structure
        tool_call = tool_calls[0]
        assert tool_call.id is not None, "Tool call should have ID"
        assert tool_call.name == "calculate", f"Expected 'calculate' but got '{tool_call.name}'"
        assert tool_call.arguments is not None, "Tool call should have arguments"

        # Verify arguments are valid JSON
        import json

        args = json.loads(tool_call.arguments)
        assert "operation" in args, "Should have 'operation' argument"
        assert "a" in args, "Should have 'a' argument"
        assert "b" in args, "Should have 'b' argument"
        assert args["operation"] == "add", f"Expected operation='add' but got '{args['operation']}'"
        assert args["a"] == 123, f"Expected a=123 but got {args['a']}"
        assert args["b"] == 456, f"Expected b=456 but got {args['b']}"

        # Verify fixture matches
        fixture_tool_call = fixture["messages"][0]["content"][0]
        assert fixture_tool_call["type"] == "tool_call"
        assert fixture_tool_call["name"] == tool_call.name
        # Note: Tool call IDs are generated by the client (UUID), not from fixtures
        # So we just verify both have IDs but don't compare them
        assert len(fixture_tool_call["id"]) > 0
        assert len(tool_call.id) > 0

    def test_multimodal_with_real_structure(self, mock_genai_client, gemini_fixtures):
        """Test multimodal content using real API response structure."""
        fixture = gemini_fixtures["multimodal"]

        # Create mock response from fixture
        mock_response = create_mock_response_from_fixture(fixture)
        mock_genai_client.models.generate_content.return_value = mock_response

        # Test client
        client = GeminiStatelessClient(api_key="test-key")
        response = client.create({
            "model": fixture["model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                    ],
                }
            ],
        })

        # Verify response structure
        assert response.provider == fixture["provider"]
        assert response.model == fixture["model"]

        # Verify usage tracking
        assert response.usage["prompt_tokens"] == fixture["usage"]["prompt_tokens"]
        assert response.usage["completion_tokens"] == fixture["usage"]["completion_tokens"]

        # Verify response has text content
        text_blocks = [b for b in response.messages[0].content if b.type == "text"]
        assert len(text_blocks) > 0
        assert len(text_blocks[0].text) > 0

    def test_cost_calculation_matches_real_api(self, mock_genai_client, gemini_fixtures):
        """Test that cost calculation matches real API costs."""
        for fixture_name, fixture in gemini_fixtures.items():
            # Create mock response
            mock_response = create_mock_response_from_fixture(fixture)
            mock_genai_client.models.generate_content.return_value = mock_response

            # Test client
            client = GeminiStatelessClient(api_key="test-key")
            response = client.create({"model": fixture["model"], "messages": [{"role": "user", "content": "test"}]})

            # Cost should match fixture (with small tolerance for floating point)
            assert abs(response.cost - fixture["cost"]) < 1e-6, (
                f"Cost mismatch for {fixture_name}: expected {fixture['cost']}, got {response.cost}"
            )

    def test_message_retrieval_with_real_responses(self, mock_genai_client, gemini_fixtures):
        """Test message_retrieval() with real API response structures."""
        fixture = gemini_fixtures["basic_text"]

        # Create mock response
        mock_response = create_mock_response_from_fixture(fixture)
        mock_genai_client.models.generate_content.return_value = mock_response

        # Test client
        client = GeminiStatelessClient(api_key="test-key")
        response = client.create({"model": fixture["model"], "messages": [{"role": "user", "content": "test"}]})

        # Test message retrieval
        messages = client.message_retrieval(response)

        # Should extract text from fixture
        fixture_texts = [
            block["text"] for msg in fixture["messages"] for block in msg["content"] if block["type"] == "text"
        ]
        assert len(messages) == len(fixture_texts)
        assert messages[0] == fixture_texts[0]

    def test_get_usage_with_real_responses(self, mock_genai_client, gemini_fixtures):
        """Test get_usage() returns correct data from real API responses."""
        for fixture_name, fixture in gemini_fixtures.items():
            # Create mock response
            mock_response = create_mock_response_from_fixture(fixture)
            mock_genai_client.models.generate_content.return_value = mock_response

            # Test client
            client = GeminiStatelessClient(api_key="test-key")
            response = client.create({"model": fixture["model"], "messages": [{"role": "user", "content": "test"}]})

            # Get usage
            usage = client.get_usage(response)

            # Verify all fields match fixture
            assert usage["prompt_tokens"] == fixture["usage"]["prompt_tokens"]
            assert usage["completion_tokens"] == fixture["usage"]["completion_tokens"]
            assert usage["total_tokens"] == fixture["usage"]["total_tokens"]
            assert usage["model"] == fixture["model"]
            assert abs(usage["cost"] - fixture["cost"]) < 1e-6

    def test_response_id_preservation(self, mock_genai_client, gemini_fixtures):
        """Test that response IDs from real API are preserved."""
        for fixture_name, fixture in gemini_fixtures.items():
            # Create mock response
            mock_response = create_mock_response_from_fixture(fixture)
            mock_genai_client.models.generate_content.return_value = mock_response

            # Test client
            client = GeminiStatelessClient(api_key="test-key")
            response = client.create({"model": fixture["model"], "messages": [{"role": "user", "content": "test"}]})

            # Response should have an ID (even if not from fixture, client generates one)
            assert response.id is not None
            assert len(response.id) > 0


class TestFixtureQuality:
    """Tests to verify fixture quality and completeness."""

    def test_fixtures_exist(self, gemini_fixtures):
        """Verify that required fixtures exist."""
        required_fixtures = ["basic_text", "thinking_mode", "tool_calling", "structured_output", "multimodal"]
        for fixture_name in required_fixtures:
            assert fixture_name in gemini_fixtures, f"Missing required fixture: {fixture_name}"

    def test_fixture_structure(self, gemini_fixtures):
        """Verify that each fixture has required fields."""
        required_fields = ["id", "model", "provider", "usage", "cost", "messages"]

        for fixture_name, fixture in gemini_fixtures.items():
            for field in required_fields:
                assert field in fixture, f"Fixture '{fixture_name}' missing field: {field}"

            # Verify usage structure
            usage = fixture["usage"]
            assert "prompt_tokens" in usage
            assert "completion_tokens" in usage
            assert "total_tokens" in usage
            assert "thinking_tokens" in usage

            # Verify messages structure
            assert len(fixture["messages"]) > 0
            for message in fixture["messages"]:
                assert "role" in message
                assert "content" in message
                assert len(message["content"]) > 0

                for content_block in message["content"]:
                    assert "type" in content_block

    def test_fixtures_have_realistic_costs(self, gemini_fixtures):
        """Verify that fixture costs are realistic."""
        for fixture_name, fixture in gemini_fixtures.items():
            # Cost should be positive and reasonable
            assert fixture["cost"] > 0, f"Fixture '{fixture_name}' has non-positive cost"
            assert fixture["cost"] < 1.0, (
                f"Fixture '{fixture_name}' has unrealistically high cost (>${fixture['cost']})"
            )

    def test_fixtures_have_valid_token_counts(self, gemini_fixtures):
        """Verify that token counts are valid."""
        for fixture_name, fixture in gemini_fixtures.items():
            usage = fixture["usage"]

            # All counts should be non-negative
            assert usage["prompt_tokens"] >= 0
            assert usage["completion_tokens"] >= 0
            assert usage["total_tokens"] >= 0
            assert usage["thinking_tokens"] >= 0

            # Total should equal prompt + completion + thinking
            # (Gemini includes thinking tokens in total_tokens)
            expected_total = usage["prompt_tokens"] + usage["completion_tokens"] + usage["thinking_tokens"]
            assert usage["total_tokens"] == expected_total, (
                f"Fixture '{fixture_name}' has incorrect total_tokens: got {usage['total_tokens']}, expected {expected_total}"
            )

    def test_thinking_mode_fixture_has_thinking_content(self, gemini_fixtures):
        """Verify that thinking_mode fixture has both thinking and text content."""
        if "thinking_mode" in gemini_fixtures:
            fixture = gemini_fixtures["thinking_mode"]

            # Should have thinking tokens
            assert fixture["usage"]["thinking_tokens"] > 0, "thinking_mode should have thinking_tokens"

            # Should have at least one message
            assert len(fixture["messages"]) > 0

            # Check content types
            content = fixture["messages"][0]["content"]
            content_types = {block["type"] for block in content}

            # Should have both thinking and text
            assert "thinking" in content_types, "thinking_mode should have thinking content"
            assert "text" in content_types, "thinking_mode should have text content"

            # Verify thinking content is not empty
            thinking_blocks = [b for b in content if b["type"] == "thinking"]
            assert len(thinking_blocks) > 0
            assert len(thinking_blocks[0]["thinking"]) > 0, "Thinking content should not be empty"

    def test_tool_calling_fixture_has_tool_call_content(self, gemini_fixtures):
        """Verify that tool_calling fixture has proper tool call structure."""
        if "tool_calling" in gemini_fixtures:
            fixture = gemini_fixtures["tool_calling"]

            # Should have at least one message
            assert len(fixture["messages"]) > 0

            # Check content types
            content = fixture["messages"][0]["content"]
            content_types = {block["type"] for block in content}

            # Should have tool_call content
            assert "tool_call" in content_types, "tool_calling fixture should have tool_call content"

            # Verify tool call structure
            tool_call_blocks = [b for b in content if b["type"] == "tool_call"]
            assert len(tool_call_blocks) > 0

            tool_call = tool_call_blocks[0]
            assert "id" in tool_call, "Tool call should have id"
            assert "name" in tool_call, "Tool call should have name"
            assert "arguments" in tool_call, "Tool call should have arguments"

            # Verify arguments are valid JSON
            import json

            args = json.loads(tool_call["arguments"])
            assert isinstance(args, dict), "Tool call arguments should be a JSON object"

            # Verify no thinking tokens for regular tool calling
            assert fixture["usage"]["thinking_tokens"] == 0, "Regular tool calling should not have thinking tokens"
