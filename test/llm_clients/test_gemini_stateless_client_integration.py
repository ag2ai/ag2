# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for GeminiStatelessClient.

These tests require actual Gemini API calls and GEMINI_API_KEY environment variable.
Run with: bash scripts/test-core-llm.sh test/llm_clients/test_gemini_stateless_client_integration.py
"""

import json
import os

import pytest

from autogen.import_utils import run_for_optional_imports
from autogen.llm_clients import GeminiStatelessClient
from autogen.llm_clients.models import UnifiedResponse
from test.credentials import Credentials

# Sample responses will be stored here for unit test fixtures
SAMPLE_RESPONSES = {}


def serialize_response(response: UnifiedResponse) -> dict:
    """Serialize UnifiedResponse to JSON-compatible dict for fixtures."""
    serialized = {
        "id": response.id,
        "model": response.model,
        "provider": response.provider,
        "usage": response.usage,
        "cost": response.cost,
        "messages": [],
    }

    for message in response.messages:
        msg_dict = {
            "role": message.role,
            "content": [],
        }

        for block in message.content:
            block_dict = {"type": block.type}

            if block.type == "text":
                block_dict["text"] = block.text
            elif block.type == "thinking":
                block_dict["thinking"] = block.thinking
            elif block.type == "tool_call":
                block_dict.update({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.arguments,
                })
            elif block.type == "image":
                block_dict["url"] = block.url if hasattr(block, "url") else None

            msg_dict["content"].append(block_dict)

        serialized["messages"].append(msg_dict)

    return serialized


@pytest.mark.gemini
@run_for_optional_imports(["google.genai"], "gemini")
class TestGeminiStatelessClientIntegration:
    """Integration tests requiring actual Gemini API calls."""

    def test_basic_text_generation(self, credentials_gemini_flash: Credentials):
        """Test basic text generation with Gemini API."""
        client = GeminiStatelessClient(api_key=credentials_gemini_flash.api_key)

        response = client.create({
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Say 'Hello, World!' and nothing else."}],
            "temperature": 0.0,
        })

        # Verify response structure
        assert response.provider == "gemini"
        assert response.model == "gemini-2.0-flash"
        assert len(response.messages) > 0
        assert response.messages[0].role == "assistant"

        # Verify content
        assert len(response.messages[0].content) > 0
        text_content = [block for block in response.messages[0].content if block.type == "text"]
        assert len(text_content) > 0
        assert "Hello" in text_content[0].text

        # Verify usage tracking
        assert "prompt_tokens" in response.usage
        assert "completion_tokens" in response.usage
        assert "total_tokens" in response.usage
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0

        # Verify cost calculation
        assert response.cost is not None
        assert response.cost > 0

        # Store complete response structure for unit tests
        SAMPLE_RESPONSES["basic_text"] = serialize_response(response)

        print(f"\n✅ Basic text generation: {response.text}")
        print(f"   Usage: {response.usage}")
        print(f"   Cost: ${response.cost:.6f}")

    @pytest.mark.gemini
    @run_for_optional_imports(["google.genai"], "gemini")
    def test_thinking_mode_gemini_25(self, credentials_gemini_flash: Credentials):
        """Test thinking mode with Gemini 2.5+ models."""
        client = GeminiStatelessClient(api_key=credentials_gemini_flash.api_key)

        response = client.create({
            "model": "gemini-2.5-flash",  # or gemini-2.5-pro
            "messages": [{"role": "user", "content": "What is 15 factorial? Think step by step."}],
            "thinking_config": {"include_thoughts": True, "thinking_budget": 100},
            "temperature": 0.0,
        })

        # Verify thinking content exists
        thinking_content = [block for block in response.messages[0].content if block.type == "thinking"]

        if thinking_content:  # Thinking may or may not be present
            print("\n✅ Thinking mode enabled:")
            for thinking in thinking_content:
                print(f"   Thought: {thinking.thinking[:100]}...")

            # Verify thinking tokens tracked
            assert "thinking_tokens" in response.usage
            if response.usage["thinking_tokens"] > 0:
                print(f"   Thinking tokens: {response.usage['thinking_tokens']}")

        # Verify answer exists
        text_content = [block for block in response.messages[0].content if block.type == "text"]
        assert len(text_content) > 0
        print(f"   Answer: {text_content[0].text[:200]}")

        # Store complete response structure
        SAMPLE_RESPONSES["thinking_mode"] = serialize_response(response)

    def test_tool_calling(self, credentials_gemini_flash: Credentials):
        """Test function/tool calling with Gemini."""
        client = GeminiStatelessClient(api_key=credentials_gemini_flash.api_key)

        # Simple calculator tool - model is very likely to call this
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform a mathematical calculation. Use this for any arithmetic operation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "description": "The mathematical operation to perform",
                                "enum": ["add", "subtract", "multiply", "divide"],
                            },
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"},
                        },
                        "required": ["operation", "a", "b"],
                    },
                },
            }
        ]

        response = client.create({
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Use the calculate function to add 123 and 456."}],
            "tools": tools,
            "temperature": 0.0,
        })

        # Verify tool calls
        tool_calls = [block for block in response.messages[0].content if block.type == "tool_call"]

        if tool_calls:  # Tool call may or may not be made
            print("\n✅ Tool calling works:")
            for tool_call in tool_calls:
                print(f"   Tool: {tool_call.name}")
                print(f"   Args: {tool_call.arguments}")

                # Verify structure
                assert tool_call.id  # Should have generated UUID
                assert tool_call.name == "calculate"

                # Parse arguments
                args = json.loads(tool_call.arguments)
                assert "operation" in args
                assert "a" in args
                assert "b" in args
                print(f"   Parsed: {args['operation']}({args['a']}, {args['b']})")

            # Store complete response structure
            SAMPLE_RESPONSES["tool_calling"] = serialize_response(response)
        else:
            # Model might respond with text instead of tool call
            text_content = [block for block in response.messages[0].content if block.type == "text"]
            print("\n⚠️  Model responded with text instead of tool call:")
            print(f"   {text_content[0].text[:200]}")

    @pytest.mark.gemini
    @run_for_optional_imports(["google.genai"], "gemini")
    def test_code_execution(self, credentials_gemini_flash: Credentials):
        """Test code execution feature (Gemini-specific)."""
        client = GeminiStatelessClient(api_key=credentials_gemini_flash.api_key)

        response = client.create({
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Calculate the sum of first 10 prime numbers using Python."}],
            "tools": [{"code_execution": {}}],
            "temperature": 0.0,
        })

        # Check for code execution content
        code_blocks = [block for block in response.messages[0].content if block.type == "executable_code"]
        result_blocks = [block for block in response.messages[0].content if block.type == "code_execution_result"]

        if code_blocks:
            print("\n✅ Code execution enabled:")
            for code_block in code_blocks:
                print(f"   Language: {code_block.content.get('language', 'unknown')}")
                print(f"   Code:\n{code_block.content.get('code', '')[:200]}")

        if result_blocks:
            for result_block in result_blocks:
                print(f"   Outcome: {result_block.content.get('outcome', 'unknown')}")
                print(f"   Output: {result_block.content.get('output', '')[:200]}")

            # Store complete response structure
            SAMPLE_RESPONSES["code_execution"] = serialize_response(response)

    def test_multimodal_with_image(self, credentials_gemini_flash: Credentials):
        """Test multimodal content with image input."""
        client = GeminiStatelessClient(api_key=credentials_gemini_flash.api_key)

        # Simple test image (1x1 red pixel PNG in base64)
        test_image_base64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )

        response = client.create({
            "model": "gemini-2.0-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{test_image_base64}"}},
                    ],
                }
            ],
            "temperature": 0.0,
        })

        # Verify response
        assert len(response.messages) > 0
        text_content = [block for block in response.messages[0].content if block.type == "text"]
        assert len(text_content) > 0

        print("\n✅ Multimodal image analysis:")
        print(f"   Response: {text_content[0].text[:200]}")

        # Store complete response structure
        SAMPLE_RESPONSES["multimodal"] = serialize_response(response)

    def test_structured_output(self, credentials_gemini_flash: Credentials):
        """Test structured JSON output."""
        client = GeminiStatelessClient(api_key=credentials_gemini_flash.api_key)

        response = client.create({
            "model": "gemini-2.0-flash",
            "messages": [
                {
                    "role": "user",
                    "content": "List 3 colors in JSON format with 'colors' array containing objects with 'name' and 'hex' fields.",
                }
            ],
            "response_format": {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "colors": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}, "hex": {"type": "string"}},
                                "required": ["name", "hex"],
                            },
                        }
                    },
                    "required": ["colors"],
                },
            },
            "temperature": 0.0,
        })

        # Verify JSON output
        text_content = [block for block in response.messages[0].content if block.type == "text"]
        assert len(text_content) > 0

        # Parse JSON
        json_response = json.loads(text_content[0].text)
        assert "colors" in json_response
        assert isinstance(json_response["colors"], list)
        assert len(json_response["colors"]) >= 3

        print("\n✅ Structured output:")
        print(f"   Colors: {json.dumps(json_response, indent=2)[:200]}")

        # Store complete response structure
        SAMPLE_RESPONSES["structured_output"] = serialize_response(response)

    def test_usage_and_cost_tracking(self, credentials_gemini_flash: Credentials):
        """Test that usage and cost tracking works correctly."""
        client = GeminiStatelessClient(api_key=credentials_gemini_flash.api_key)

        response = client.create({
            "model": "gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            "temperature": 0.0,
        })

        # Verify all usage fields
        assert "prompt_tokens" in response.usage
        assert "completion_tokens" in response.usage
        assert "total_tokens" in response.usage
        assert "thinking_tokens" in response.usage  # Should be 0 for non-thinking

        # Verify values
        assert response.usage["prompt_tokens"] > 0
        assert response.usage["completion_tokens"] > 0
        assert response.usage["total_tokens"] == response.usage["prompt_tokens"] + response.usage["completion_tokens"]
        assert response.usage["thinking_tokens"] == 0  # No thinking mode

        # Verify cost
        assert response.cost is not None
        assert response.cost > 0

        # Verify cost calculation method
        cost_from_method = client.cost(response)
        assert cost_from_method == response.cost

        print("\n✅ Usage tracking:")
        print(f"   Prompt tokens: {response.usage['prompt_tokens']}")
        print(f"   Completion tokens: {response.usage['completion_tokens']}")
        print(f"   Thinking tokens: {response.usage['thinking_tokens']}")
        print(f"   Total tokens: {response.usage['total_tokens']}")
        print(f"   Cost: ${response.cost:.6f}")


@pytest.mark.gemini
@run_for_optional_imports(["google.genai"], "gemini")
def test_vertex_ai_mode():
    """Test Vertex AI initialization (without actual API call)."""
    # Only test initialization, not actual API call
    # This would require GCP credentials

    # Test that initialization doesn't crash
    try:
        client = GeminiStatelessClient(
            vertexai=True,
            project="test-project",
            location="us-central1",
        )
        assert client.use_vertexai is True
        print("\n✅ Vertex AI mode initialization successful")
    except Exception as e:
        # Expected if no GCP credentials
        print(f"\n⚠️  Vertex AI initialization failed (expected without credentials): {e}")


def test_save_sample_responses():
    """Save captured sample responses for unit test fixtures."""
    if SAMPLE_RESPONSES:
        output_file = "test/llm_clients/fixtures/gemini_sample_responses.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(SAMPLE_RESPONSES, f, indent=2)

        print(f"\n✅ Saved {len(SAMPLE_RESPONSES)} sample responses to {output_file}")
