# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for OpenAIResponsesV2Client.

These tests require a valid OPENAI_API_KEY environment variable.
Run with: pytest test/llm_clients/test_openai_responses_v2_client_integration.py -m integration
"""

import os

import pytest

# Skip all tests if no API key
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set",
    ),
]


@pytest.fixture
def client():
    """Create a fresh client for each test."""
    from autogen.llm_clients.openai_responses_v2 import OpenAIResponsesV2Client

    return OpenAIResponsesV2Client()


class TestBasicUsage:
    """Test basic API usage."""

    @pytest.mark.integration
    def test_simple_request(self, client):
        """Test basic request and response structure."""
        response = client.create({
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
        })

        assert response.id is not None
        assert response.model is not None
        assert len(response.messages) >= 1
        assert response.messages[0].get_text() is not None

    @pytest.mark.integration
    def test_response_structure(self, client):
        """Test UnifiedResponse structure."""
        response = client.create({
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": "Count to 3"}],
        })

        # Check usage
        assert response.usage is not None
        assert response.usage.get("total_tokens", 0) > 0

        # Check cost
        assert response.cost >= 0


class TestStatefulConversations:
    """Test stateful conversation management."""

    @pytest.mark.integration
    def test_conversation_context(self, client):
        """Test that client maintains conversation context."""
        # First message
        response1 = client.create({
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": "My name is TestUser. Remember this."}],
        })
        assert response1.id is not None

        # Second message - should remember context
        response2 = client.create({
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": "What is my name?"}],
        })

        # The model should remember the name
        text = response2.messages[0].get_text().lower()
        assert "testuser" in text or "test" in text

    @pytest.mark.integration
    def test_reset_conversation(self, client):
        """Test conversation reset."""
        # Set context
        client.create({
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": "My secret code is XYZ123."}],
        })

        # Reset conversation
        client.reset_conversation()

        # Ask for secret - model shouldn't know
        response = client.create({
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": "What is my secret code?"}],
        })

        text = response.messages[0].get_text().lower()
        # Model should not know the code after reset
        assert "xyz123" not in text or "don't" in text or "haven't" in text


class TestMultimodal:
    """Test multimodal support."""

    @pytest.mark.integration
    def test_create_multimodal_message(self):
        """Test creating multimodal message structure."""
        from autogen.llm_clients.openai_responses_v2 import OpenAIResponsesV2Client

        message = OpenAIResponsesV2Client.create_multimodal_message(
            text="What do you see?",
            images=["https://via.placeholder.com/150"],
            role="user",
        )

        assert message["role"] == "user"
        assert isinstance(message["content"], list)
        assert len(message["content"]) == 2  # 1 text + 1 image

    @pytest.mark.integration
    def test_image_description(self, client):
        """Test image description with vision model."""
        from autogen.llm_clients.openai_responses_v2 import OpenAIResponsesV2Client

        message = OpenAIResponsesV2Client.create_multimodal_message(
            text="Describe this image briefly.",
            images=["https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400"],
            role="user",
        )

        response = client.create({
            "model": "gpt-4.1-nano",
            "messages": [message],
        })

        assert response.messages[0].get_text() is not None


class TestBuiltInTools:
    """Test built-in tools."""

    @pytest.mark.integration
    def test_web_search(self, client):
        """Test web search built-in tool."""
        response = client.create({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "What is the current weather in New York?"}],
            "built_in_tools": ["web_search"],
        })

        # Should have a response
        assert response.messages[0].get_text() is not None

        # Check for citations
        from autogen.llm_clients.openai_responses_v2 import OpenAIResponsesV2Client

        citations = OpenAIResponsesV2Client.get_citations(response)
        # May or may not have citations depending on query
        assert isinstance(citations, list)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_image_generation(self, client):
        """Test image generation built-in tool."""
        from autogen.llm_clients.openai_responses_v2 import OpenAIResponsesV2Client

        client.set_image_output_params(quality="low", size="1024x1024")

        response = client.create({
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": "Generate a simple red square"}],
            "built_in_tools": ["image_generation"],
        })

        images = OpenAIResponsesV2Client.get_generated_images(response)
        # Note: Image generation may not always produce images
        assert isinstance(images, list)


class TestStructuredOutput:
    """Test structured output with Pydantic models."""

    @pytest.mark.integration
    def test_pydantic_model_output(self, client):
        """Test structured output with Pydantic model."""
        from pydantic import BaseModel

        from autogen.llm_clients.openai_responses_v2 import OpenAIResponsesV2Client

        class Person(BaseModel):
            name: str
            age: int
            occupation: str

        response = client.create({
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": "Generate a fictional person's profile"}],
            "response_format": Person,
        })

        parsed = OpenAIResponsesV2Client.get_parsed_object(response)

        if parsed:
            assert isinstance(parsed.name, str)
            assert isinstance(parsed.age, int)
            assert isinstance(parsed.occupation, str)


class TestCostTracking:
    """Test cost tracking functionality."""

    @pytest.mark.integration
    def test_per_request_cost(self, client):
        """Test per-request cost tracking."""
        from autogen.llm_clients.openai_responses_v2 import OpenAIResponsesV2Client

        response = client.create({
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": "Say hi"}],
        })

        usage = OpenAIResponsesV2Client.get_usage(response)

        assert "total_tokens" in usage
        assert "cost" in usage
        assert usage["total_tokens"] > 0

    @pytest.mark.integration
    def test_cumulative_cost(self, client):
        """Test cumulative cost tracking."""
        # Make multiple requests
        for i in range(3):
            client.create({
                "model": "gpt-4.1-nano",
                "messages": [{"role": "user", "content": f"Count to {i + 1}"}],
            })

        cumulative = client.get_cumulative_usage()

        assert cumulative["total_tokens"] > 0
        assert cumulative["prompt_tokens"] > 0
        assert cumulative["completion_tokens"] > 0

    @pytest.mark.integration
    def test_cost_reset(self, client):
        """Test cost reset functionality."""
        client.create({
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        client.reset_all_costs()

        assert client.get_total_costs() == 0.0


class TestV1Compatibility:
    """Test V1 backward compatibility."""

    @pytest.mark.integration
    def test_create_v1_compatible(self, client):
        """Test ChatCompletion-like response format."""
        response = client.create_v1_compatible({
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": "Hello!"}],
        })

        # Should have ChatCompletion-like structure
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert response.choices[0].message.content is not None

        # Should have usage
        assert hasattr(response, "usage")
        assert response.usage.total_tokens > 0

        # Should have cost
        assert hasattr(response, "cost")


class TestCustomTools:
    """Test custom function tools."""

    @pytest.mark.integration
    def test_function_tool_call(self, client):
        """Test that model can call custom function tools."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "City name"}},
                        "required": ["city"],
                    },
                },
            }
        ]

        response = client.create({
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
            "tools": tools,
        })

        # Check for tool calls in response
        from autogen.llm_clients.models.content_blocks import ToolCallContent

        has_tool_call = False
        for msg in response.messages:
            for block in msg.content:
                if isinstance(block, ToolCallContent):
                    has_tool_call = True
                    assert block.name == "get_weather"
                    break

        # Model may or may not decide to use the tool
        assert response.messages[0].get_text() is not None or has_tool_call


class TestShellTool:
    """Test shell tool integration."""

    @pytest.mark.integration
    def test_shell_tool_config(self):
        """Test shell tool configuration."""
        from autogen.llm_clients.openai_responses_v2 import OpenAIResponsesV2Client

        client = OpenAIResponsesV2Client()
        client.set_shell_params(
            allowed_commands=["ls", "pwd", "echo"],
            denied_commands=["rm", "sudo"],
            enable_command_filtering=True,
        )

        assert client._shell_allowed_commands == ["ls", "pwd", "echo"]
        assert client._shell_denied_commands == ["rm", "sudo"]
        assert client._shell_enable_command_filtering is True

    @pytest.mark.integration
    def test_get_shell_calls(self, client):
        """Test get_shell_calls static method."""
        from autogen.llm_clients.openai_responses_v2 import OpenAIResponsesV2Client

        # This test verifies the method exists and works
        # Actual shell execution requires API support
        response = client.create({
            "model": "gpt-4.1-nano",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        shell_calls = OpenAIResponsesV2Client.get_shell_calls(response)
        assert isinstance(shell_calls, list)
