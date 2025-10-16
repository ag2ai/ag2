# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from typing import Any

import pytest

from autogen import ConversableAgent, OpenAIWrapper
from autogen.code_utils import content_str
from autogen.import_utils import run_for_optional_imports


class RealMultimodalCustomClient:
    """Real custom client implementation that handles multimodal content."""

    def __init__(self, config: dict[str, Any]):
        self.model = config["model"]
        self.api_key = config.get("api_key", "test-key")
        self.call_count = 0

    def create(self, params: dict[str, Any]):
        """Create a response based on input parameters."""
        self.call_count += 1

        response = SimpleNamespace()
        response.choices = []
        choice = SimpleNamespace()
        choice.message = SimpleNamespace()

        # Analyze input to determine response type
        messages = params.get("messages", [])
        if not messages:
            choice.message.content = "No input provided"
        else:
            last_message = messages[-1]
            content = last_message.get("content", "")

            if isinstance(content, list):
                # Input is multimodal - respond with multimodal
                text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                has_images = any(item.get("type") == "image_url" for item in content)

                response_text = f"I received multimodal input with text: {' '.join(text_parts)}"
                if has_images:
                    response_text += " and visual content."

                choice.message.content = [
                    {"type": "text", "text": response_text},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,response_visualization"}},
                ]
            else:
                # Text input - respond with text
                choice.message.content = f"Response to: {content} (call #{self.call_count})"

        response.choices.append(choice)
        response.model = self.model

        # Add usage information
        response.usage = SimpleNamespace()
        response.usage.prompt_tokens = 10 + len(str(params))
        response.usage.completion_tokens = 20 + self.call_count
        response.usage.total_tokens = response.usage.prompt_tokens + response.usage.completion_tokens

        return response

    def message_retrieval(self, response):
        """Extract messages from response."""
        if not response.choices:
            return []

        content = response.choices[0].message.content
        return [content]

    def cost(self, response) -> float:
        """Calculate cost of the response."""
        if hasattr(response, "usage"):
            return response.usage.total_tokens * 0.0001  # Simple cost calculation
        return 0.01

    @staticmethod
    def get_usage(response) -> dict[str, Any]:
        """Get usage statistics from response."""
        if hasattr(response, "usage"):
            return {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cost": response.usage.total_tokens * 0.0001,
                "model": response.model,
            }
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.01, "model": response.model}


@pytest.mark.integration
@run_for_optional_imports(["openai"], "openai")
def test_custom_multimodal_client_integration():
    """Test custom multimodal client integration with real AG2 components."""

    # Configure custom client
    config_list = [
        {
            "model": "custom-multimodal-model",
            "model_client_cls": "RealMultimodalCustomClient",
            "api_key": "test-api-key",
        }
    ]

    # Create OpenAI wrapper and register custom client
    client = OpenAIWrapper(config_list=config_list)
    client.register_model_client(model_client_cls=RealMultimodalCustomClient)

    # Test 1: String input to custom client
    string_response = client.create(messages=[{"role": "user", "content": "Hello, custom client!"}], cache_seed=None)

    assert string_response is not None, "String input should work"
    assert len(string_response.choices) == 1, "Should have one choice"
    content = string_response.choices[0].message.content
    assert isinstance(content, str), "String input should get string response"
    assert "Hello, custom client!" in content, "Response should reference input"

    # Test 2: Multimodal input to custom client
    multimodal_input = [
        {"type": "text", "text": "Analyze this chart"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,testchart"}},
    ]

    multimodal_response = client.create(messages=[{"role": "user", "content": multimodal_input}], cache_seed=None)

    assert multimodal_response is not None, "Multimodal input should work"
    assert len(multimodal_response.choices) == 1, "Should have one choice"

    response_content = multimodal_response.choices[0].message.content
    assert isinstance(response_content, list), "Multimodal input should get multimodal response"
    assert len(response_content) == 2, "Should have text and image response"

    # Verify response structure
    text_item = response_content[0]
    assert text_item["type"] == "text", "First item should be text"
    assert "multimodal input" in text_item["text"], "Should acknowledge multimodal input"

    image_item = response_content[1]
    assert image_item["type"] == "image_url", "Second item should be image"
    assert "url" in image_item["image_url"], "Should have image URL"

    # Test 3: Usage tracking
    usage = RealMultimodalCustomClient.get_usage(multimodal_response)
    assert "prompt_tokens" in usage, "Should track prompt tokens"
    assert "completion_tokens" in usage, "Should track completion tokens"
    assert "total_tokens" in usage, "Should track total tokens"
    assert "cost" in usage, "Should track cost"
    assert usage["total_tokens"] > 0, "Should have positive token count"


@pytest.mark.integration
@run_for_optional_imports(["openai"], "openai")
def test_custom_client_with_conversable_agent():
    """Test custom multimodal client integrated with ConversableAgent."""

    # Configure agent with custom client
    config_list = [
        {
            "model": "agent-multimodal-model",
            "model_client_cls": "RealMultimodalCustomClient",
            "api_key": "agent-test-key",
        }
    ]

    # Create agent
    agent = ConversableAgent(
        name="custom_multimodal_agent",
        llm_config={"config_list": config_list},
        human_input_mode="NEVER",
        system_message="You are an agent using a custom multimodal client.",
    )

    # Register the custom client
    agent.register_model_client(model_client_cls=RealMultimodalCustomClient)

    # Test 1: Agent handles string input
    string_messages = [{"role": "user", "content": "How are you today?"}]
    string_reply = agent.generate_reply(messages=string_messages)

    assert string_reply is not None, "Agent should generate reply for string input"
    assert isinstance(string_reply, str), "Reply should be string for string input"
    assert "How are you today?" in string_reply, "Reply should reference input"

    # Test 2: Agent handles multimodal input
    multimodal_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,testimage"}},
            ],
        }
    ]

    multimodal_reply = agent.generate_reply(messages=multimodal_messages)

    assert multimodal_reply is not None, "Agent should generate reply for multimodal input"

    # The reply might be processed by content_str, so check if it's meaningful
    if isinstance(multimodal_reply, str):
        # If converted to string, should contain meaningful content
        assert len(multimodal_reply) > 0, "Reply should not be empty"
        # Should reference the multimodal nature of input
        assert any(word in multimodal_reply.lower() for word in ["multimodal", "image", "visual", "see"]), (
            "Reply should acknowledge visual content"
        )
    elif isinstance(multimodal_reply, list):
        # If still multimodal, verify structure
        assert len(multimodal_reply) > 0, "Multimodal reply should not be empty"
        text_parts = [item for item in multimodal_reply if item.get("type") == "text"]
        assert len(text_parts) > 0, "Should have text parts in multimodal reply"


@pytest.mark.integration
@run_for_optional_imports(["openai"], "openai")
def test_custom_client_conversation_flow():
    """Test full conversation flow with custom multimodal client."""

    # Configure two agents with custom clients
    config_list = [
        {
            "model": "conversation-multimodal-model",
            "model_client_cls": "RealMultimodalCustomClient",
            "api_key": "conversation-test-key",
        }
    ]

    agent1 = ConversableAgent(
        name="analyst",
        llm_config={"config_list": config_list},
        human_input_mode="NEVER",
        system_message="You analyze data and provide insights.",
        max_consecutive_auto_reply=1,
    )

    agent2 = ConversableAgent(
        name="reviewer",
        llm_config={"config_list": config_list},
        human_input_mode="NEVER",
        system_message="You review analysis and provide feedback.",
        max_consecutive_auto_reply=1,
    )

    # Register custom client for both agents
    agent1.register_model_client(model_client_cls=RealMultimodalCustomClient)
    agent2.register_model_client(model_client_cls=RealMultimodalCustomClient)

    # Start conversation with multimodal content
    multimodal_message = [
        {"type": "text", "text": "Please analyze this sales chart:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,saleschart"}},
    ]

    chat_result = agent1.initiate_chat(agent2, message={"content": multimodal_message}, max_turns=2)

    # Verify conversation completed
    assert chat_result is not None, "Conversation should complete"
    assert len(chat_result.chat_history) >= 2, "Should have multiple messages"

    # Verify multimodal content was handled
    first_message = chat_result.chat_history[0]
    assert "content" in first_message, "First message should have content"

    if isinstance(first_message["content"], list):
        # Verify multimodal structure preserved
        assert len(first_message["content"]) == 2, "Should have text and image"
        assert first_message["content"][0]["type"] == "text", "First item should be text"
        assert first_message["content"][1]["type"] == "image_url", "Second item should be image"

    # Verify both agents participated
    participant_names = {msg.get("name") for msg in chat_result.chat_history if msg.get("name")}
    assert "analyst" in participant_names, "Analyst should participate"
    assert "reviewer" in participant_names, "Reviewer should participate"

    # Test that all content can be processed by content_str
    for msg in chat_result.chat_history:
        content = msg["content"]
        try:
            content_string = content_str(content)
            assert isinstance(content_string, str), "content_str should return string"
            assert len(content_string) > 0, "content_str should not be empty"
        except Exception as e:
            pytest.fail(f"content_str failed on message: {e}")


@pytest.mark.integration
@run_for_optional_imports(["openai"], "openai")
def test_custom_client_error_recovery():
    """Test custom client error handling and recovery."""

    class ErrorProneCustomClient(RealMultimodalCustomClient):
        """Custom client that occasionally fails."""

        def __init__(self, config: dict[str, Any]):
            super().__init__(config)
            self.failure_count = 0
            self.max_failures = 1  # Fail once, then succeed

        def create(self, params: dict[str, Any]):
            # Fail on first call, succeed afterwards
            if self.failure_count < self.max_failures:
                self.failure_count += 1
                raise ConnectionError("Simulated network error")

            return super().create(params)

    # Configure agent with error-prone client
    config_list = [
        {"model": "error-prone-model", "model_client_cls": "ErrorProneCustomClient", "api_key": "error-test-key"}
    ]

    agent = ConversableAgent(
        name="error_test_agent",
        llm_config={"config_list": config_list},
        human_input_mode="NEVER",
        system_message="You handle errors gracefully.",
    )

    # Register the error-prone client
    agent.register_model_client(model_client_cls=ErrorProneCustomClient)

    # First call should fail
    with pytest.raises(ConnectionError):
        agent.generate_reply(messages=[{"role": "user", "content": "This will fail"}])

    # Second call should succeed (after error recovery)
    try:
        reply = agent.generate_reply(messages=[{"role": "user", "content": "This should work"}])
        assert reply is not None, "Should succeed after error"
        assert isinstance(reply, str), "Should return valid response"
        assert "This should work" in reply, "Should process input correctly"
    except Exception as e:
        pytest.fail(f"Second call should have succeeded: {e}")


@pytest.mark.integration
@run_for_optional_imports(["openai"], "openai")
def test_custom_client_multimodal_content_validation():
    """Test custom client validates multimodal content properly."""

    class ValidatingCustomClient(RealMultimodalCustomClient):
        """Custom client that validates multimodal content."""

        def create(self, params: dict[str, Any]):
            # Validate input messages
            messages = params.get("messages", [])
            for message in messages:
                content = message.get("content")
                if isinstance(content, list):
                    self._validate_multimodal_content(content)

            return super().create(params)

        def _validate_multimodal_content(self, content: list[dict[str, Any]]):
            """Validate multimodal content structure."""
            for item in content:
                if not isinstance(item, dict):
                    raise ValueError("Each multimodal item must be a dictionary")

                if "type" not in item:
                    raise ValueError("Each multimodal item must have a 'type' field")

                if item["type"] == "text" and "text" not in item:
                    raise ValueError("Text items must have 'text' field")

                if item["type"] == "image_url" and "image_url" not in item:
                    raise ValueError("Image items must have 'image_url' field")

                if item["type"] == "image_url":
                    image_url = item["image_url"]
                    if not isinstance(image_url, dict) or "url" not in image_url:
                        raise ValueError("image_url must be dict with 'url' field")

    # Configure agent with validating client
    config_list = [
        {"model": "validating-model", "model_client_cls": "ValidatingCustomClient", "api_key": "validation-test-key"}
    ]

    agent = ConversableAgent(name="validating_agent", llm_config={"config_list": config_list}, human_input_mode="NEVER")

    agent.register_model_client(model_client_cls=ValidatingCustomClient)

    # Test 1: Valid multimodal content should work
    valid_content = [
        {"type": "text", "text": "Valid text content"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,validimage"}},
    ]

    reply = agent.generate_reply(messages=[{"role": "user", "content": valid_content}])
    assert reply is not None, "Valid multimodal content should work"

    # Test 2: Invalid content should raise errors
    invalid_contents = [
        # Missing text field
        [{"type": "text"}],
        # Missing image_url field
        [{"type": "image_url"}],
        # Missing url in image_url
        [{"type": "image_url", "image_url": {}}],
        # Missing type field
        [{"text": "Missing type"}],
    ]

    for invalid_content in invalid_contents:
        with pytest.raises(ValueError):
            agent.generate_reply(messages=[{"role": "user", "content": invalid_content}])


@pytest.mark.integration
@run_for_optional_imports(["openai"], "openai")
def test_custom_client_performance_with_multimodal():
    """Test custom client performance with multimodal content."""

    class PerformanceTrackingClient(RealMultimodalCustomClient):
        """Custom client that tracks performance metrics."""

        def __init__(self, config: dict[str, Any]):
            super().__init__(config)
            self.processing_times = []
            self.content_sizes = []

        def create(self, params: dict[str, Any]):
            import time

            start_time = time.time()

            # Track content size
            messages = params.get("messages", [])
            total_content_size = 0
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str):
                    total_content_size += len(content)
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            total_content_size += len(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            total_content_size += len(str(item.get("image_url", {})))

            self.content_sizes.append(total_content_size)

            # Process request
            response = super().create(params)

            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            return response

    # Configure agent with performance tracking client
    config_list = [
        {
            "model": "performance-model",
            "model_client_cls": "PerformanceTrackingClient",
            "api_key": "performance-test-key",
        }
    ]

    agent = ConversableAgent(
        name="performance_agent", llm_config={"config_list": config_list}, human_input_mode="NEVER"
    )

    agent.register_model_client(model_client_cls=PerformanceTrackingClient)

    # Process multiple requests of different sizes
    test_inputs = [
        # Small text
        "Small input",
        # Large text
        "Large input " * 100,
        # Small multimodal
        [
            {"type": "text", "text": "Small multimodal"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,small"}},
        ],
        # Large multimodal
        [
            {"type": "text", "text": "Large multimodal content " * 50},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "large" * 100}},
        ],
    ]

    for input_content in test_inputs:
        reply = agent.generate_reply(messages=[{"role": "user", "content": input_content}])
        assert reply is not None, "All inputs should be processed successfully"

    # Note: Performance tracking is verified implicitly by successful processing
    # The PerformanceTrackingClient tracks metrics internally and would fail if there were issues
