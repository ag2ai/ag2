# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for UnifiedResponse."""

from autogen.llm_clients.models import (
    CitationContent,
    GenericContent,
    ReasoningContent,
    TextContent,
    UnifiedMessage,
    UnifiedResponse,
)


class TestUnifiedResponseCreation:
    """Test creating UnifiedResponse instances."""

    def test_create_simple_response(self):
        """Test creating a simple response."""
        message = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="Hello")])
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[message])

        assert response.id == "resp-123"
        assert response.model == "gpt-4"
        assert response.provider == "openai"
        assert len(response.messages) == 1
        assert response.usage == {}
        assert response.cost is None
        assert response.finish_reason is None
        assert response.status is None

    def test_create_response_with_usage(self):
        """Test creating a response with usage information."""
        message = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="Hello")])
        usage = {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150}
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[message], usage=usage)

        assert response.usage == usage
        assert response.usage["prompt_tokens"] == 50
        assert response.usage["completion_tokens"] == 100

    def test_create_response_with_cost(self):
        """Test creating a response with cost."""
        message = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="Hello")])
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[message], cost=0.015)

        assert response.cost == 0.015

    def test_create_response_with_provider_metadata(self):
        """Test creating a response with provider-specific metadata."""
        message = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="Hello")])
        metadata = {"system_fingerprint": "fp_xyz", "service_tier": "default"}
        response = UnifiedResponse(
            id="resp-123", model="gpt-4", provider="openai", messages=[message], provider_metadata=metadata
        )

        assert response.provider_metadata == metadata

    def test_create_response_with_finish_reason(self):
        """Test creating a response with finish reason."""
        message = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="Hello")])
        response = UnifiedResponse(
            id="resp-123", model="gpt-4", provider="openai", messages=[message], finish_reason="stop"
        )

        assert response.finish_reason == "stop"

    def test_create_response_with_status(self):
        """Test creating a response with status."""
        message = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="Hello")])
        response = UnifiedResponse(
            id="resp-123", model="gpt-4", provider="openai", messages=[message], status="completed"
        )

        assert response.status == "completed"

    def test_create_response_with_custom_status(self):
        """Test creating response with custom/future status (extensibility)."""
        message = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="Hello")])
        # Future providers might introduce new status values
        custom_statuses = ["streaming", "rate_limited", "queued", "processing"]
        for status in custom_statuses:
            response = UnifiedResponse(
                id="resp-123", model="gpt-4", provider="openai", messages=[message], status=status
            )
            assert response.status == status

    def test_standard_statuses_constant(self):
        """Test that STANDARD_STATUSES constant is defined."""
        assert hasattr(UnifiedResponse, "STANDARD_STATUSES")
        assert "completed" in UnifiedResponse.STANDARD_STATUSES
        assert "in_progress" in UnifiedResponse.STANDARD_STATUSES
        assert "failed" in UnifiedResponse.STANDARD_STATUSES

    def test_is_standard_status_method(self):
        """Test is_standard_status() method."""
        message = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="test")])

        # Standard statuses
        for status in ["completed", "in_progress", "failed"]:
            response = UnifiedResponse(
                id="resp-123", model="gpt-4", provider="openai", messages=[message], status=status
            )
            assert response.is_standard_status() is True

        # Custom statuses
        for status in ["streaming", "queued", "custom"]:
            response = UnifiedResponse(
                id="resp-123", model="gpt-4", provider="openai", messages=[message], status=status
            )
            assert response.is_standard_status() is False

        # None status
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[message], status=None)
        assert response.is_standard_status() is False

    def test_create_response_different_providers(self):
        """Test creating responses for different providers."""
        providers = ["openai", "anthropic", "gemini", "bedrock"]
        for provider in providers:
            message = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="Hello")])
            response = UnifiedResponse(id="resp-123", model="model", provider=provider, messages=[message])
            assert response.provider == provider


class TestUnifiedResponseTextProperty:
    """Test the text property."""

    def test_text_property_single_message(self):
        """Test text property with single message."""
        message = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="The answer is 42")])
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[message])

        assert response.text == "The answer is 42"

    def test_text_property_multiple_content_blocks(self):
        """Test text property with multiple content blocks in first message."""
        contents = [
            ReasoningContent(type="reasoning", reasoning="Let me think..."),
            TextContent(type="text", text="The answer is 42"),
        ]
        message = UnifiedMessage(role="assistant", content=contents)
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[message])

        assert response.text == "Let me think... The answer is 42"

    def test_text_property_multiple_messages(self):
        """Test text property with multiple messages (returns text from all messages)."""
        message1 = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="First message")])
        message2 = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="Second message")])
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[message1, message2])

        # Should return text from all messages joined with space
        assert response.text == "First message Second message"

    def test_text_property_empty_messages(self):
        """Test text property with no messages."""
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[])

        assert response.text == ""


class TestUnifiedResponseReasoningProperty:
    """Test the reasoning property."""

    def test_reasoning_property_single_block(self):
        """Test reasoning property with single reasoning block."""
        reasoning = ReasoningContent(type="reasoning", reasoning="Step 1: analyze")
        message = UnifiedMessage(role="assistant", content=[reasoning])
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[message])

        reasoning_blocks = response.reasoning
        assert len(reasoning_blocks) == 1
        assert reasoning_blocks[0].reasoning == "Step 1: analyze"

    def test_reasoning_property_multiple_blocks_single_message(self):
        """Test reasoning property with multiple reasoning blocks in one message."""
        contents = [
            ReasoningContent(type="reasoning", reasoning="Step 1"),
            TextContent(type="text", text="Interim"),
            ReasoningContent(type="reasoning", reasoning="Step 2"),
        ]
        message = UnifiedMessage(role="assistant", content=contents)
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[message])

        reasoning_blocks = response.reasoning
        assert len(reasoning_blocks) == 2

    def test_reasoning_property_multiple_messages(self):
        """Test reasoning property across multiple messages."""
        message1 = UnifiedMessage(
            role="assistant",
            content=[ReasoningContent(type="reasoning", reasoning="Reasoning 1")],
        )
        message2 = UnifiedMessage(
            role="assistant",
            content=[ReasoningContent(type="reasoning", reasoning="Reasoning 2")],
        )
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[message1, message2])

        reasoning_blocks = response.reasoning
        assert len(reasoning_blocks) == 2
        assert reasoning_blocks[0].reasoning == "Reasoning 1"
        assert reasoning_blocks[1].reasoning == "Reasoning 2"

    def test_reasoning_property_no_reasoning(self):
        """Test reasoning property when no reasoning blocks present."""
        message = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="No reasoning")])
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[message])

        assert len(response.reasoning) == 0


class TestUnifiedResponseContentByType:
    """Test get_content_by_type() method."""

    def test_get_content_by_type_across_messages(self):
        """Test getting content by type across multiple messages."""
        message1 = UnifiedMessage(
            role="assistant",
            content=[
                TextContent(type="text", text="Hello"),
                CitationContent(type="citation", url="url1", title="title1", snippet="snippet1"),
            ],
        )
        message2 = UnifiedMessage(
            role="assistant",
            content=[
                TextContent(type="text", text="World"),
                CitationContent(type="citation", url="url2", title="title2", snippet="snippet2"),
            ],
        )
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[message1, message2])

        citations = response.get_content_by_type("citation")
        assert len(citations) == 2
        assert all(c.type == "citation" for c in citations)

    def test_get_content_by_type_unknown_type(self):
        """Test getting unknown content type (GenericContent)."""
        message1 = UnifiedMessage(
            role="assistant",
            content=[GenericContent(type="reflection", reflection="Reflection 1")],
        )
        message2 = UnifiedMessage(
            role="assistant",
            content=[GenericContent(type="reflection", reflection="Reflection 2")],
        )
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[message1, message2])

        reflections = response.get_content_by_type("reflection")
        assert len(reflections) == 2
        assert all(isinstance(r, GenericContent) for r in reflections)

    def test_get_content_by_type_no_match(self):
        """Test get_content_by_type when no blocks match."""
        message = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="Hello")])
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[message])

        blocks = response.get_content_by_type("nonexistent")
        assert len(blocks) == 0


class TestUnifiedResponseSerialization:
    """Test response serialization."""

    def test_response_serialization(self):
        """Test that responses can be serialized to dict."""
        message = UnifiedMessage(
            role="assistant",
            content=[
                TextContent(type="text", text="Hello"),
                ReasoningContent(type="reasoning", reasoning="Think"),
            ],
        )
        response = UnifiedResponse(
            id="resp-123",
            model="gpt-4",
            provider="openai",
            messages=[message],
            usage={"prompt_tokens": 50, "completion_tokens": 100},
            cost=0.015,
            finish_reason="stop",
            status="completed",
        )

        response_dict = response.model_dump()
        assert response_dict["id"] == "resp-123"
        assert response_dict["model"] == "gpt-4"
        assert response_dict["provider"] == "openai"
        assert len(response_dict["messages"]) == 1
        assert response_dict["cost"] == 0.015

    def test_response_serialization_to_json(self):
        """Test that responses can be serialized to JSON."""
        import json

        message = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="Hello")])
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[message])

        response_dict = response.model_dump()
        json_str = json.dumps(response_dict)
        assert "resp-123" in json_str
        assert "gpt-4" in json_str


class TestUnifiedResponseComplexScenarios:
    """Test complex real-world scenarios."""

    def test_openai_o1_with_reasoning(self):
        """Test representing an OpenAI o1 response with reasoning blocks."""
        contents = [
            ReasoningContent(
                type="reasoning",
                reasoning="Let me break this down step by step:\n1. First, I'll analyze...",
                summary="Systematic analysis",
            ),
            TextContent(type="text", text="Based on the reasoning above, the answer is 42."),
        ]
        message = UnifiedMessage(role="assistant", content=contents)
        response = UnifiedResponse(
            id="chatcmpl-abc123",
            model="o1-preview",
            provider="openai",
            messages=[message],
            usage={"prompt_tokens": 100, "completion_tokens": 500, "total_tokens": 600},
            cost=0.05,
            finish_reason="stop",
            status="completed",
        )

        assert len(response.reasoning) == 1
        assert (
            response.text
            == "Let me break this down step by step:\n1. First, I'll analyze... Based on the reasoning above, the answer is 42."
        )
        assert response.reasoning[0].summary == "Systematic analysis"

    def test_anthropic_claude_with_reasoning(self):
        """Test representing an Anthropic Claude response with reasoning blocks."""
        contents = [
            ReasoningContent(type="reasoning", reasoning="Let me consider this carefully..."),
            TextContent(type="text", text="Here's my response."),
        ]
        message = UnifiedMessage(role="assistant", content=contents)
        response = UnifiedResponse(
            id="msg_abc123",
            model="claude-3-5-sonnet-20250101",
            provider="anthropic",
            messages=[message],
            usage={"input_tokens": 50, "output_tokens": 150},
            finish_reason="end_turn",
            status="completed",
        )

        assert len(response.reasoning) == 1
        assert response.reasoning[0].reasoning == "Let me consider this carefully..."

    def test_web_search_with_citations(self):
        """Test representing a response with web search citations."""
        contents = [
            CitationContent(
                type="citation",
                url="https://en.wikipedia.org/wiki/Artificial_intelligence",
                title="Artificial Intelligence - Wikipedia",
                snippet="AI is intelligence demonstrated by machines...",
                relevance_score=0.95,
            ),
            CitationContent(
                type="citation",
                url="https://example.com/ai-guide",
                title="Guide to AI",
                snippet="A comprehensive guide...",
                relevance_score=0.87,
            ),
            TextContent(type="text", text="Based on the sources above, AI is..."),
        ]
        message = UnifiedMessage(role="assistant", content=contents)
        response = UnifiedResponse(
            id="resp-search-123", model="gpt-4", provider="openai", messages=[message], status="completed"
        )

        citations = [block for msg in response.messages for block in msg.get_citations()]
        assert len(citations) == 2
        assert all(c.relevance_score is not None for c in citations)

    def test_future_unknown_content_type(self):
        """Test handling future unknown content types (forward compatibility)."""
        contents = [
            GenericContent(
                type="video_analysis",
                analysis="The video shows a dog playing in a park",
                timestamp="00:01:23",
                confidence=0.92,
                objects_detected=["dog", "park", "ball"],
            ),
            TextContent(type="text", text="Summary: Dog playing with ball in park."),
        ]
        message = UnifiedMessage(role="assistant", content=contents)
        response = UnifiedResponse(
            id="resp-video-123", model="gemini-pro-vision", provider="gemini", messages=[message]
        )

        video_blocks = response.get_content_by_type("video_analysis")
        assert len(video_blocks) == 1
        assert isinstance(video_blocks[0], GenericContent)
        assert video_blocks[0].analysis == "The video shows a dog playing in a park"
        assert video_blocks[0].confidence == 0.92

    def test_multi_message_conversation(self):
        """Test multi-turn conversation with mixed content types."""
        messages = [
            UnifiedMessage(role="user", content=[TextContent(type="text", text="What is 2+2?")]),
            UnifiedMessage(
                role="assistant",
                content=[
                    ReasoningContent(type="reasoning", reasoning="2 + 2 = 4"),
                    TextContent(type="text", text="The answer is 4"),
                ],
            ),
            UnifiedMessage(role="user", content=[TextContent(type="text", text="Are you sure?")]),
            UnifiedMessage(
                role="assistant",
                content=[
                    ReasoningContent(type="reasoning", reasoning="Yes, I'm certain"),
                    TextContent(type="text", text="Yes, I'm confident that 2+2=4"),
                ],
            ),
        ]
        response = UnifiedResponse(id="resp-multi-123", model="gpt-4", provider="openai", messages=messages)

        assert len(response.messages) == 4
        assert len(response.reasoning) == 2  # Two reasoning blocks now (both were converted from thinking)
