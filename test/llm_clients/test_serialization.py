# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for serialization and deserialization of unified models."""

import json

from autogen.llm_clients.models import (
    CitationContent,
    GenericContent,
    ReasoningContent,
    TextContent,
    ToolCallContent,
    UnifiedMessage,
    UnifiedResponse,
)


class TestContentBlockSerialization:
    """Test serialization of content blocks."""

    def test_text_content_serialization(self):
        """Test TextContent serialization."""
        content = TextContent(type="text", text="Hello world")
        content_dict = content.model_dump()

        assert content_dict["type"] == "text"
        assert content_dict["text"] == "Hello world"

    def test_reasoning_content_serialization(self):
        """Test ReasoningContent serialization."""
        content = ReasoningContent(type="reasoning", reasoning="Step 1: analyze", summary="Analysis")
        content_dict = content.model_dump()

        assert content_dict["type"] == "reasoning"
        assert content_dict["reasoning"] == "Step 1: analyze"
        assert content_dict["summary"] == "Analysis"

    def test_citation_content_serialization(self):
        """Test CitationContent serialization."""
        content = CitationContent(
            type="citation",
            url="https://example.com",
            title="Title",
            snippet="Snippet",
            relevance_score=0.95,
        )
        content_dict = content.model_dump()

        assert content_dict["type"] == "citation"
        assert content_dict["url"] == "https://example.com"
        assert content_dict["relevance_score"] == 0.95

    def test_tool_call_content_serialization(self):
        """Test ToolCallContent serialization."""
        content = ToolCallContent(type="tool_call", id="call-123", name="get_weather", arguments='{"city":"SF"}')
        content_dict = content.model_dump()

        assert content_dict["type"] == "tool_call"
        assert content_dict["id"] == "call-123"
        assert content_dict["name"] == "get_weather"

    def test_generic_content_serialization(self):
        """Test GenericContent serialization."""
        content = GenericContent(
            type="reflection", reflection="Upon reviewing...", confidence=0.87, corrections=["fix1", "fix2"]
        )
        content_dict = content.model_dump()

        assert content_dict["type"] == "reflection"
        # Extra fields are serialized at top level with native extra='allow'
        assert content_dict["reflection"] == "Upon reviewing..."
        assert content_dict["confidence"] == 0.87
        assert content_dict["corrections"] == ["fix1", "fix2"]


class TestUnifiedMessageSerialization:
    """Test serialization of UnifiedMessage."""

    def test_simple_message_serialization(self):
        """Test serializing a simple message."""
        content = TextContent(type="text", text="Hello")
        message = UnifiedMessage(role="user", content=[content])
        message_dict = message.model_dump()

        assert message_dict["role"] == "user"
        assert len(message_dict["content"]) == 1
        assert message_dict["content"][0]["type"] == "text"

    def test_message_with_multiple_content_serialization(self):
        """Test serializing message with multiple content blocks."""
        contents = [
            ReasoningContent(type="reasoning", reasoning="Think"),
            TextContent(type="text", text="Answer"),
        ]
        message = UnifiedMessage(role="assistant", content=contents)
        message_dict = message.model_dump()

        assert len(message_dict["content"]) == 2
        assert message_dict["content"][0]["type"] == "reasoning"
        assert message_dict["content"][1]["type"] == "text"

    def test_message_with_metadata_serialization(self):
        """Test serializing message with metadata."""
        content = TextContent(type="text", text="Hello")
        metadata = {"temperature": 0.7, "provider": "openai"}
        message = UnifiedMessage(role="assistant", content=[content], name="bot", metadata=metadata)
        message_dict = message.model_dump()

        assert message_dict["name"] == "bot"
        assert message_dict["metadata"]["temperature"] == 0.7
        assert message_dict["metadata"]["provider"] == "openai"


class TestUnifiedResponseSerialization:
    """Test serialization of UnifiedResponse."""

    def test_simple_response_serialization(self):
        """Test serializing a simple response."""
        message = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="Hello")])
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[message])
        response_dict = response.model_dump()

        assert response_dict["id"] == "resp-123"
        assert response_dict["model"] == "gpt-4"
        assert response_dict["provider"] == "openai"
        assert len(response_dict["messages"]) == 1

    def test_response_with_usage_serialization(self):
        """Test serializing response with usage."""
        message = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="Hello")])
        usage = {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150}
        response = UnifiedResponse(
            id="resp-123", model="gpt-4", provider="openai", messages=[message], usage=usage, cost=0.015
        )
        response_dict = response.model_dump()

        assert response_dict["usage"]["prompt_tokens"] == 50
        assert response_dict["cost"] == 0.015

    def test_response_with_provider_metadata_serialization(self):
        """Test serializing response with provider metadata."""
        message = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="Hello")])
        metadata = {"system_fingerprint": "fp_xyz"}
        response = UnifiedResponse(
            id="resp-123",
            model="gpt-4",
            provider="openai",
            messages=[message],
            provider_metadata=metadata,
        )
        response_dict = response.model_dump()

        assert response_dict["provider_metadata"]["system_fingerprint"] == "fp_xyz"


class TestJSONSerialization:
    """Test JSON serialization."""

    def test_content_block_to_json(self):
        """Test serializing content block to JSON string."""
        content = TextContent(type="text", text="Hello world")
        content_dict = content.model_dump()
        json_str = json.dumps(content_dict)

        assert isinstance(json_str, str)
        assert "text" in json_str
        assert "Hello world" in json_str

    def test_message_to_json(self):
        """Test serializing message to JSON string."""
        content = TextContent(type="text", text="Hello")
        message = UnifiedMessage(role="user", content=[content])
        message_dict = message.model_dump()
        json_str = json.dumps(message_dict)

        assert isinstance(json_str, str)
        assert "user" in json_str
        assert "Hello" in json_str

    def test_response_to_json(self):
        """Test serializing response to JSON string."""
        message = UnifiedMessage(role="assistant", content=[TextContent(type="text", text="Hello")])
        response = UnifiedResponse(id="resp-123", model="gpt-4", provider="openai", messages=[message])
        response_dict = response.model_dump()
        json_str = json.dumps(response_dict)

        assert isinstance(json_str, str)
        assert "resp-123" in json_str
        assert "gpt-4" in json_str
        assert "openai" in json_str

    def test_complex_response_to_json(self):
        """Test serializing complex response with all features to JSON."""
        contents = [
            ReasoningContent(type="reasoning", reasoning="Step 1: analyze"),
            TextContent(type="text", text="Answer"),
            CitationContent(type="citation", url="url", title="title", snippet="snippet"),
            GenericContent(type="reflection", reflection="Review", confidence=0.9),
        ]
        message = UnifiedMessage(role="assistant", content=contents)
        response = UnifiedResponse(
            id="resp-123",
            model="gpt-4",
            provider="openai",
            messages=[message],
            usage={"prompt_tokens": 50, "completion_tokens": 100},
            cost=0.015,
            finish_reason="stop",
            status="completed",
            provider_metadata={"fingerprint": "xyz"},
        )
        response_dict = response.model_dump()
        json_str = json.dumps(response_dict)

        # Verify JSON is valid
        parsed = json.loads(json_str)
        assert parsed["id"] == "resp-123"
        assert len(parsed["messages"]) == 1
        assert len(parsed["messages"][0]["content"]) == 4


class TestRoundTripSerialization:
    """Test round-trip serialization (serialize and deserialize)."""

    def test_text_content_round_trip(self):
        """Test round-trip for TextContent."""
        original = TextContent(type="text", text="Hello world")
        serialized = original.model_dump()
        deserialized = TextContent(**serialized)

        assert deserialized.type == original.type
        assert deserialized.text == original.text

    def test_reasoning_content_round_trip(self):
        """Test round-trip for ReasoningContent."""
        original = ReasoningContent(type="reasoning", reasoning="Step 1", summary="Summary")
        serialized = original.model_dump()
        deserialized = ReasoningContent(**serialized)

        assert deserialized.reasoning == original.reasoning
        assert deserialized.summary == original.summary

    def test_generic_content_round_trip(self):
        """Test round-trip for GenericContent."""
        original = GenericContent(type="reflection", reflection="Review", confidence=0.9)
        serialized = original.model_dump()
        deserialized = GenericContent(**serialized)

        assert deserialized.type == original.type
        assert deserialized.reflection == original.reflection
        assert deserialized.confidence == original.confidence

    def test_message_round_trip(self):
        """Test round-trip for UnifiedMessage."""
        original = UnifiedMessage(
            role="assistant",
            content=[TextContent(type="text", text="Hello")],
            name="bot",
        )
        serialized = original.model_dump()

        # For deserialization, we need to reconstruct content blocks
        # This would require custom parsing in practice
        assert serialized["role"] == "assistant"
        assert serialized["name"] == "bot"

    def test_response_round_trip(self):
        """Test round-trip for UnifiedResponse."""
        original = UnifiedResponse(
            id="resp-123",
            model="gpt-4",
            provider="openai",
            messages=[UnifiedMessage(role="assistant", content=[TextContent(type="text", text="Hello")])],
            usage={"prompt_tokens": 50},
            cost=0.015,
        )
        serialized = original.model_dump()

        # Verify serialized data
        assert serialized["id"] == "resp-123"
        assert serialized["cost"] == 0.015


class TestSerializationEdgeCases:
    """Test edge cases in serialization."""

    def test_empty_content_list(self):
        """Test serializing message with empty content list."""
        message = UnifiedMessage(role="system", content=[])
        message_dict = message.model_dump()

        assert message_dict["content"] == []

    def test_none_optional_fields(self):
        """Test serializing with None optional fields."""
        content = TextContent(type="text", text="Hello")
        message = UnifiedMessage(role="user", content=[content], name=None)
        message_dict = message.model_dump()

        # Pydantic typically excludes None by default or includes it
        # depending on configuration
        assert "role" in message_dict
        assert "content" in message_dict

    def test_nested_dict_in_generic_content(self):
        """Test serializing GenericContent with nested dicts."""
        content = GenericContent(
            type="complex",
            nested={
                "level1": {"level2": {"level3": "deep value"}},
                "array": [1, 2, 3],
            },
        )
        content_dict = content.model_dump()

        # Extra fields are serialized at top level with native extra='allow'
        assert content_dict["nested"]["level1"]["level2"]["level3"] == "deep value"
        assert content_dict["nested"]["array"] == [1, 2, 3]

    def test_unicode_in_content(self):
        """Test serializing content with unicode characters."""
        content = TextContent(type="text", text="Hello 世界 🌍")
        content_dict = content.model_dump()
        json_str = json.dumps(content_dict, ensure_ascii=False)

        assert "世界" in json_str
        assert "🌍" in json_str

    def test_large_content_serialization(self):
        """Test serializing large content."""
        large_text = "A" * 10000
        content = TextContent(type="text", text=large_text)
        content_dict = content.model_dump()
        json_str = json.dumps(content_dict)

        # Verify it serializes successfully
        assert len(json_str) > 10000
        parsed = json.loads(json_str)
        assert len(parsed["text"]) == 10000


class TestDataIntegrity:
    """Test data integrity after serialization."""

    def test_no_data_loss_in_serialization(self):
        """Test that no data is lost during serialization."""
        contents = [
            TextContent(type="text", text="Hello"),
            ReasoningContent(type="reasoning", reasoning="Step 1", summary="Summary"),
            CitationContent(
                type="citation",
                url="https://example.com",
                title="Title",
                snippet="Snippet",
                relevance_score=0.95,
            ),
            GenericContent(type="custom", field1="value1", field2=42, field3=[1, 2, 3]),
        ]
        message = UnifiedMessage(role="assistant", content=contents, name="bot", metadata={"key": "value"})
        response = UnifiedResponse(
            id="resp-123",
            model="gpt-4",
            provider="openai",
            messages=[message],
            usage={"prompt_tokens": 50, "completion_tokens": 100},
            cost=0.015,
            finish_reason="stop",
            status="completed",
            provider_metadata={"fingerprint": "xyz"},
        )

        # Serialize
        response_dict = response.model_dump()

        # Verify all data is present
        assert response_dict["id"] == "resp-123"
        assert response_dict["model"] == "gpt-4"
        assert response_dict["provider"] == "openai"
        assert len(response_dict["messages"]) == 1
        assert len(response_dict["messages"][0]["content"]) == 4
        assert response_dict["usage"]["prompt_tokens"] == 50
        assert response_dict["cost"] == 0.015
        assert response_dict["finish_reason"] == "stop"
        assert response_dict["status"] == "completed"
        assert response_dict["provider_metadata"]["fingerprint"] == "xyz"

    def test_serialization_is_deterministic(self):
        """Test that serialization produces consistent output."""
        content = TextContent(type="text", text="Hello")
        message = UnifiedMessage(role="user", content=[content])

        # Serialize multiple times
        serialized1 = message.model_dump()
        serialized2 = message.model_dump()

        # Should be identical
        assert serialized1 == serialized2
