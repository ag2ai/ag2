# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for content block types and ContentParser."""

import warnings

import pytest

from autogen.llm_clients.models import (
    AudioContent,
    BaseContent,
    CitationContent,
    ContentParser,
    GenericContent,
    ImageContent,
    ReasoningContent,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    ToolResultContent,
    VideoContent,
)


class TestTextContent:
    """Test TextContent block."""

    def test_create_text_content(self):
        """Test creating a text content block."""
        content = TextContent(type="text", text="Hello world")
        assert content.type == "text"
        assert content.text == "Hello world"

    def test_text_content_with_extra_fields(self):
        """Test that extra fields are stored."""
        content = TextContent(type="text", text="Test", extra={"custom": "data"})
        assert content.extra == {"custom": "data"}


class TestImageContent:
    """Test ImageContent block."""

    def test_create_image_content(self):
        """Test creating an image content block."""
        content = ImageContent(type="image", image_url="https://example.com/image.jpg")
        assert content.type == "image"
        assert content.image_url == "https://example.com/image.jpg"
        assert content.detail is None

    def test_image_content_with_detail(self):
        """Test creating an image with detail level."""
        content = ImageContent(type="image", image_url="https://example.com/image.jpg", detail="high")
        assert content.detail == "high"


class TestAudioContent:
    """Test AudioContent block."""

    def test_create_audio_content(self):
        """Test creating an audio content block."""
        content = AudioContent(type="audio", audio_url="https://example.com/audio.mp3")
        assert content.type == "audio"
        assert content.audio_url == "https://example.com/audio.mp3"
        assert content.transcript is None

    def test_audio_content_with_transcript(self):
        """Test creating audio with transcript."""
        content = AudioContent(type="audio", audio_url="https://example.com/audio.mp3", transcript="Hello world")
        assert content.transcript == "Hello world"


class TestVideoContent:
    """Test VideoContent block."""

    def test_create_video_content(self):
        """Test creating a video content block."""
        content = VideoContent(type="video", video_url="https://example.com/video.mp4")
        assert content.type == "video"
        assert content.video_url == "https://example.com/video.mp4"


class TestReasoningContent:
    """Test ReasoningContent block."""

    def test_create_reasoning_content(self):
        """Test creating a reasoning content block."""
        content = ReasoningContent(type="reasoning", reasoning="Let me think through this...")
        assert content.type == "reasoning"
        assert content.reasoning == "Let me think through this..."
        assert content.summary is None

    def test_reasoning_content_with_summary(self):
        """Test creating reasoning with summary."""
        content = ReasoningContent(
            type="reasoning", reasoning="Step 1: ... Step 2: ...", summary="Analyzed systematically"
        )
        assert content.summary == "Analyzed systematically"


class TestThinkingContent:
    """Test ThinkingContent block."""

    def test_create_thinking_content(self):
        """Test creating a thinking content block."""
        content = ThinkingContent(type="thinking", thinking="Hmm, interesting question...")
        assert content.type == "thinking"
        assert content.thinking == "Hmm, interesting question..."


class TestCitationContent:
    """Test CitationContent block."""

    def test_create_citation_content(self):
        """Test creating a citation content block."""
        content = CitationContent(
            type="citation",
            url="https://example.com",
            title="Example Article",
            snippet="This is a snippet...",
        )
        assert content.type == "citation"
        assert content.url == "https://example.com"
        assert content.title == "Example Article"
        assert content.snippet == "This is a snippet..."
        assert content.relevance_score is None

    def test_citation_content_with_relevance(self):
        """Test creating citation with relevance score."""
        content = CitationContent(
            type="citation",
            url="https://example.com",
            title="Example",
            snippet="Snippet",
            relevance_score=0.87,
        )
        assert content.relevance_score == 0.87


class TestToolCallContent:
    """Test ToolCallContent block."""

    def test_create_tool_call_content(self):
        """Test creating a tool call content block."""
        content = ToolCallContent(type="tool_call", id="call-123", name="get_weather", arguments='{"city": "SF"}')
        assert content.type == "tool_call"
        assert content.id == "call-123"
        assert content.name == "get_weather"
        assert content.arguments == '{"city": "SF"}'


class TestToolResultContent:
    """Test ToolResultContent block."""

    def test_create_tool_result_content(self):
        """Test creating a tool result content block."""
        content = ToolResultContent(type="tool_result", tool_call_id="call-123", output="Sunny, 72°F")
        assert content.type == "tool_result"
        assert content.tool_call_id == "call-123"
        assert content.output == "Sunny, 72°F"


class TestGenericContent:
    """Test GenericContent for forward compatibility."""

    def test_create_generic_content(self):
        """Test creating generic content with unknown type."""
        content = GenericContent(type="reflection", reflection="Upon reviewing...", confidence=0.87)
        assert content.type == "reflection"
        assert content.data["reflection"] == "Upon reviewing..."
        assert content.data["confidence"] == 0.87

    def test_generic_content_attribute_access(self):
        """Test attribute-style access to generic content fields."""
        content = GenericContent(
            type="video_analysis", analysis="Scene contains a dog", timestamp="00:01:23", confidence=0.95
        )
        assert content.analysis == "Scene contains a dog"
        assert content.timestamp == "00:01:23"
        assert content.confidence == 0.95

    def test_generic_content_get_method(self):
        """Test dict-style get method."""
        content = GenericContent(type="custom", field1="value1")
        assert content.get("field1") == "value1"
        assert content.get("missing", "default") == "default"

    def test_generic_content_missing_attribute(self):
        """Test accessing non-existent attribute raises AttributeError."""
        content = GenericContent(type="custom", field1="value1")
        with pytest.raises(AttributeError, match="custom.*has no attribute.*missing"):
            _ = content.missing

    def test_generic_content_preserves_all_fields(self):
        """Test that all custom fields are preserved."""
        data = {
            "type": "complex_type",
            "field1": "value1",
            "field2": 42,
            "field3": [1, 2, 3],
            "field4": {"nested": "dict"},
        }
        content = GenericContent(**data)
        assert content.type == "complex_type"
        assert content.field1 == "value1"
        assert content.field2 == 42
        assert content.field3 == [1, 2, 3]
        assert content.field4 == {"nested": "dict"}


class TestContentParser:
    """Test ContentParser registry and parsing."""

    def test_parse_known_text_type(self):
        """Test parsing known text type."""
        data = {"type": "text", "text": "Hello world"}
        content = ContentParser.parse(data)
        assert isinstance(content, TextContent)
        assert content.text == "Hello world"

    def test_parse_known_reasoning_type(self):
        """Test parsing known reasoning type."""
        data = {"type": "reasoning", "reasoning": "Let me think...", "summary": "Analyzed"}
        content = ContentParser.parse(data)
        assert isinstance(content, ReasoningContent)
        assert content.reasoning == "Let me think..."
        assert content.summary == "Analyzed"

    def test_parse_known_thinking_type(self):
        """Test parsing known thinking type."""
        data = {"type": "thinking", "thinking": "Hmm..."}
        content = ContentParser.parse(data)
        assert isinstance(content, ThinkingContent)
        assert content.thinking == "Hmm..."

    def test_parse_known_citation_type(self):
        """Test parsing known citation type."""
        data = {"type": "citation", "url": "https://example.com", "title": "Title", "snippet": "Snippet"}
        content = ContentParser.parse(data)
        assert isinstance(content, CitationContent)
        assert content.url == "https://example.com"

    def test_parse_known_tool_call_type(self):
        """Test parsing known tool call type."""
        data = {"type": "tool_call", "id": "call-123", "name": "func", "arguments": "{}"}
        content = ContentParser.parse(data)
        assert isinstance(content, ToolCallContent)
        assert content.id == "call-123"

    def test_parse_unknown_type(self):
        """Test parsing unknown type falls back to GenericContent."""
        data = {"type": "unknown_type", "custom_field": "value"}
        content = ContentParser.parse(data)
        assert isinstance(content, GenericContent)
        assert content.type == "unknown_type"
        assert content.custom_field == "value"

    def test_parse_invalid_known_type_falls_back(self):
        """Test that invalid data for known type falls back to GenericContent."""
        # Missing required 'text' field for TextContent
        data = {"type": "text"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            content = ContentParser.parse(data)
            assert len(w) == 1
            assert "Failed to parse" in str(w[0].message)
            assert isinstance(content, GenericContent)

    def test_register_custom_type(self):
        """Test registering a custom content type."""

        class CustomContent(BaseContent):
            type: str = "custom"
            custom_field: str

        ContentParser.register("custom", CustomContent)

        data = {"type": "custom", "custom_field": "value"}
        content = ContentParser.parse(data)
        assert isinstance(content, CustomContent)
        assert content.custom_field == "value"

    def test_parse_missing_type_field(self):
        """Test parsing data without type field defaults to 'unknown'."""
        data = {"custom_field": "value"}
        content = ContentParser.parse(data)
        assert isinstance(content, GenericContent)
        assert content.type == "unknown"

    def test_parse_all_known_types(self):
        """Test parsing all registered known types."""
        test_cases = [
            ({"type": "text", "text": "test"}, TextContent),
            ({"type": "image", "image_url": "url"}, ImageContent),
            ({"type": "audio", "audio_url": "url"}, AudioContent),
            ({"type": "video", "video_url": "url"}, VideoContent),
            ({"type": "reasoning", "reasoning": "test"}, ReasoningContent),
            ({"type": "thinking", "thinking": "test"}, ThinkingContent),
            (
                {"type": "citation", "url": "url", "title": "title", "snippet": "snippet"},
                CitationContent,
            ),
            ({"type": "tool_call", "id": "id", "name": "name", "arguments": "{}"}, ToolCallContent),
            ({"type": "tool_result", "tool_call_id": "id", "output": "output"}, ToolResultContent),
        ]

        for data, expected_type in test_cases:
            content = ContentParser.parse(data)
            assert isinstance(content, expected_type), f"Failed for type {data['type']}"


class TestContentBlockInteroperability:
    """Test interoperability between different content block types."""

    def test_all_content_blocks_are_base_content(self):
        """Test that all content block types inherit from BaseContent."""
        content_types = [
            TextContent(type="text", text="test"),
            ImageContent(type="image", image_url="url"),
            AudioContent(type="audio", audio_url="url"),
            VideoContent(type="video", video_url="url"),
            ReasoningContent(type="reasoning", reasoning="test"),
            ThinkingContent(type="thinking", thinking="test"),
            CitationContent(type="citation", url="url", title="title", snippet="snippet"),
            ToolCallContent(type="tool_call", id="id", name="name", arguments="{}"),
            ToolResultContent(type="tool_result", tool_call_id="id", output="output"),
            GenericContent(type="custom", field="value"),
        ]

        for content in content_types:
            assert isinstance(content, BaseContent)

    def test_content_blocks_have_type_field(self):
        """Test that all content blocks have a type field."""
        content_types = [
            TextContent(type="text", text="test"),
            GenericContent(type="custom", field="value"),
        ]

        for content in content_types:
            assert hasattr(content, "type")
            assert isinstance(content.type, str)

    def test_content_blocks_have_extra_field(self):
        """Test that all content blocks have an extra field for unknown data."""
        content = TextContent(type="text", text="test", extra={"provider_specific": "data"})
        assert content.extra == {"provider_specific": "data"}
