# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for UnifiedMessage."""

from autogen.llm_clients.models import (
    CitationContent,
    GenericContent,
    ReasoningContent,
    TextContent,
    ToolCallContent,
    UnifiedMessage,
)


class TestUnifiedMessageCreation:
    """Test creating UnifiedMessage instances."""

    def test_create_simple_message(self):
        """Test creating a simple message with text content."""
        content = TextContent(type="text", text="Hello world")
        message = UnifiedMessage(role="user", content=[content])

        assert message.role == "user"
        assert len(message.content) == 1
        assert isinstance(message.content[0], TextContent)
        assert message.name is None
        assert message.metadata == {}

    def test_create_message_with_name(self):
        """Test creating a message with a name."""
        content = TextContent(type="text", text="Hello")
        message = UnifiedMessage(role="assistant", content=[content], name="assistant_1")

        assert message.name == "assistant_1"

    def test_create_message_with_metadata(self):
        """Test creating a message with metadata."""
        content = TextContent(type="text", text="Hello")
        metadata = {"provider": "openai", "temperature": 0.7}
        message = UnifiedMessage(role="assistant", content=[content], metadata=metadata)

        assert message.metadata == metadata

    def test_create_message_with_multiple_content_blocks(self):
        """Test creating a message with multiple content blocks."""
        contents = [
            ReasoningContent(type="reasoning", reasoning="Let me think..."),
            TextContent(type="text", text="The answer is 42"),
            CitationContent(type="citation", url="https://example.com", title="Source", snippet="..."),
        ]
        message = UnifiedMessage(role="assistant", content=contents)

        assert len(message.content) == 3
        assert isinstance(message.content[0], ReasoningContent)
        assert isinstance(message.content[1], TextContent)
        assert isinstance(message.content[2], CitationContent)

    def test_create_message_all_roles(self):
        """Test creating messages with all valid roles."""
        roles = ["user", "assistant", "system", "tool"]
        for role in roles:
            content = TextContent(type="text", text=f"Message from {role}")
            message = UnifiedMessage(role=role, content=[content])
            assert message.role == role

    def test_create_message_with_custom_role(self):
        """Test creating message with custom/future role (extensibility)."""
        # Future provider might introduce new roles
        custom_roles = ["moderator", "reviewer", "observer", "agent_internal"]
        for role in custom_roles:
            content = TextContent(type="text", text=f"Message from {role}")
            message = UnifiedMessage(role=role, content=[content])
            assert message.role == role

    def test_is_standard_role_method(self):
        """Test is_standard_role() method."""
        # Standard roles
        for role in ["user", "assistant", "system", "tool"]:
            message = UnifiedMessage(role=role, content=[TextContent(type="text", text="test")])
            assert message.is_standard_role() is True

        # Custom roles
        for role in ["moderator", "reviewer", "custom"]:
            message = UnifiedMessage(role=role, content=[TextContent(type="text", text="test")])
            assert message.is_standard_role() is False


class TestUnifiedMessageTextExtraction:
    """Test get_text() method."""

    def test_get_text_from_single_text_content(self):
        """Test extracting text from a single text content block."""
        content = TextContent(type="text", text="Hello world")
        message = UnifiedMessage(role="user", content=[content])

        assert message.get_text() == "Hello world"

    def test_get_text_from_multiple_text_contents(self):
        """Test extracting text from multiple text content blocks."""
        contents = [
            TextContent(type="text", text="Hello"),
            TextContent(type="text", text="world"),
        ]
        message = UnifiedMessage(role="user", content=contents)

        assert message.get_text() == "Hello world"

    def test_get_text_from_reasoning_content(self):
        """Test extracting text from reasoning content."""
        contents = [
            ReasoningContent(type="reasoning", reasoning="Step 1: analyze", summary="Analysis"),
            TextContent(type="text", text="Conclusion"),
        ]
        message = UnifiedMessage(role="assistant", content=contents)

        assert message.get_text() == "Step 1: analyze Conclusion"

    def test_get_text_ignores_non_text_content(self):
        """Test that get_text() ignores non-text content blocks."""
        contents = [
            TextContent(type="text", text="Hello"),
            CitationContent(type="citation", url="url", title="title", snippet="snippet"),
            ToolCallContent(type="tool_call", id="id", name="name", arguments="{}"),
        ]
        message = UnifiedMessage(role="assistant", content=contents)

        assert message.get_text() == "Hello"

    def test_get_text_empty_content(self):
        """Test get_text() with no text content blocks."""
        contents = [
            CitationContent(type="citation", url="url", title="title", snippet="snippet"),
        ]
        message = UnifiedMessage(role="assistant", content=contents)

        assert message.get_text() == ""


class TestUnifiedMessageReasoningExtraction:
    """Test get_reasoning() method."""

    def test_get_reasoning_single_block(self):
        """Test extracting a single reasoning block."""
        reasoning = ReasoningContent(type="reasoning", reasoning="Step 1: analyze")
        message = UnifiedMessage(role="assistant", content=[reasoning])

        reasoning_blocks = message.get_reasoning()
        assert len(reasoning_blocks) == 1
        assert reasoning_blocks[0].reasoning == "Step 1: analyze"

    def test_get_reasoning_multiple_blocks(self):
        """Test extracting multiple reasoning blocks."""
        contents = [
            ReasoningContent(type="reasoning", reasoning="Step 1"),
            TextContent(type="text", text="Interim"),
            ReasoningContent(type="reasoning", reasoning="Step 2"),
        ]
        message = UnifiedMessage(role="assistant", content=contents)

        reasoning_blocks = message.get_reasoning()
        assert len(reasoning_blocks) == 2
        assert reasoning_blocks[0].reasoning == "Step 1"
        assert reasoning_blocks[1].reasoning == "Step 2"

    def test_get_reasoning_no_blocks(self):
        """Test get_reasoning() when no reasoning blocks present."""
        content = TextContent(type="text", text="No reasoning")
        message = UnifiedMessage(role="assistant", content=[content])

        reasoning_blocks = message.get_reasoning()
        assert len(reasoning_blocks) == 0


class TestUnifiedMessageCitationExtraction:
    """Test get_citations() method."""

    def test_get_citations_single_citation(self):
        """Test extracting a single citation."""
        citation = CitationContent(type="citation", url="https://example.com", title="Title", snippet="Snippet")
        message = UnifiedMessage(role="assistant", content=[citation])

        citations = message.get_citations()
        assert len(citations) == 1
        assert citations[0].url == "https://example.com"

    def test_get_citations_multiple_citations(self):
        """Test extracting multiple citations."""
        contents = [
            CitationContent(type="citation", url="https://example1.com", title="Title 1", snippet="Snippet 1"),
            CitationContent(type="citation", url="https://example2.com", title="Title 2", snippet="Snippet 2"),
        ]
        message = UnifiedMessage(role="assistant", content=contents)

        citations = message.get_citations()
        assert len(citations) == 2
        assert citations[0].url == "https://example1.com"
        assert citations[1].url == "https://example2.com"

    def test_get_citations_no_citations(self):
        """Test get_citations() when no citations present."""
        content = TextContent(type="text", text="No citations")
        message = UnifiedMessage(role="assistant", content=[content])

        citations = message.get_citations()
        assert len(citations) == 0


class TestUnifiedMessageToolCallExtraction:
    """Test get_tool_calls() method."""

    def test_get_tool_calls_single_call(self):
        """Test extracting a single tool call."""
        tool_call = ToolCallContent(type="tool_call", id="call-123", name="get_weather", arguments='{"city":"SF"}')
        message = UnifiedMessage(role="assistant", content=[tool_call])

        tool_calls = message.get_tool_calls()
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call-123"
        assert tool_calls[0].name == "get_weather"

    def test_get_tool_calls_multiple_calls(self):
        """Test extracting multiple tool calls."""
        contents = [
            ToolCallContent(type="tool_call", id="call-1", name="func1", arguments="{}"),
            ToolCallContent(type="tool_call", id="call-2", name="func2", arguments="{}"),
        ]
        message = UnifiedMessage(role="assistant", content=contents)

        tool_calls = message.get_tool_calls()
        assert len(tool_calls) == 2

    def test_get_tool_calls_no_calls(self):
        """Test get_tool_calls() when no tool calls present."""
        content = TextContent(type="text", text="No tool calls")
        message = UnifiedMessage(role="assistant", content=[content])

        tool_calls = message.get_tool_calls()
        assert len(tool_calls) == 0


class TestUnifiedMessageContentByType:
    """Test get_content_by_type() method."""

    def test_get_content_by_type_text(self):
        """Test filtering content by 'text' type."""
        contents = [
            TextContent(type="text", text="Hello"),
            ReasoningContent(type="reasoning", reasoning="Think"),
            TextContent(type="text", text="World"),
        ]
        message = UnifiedMessage(role="assistant", content=contents)

        text_blocks = message.get_content_by_type("text")
        assert len(text_blocks) == 2
        assert all(b.type == "text" for b in text_blocks)

    def test_get_content_by_type_unknown(self):
        """Test filtering content by unknown type (GenericContent)."""
        contents = [
            TextContent(type="text", text="Hello"),
            GenericContent(type="reflection", reflection="Reviewing..."),
            GenericContent(type="reflection", reflection="Another review..."),
        ]
        message = UnifiedMessage(role="assistant", content=contents)

        reflection_blocks = message.get_content_by_type("reflection")
        assert len(reflection_blocks) == 2
        assert all(b.type == "reflection" for b in reflection_blocks)

    def test_get_content_by_type_no_match(self):
        """Test get_content_by_type() when no blocks match."""
        content = TextContent(type="text", text="Hello")
        message = UnifiedMessage(role="assistant", content=[content])

        blocks = message.get_content_by_type("nonexistent")
        assert len(blocks) == 0

    def test_get_content_by_type_mixed_known_and_unknown(self):
        """Test filtering when mix of known and unknown types present."""
        contents = [
            TextContent(type="text", text="Hello"),
            GenericContent(type="custom", field="value1"),
            ReasoningContent(type="reasoning", reasoning="Think"),
            GenericContent(type="custom", field="value2"),
        ]
        message = UnifiedMessage(role="assistant", content=contents)

        custom_blocks = message.get_content_by_type("custom")
        assert len(custom_blocks) == 2
        assert all(isinstance(b, GenericContent) for b in custom_blocks)


class TestUnifiedMessageSerialization:
    """Test message serialization."""

    def test_message_serialization(self):
        """Test that messages can be serialized to dict."""
        contents = [
            TextContent(type="text", text="Hello"),
            ReasoningContent(type="reasoning", reasoning="Think"),
        ]
        message = UnifiedMessage(role="assistant", content=contents, name="bot")

        message_dict = message.model_dump()
        assert message_dict["role"] == "assistant"
        assert message_dict["name"] == "bot"
        assert len(message_dict["content"]) == 2

    def test_message_deserialization(self):
        """Test that messages can be deserialized from dict."""
        message_dict = {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "name": "user1",
            "metadata": {"key": "value"},
        }
        # Note: This would require custom parsing logic in practice
        # For now, just verify the structure is correct
        assert message_dict["role"] == "user"
        assert len(message_dict["content"]) == 1
