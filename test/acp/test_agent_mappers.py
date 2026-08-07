# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the AG2 -> ACP direction of :mod:`ag2.acp.mappers`."""

import base64

import acp
import pytest
from acp import schema

from ag2.acp.mappers import event_to_session_update, prompt_to_inputs, tool_result_text
from ag2.events import (
    BinaryInput,
    ModelMessageChunk,
    ModelReasoning,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
)
from ag2.events.tool_events import ToolResult
from ag2.events.types import ModelMessage


class TestPromptToInputs:
    def test_text_becomes_text_input(self) -> None:
        [mapped] = prompt_to_inputs([acp.text_block("hello")])

        assert mapped == TextInput("hello")

    def test_order_is_preserved(self) -> None:
        mapped = prompt_to_inputs([acp.text_block("one"), acp.text_block("two")])

        assert mapped == [TextInput("one"), TextInput("two")]

    def test_an_image_becomes_binary_input(self) -> None:
        block = schema.ImageContentBlock(
            type="image",
            data=base64.b64encode(b"png-bytes").decode(),
            mime_type="image/png",
        )

        [mapped] = prompt_to_inputs([block])

        assert isinstance(mapped, BinaryInput)
        assert mapped.data == b"png-bytes"

    def test_an_embedded_text_resource_becomes_text(self) -> None:
        block = schema.EmbeddedResourceContentBlock(
            type="resource",
            resource=schema.TextResourceContents(uri="file:///a.md", text="body"),
        )

        [mapped] = prompt_to_inputs([block])

        assert mapped == TextInput("body")

    def test_an_embedded_blob_uses_the_inlined_bytes(self) -> None:
        block = schema.EmbeddedResourceContentBlock(
            type="resource",
            resource=schema.BlobResourceContents(
                uri="file:///a.pdf",
                blob=base64.b64encode(b"%PDF-1.4").decode(),
                mime_type="application/pdf",
            ),
        )

        [mapped] = prompt_to_inputs([block])

        assert isinstance(mapped, BinaryInput)
        assert mapped.data == b"%PDF-1.4"

    def test_a_resource_link_is_referenced_not_dereferenced(self) -> None:
        block = schema.ResourceContentBlock(type="resource_link", uri="file:///secret", name="secret")

        [mapped] = prompt_to_inputs([block])

        assert isinstance(mapped, TextInput)
        assert "file:///secret" in mapped.content

    def test_an_unmappable_block_does_not_drop_the_rest(self) -> None:
        unmappable = schema.ImageContentBlock(type="image", data="", mime_type="image/png")

        mapped = prompt_to_inputs([unmappable, acp.text_block("kept")])

        assert mapped == [TextInput("kept")]


class TestEventToSessionUpdate:
    def test_a_message_chunk_becomes_an_agent_message_chunk(self) -> None:
        update = event_to_session_update(ModelMessageChunk("hello"))

        assert isinstance(update, schema.AgentMessageChunk)
        assert update.content.text == "hello"

    def test_reasoning_is_withheld_by_default(self) -> None:
        assert event_to_session_update(ModelReasoning("internal")) is None

    def test_reasoning_is_projected_when_opted_in(self) -> None:
        update = event_to_session_update(ModelReasoning("internal"), stream_thoughts=True)

        assert isinstance(update, schema.AgentThoughtChunk)
        assert update.content.text == "internal"

    def test_a_tool_call_carries_its_id_name_and_arguments(self) -> None:
        update = event_to_session_update(ToolCallEvent(id="c1", name="add", arguments='{"a": 1}'))

        assert isinstance(update, schema.ToolCallStart)
        assert (update.tool_call_id, update.title, update.raw_input) == ("c1", "add", {"a": 1})

    def test_a_tool_result_is_reported_completed(self) -> None:
        update = event_to_session_update(ToolResultEvent(parent_id="c1", name="add", result=ToolResult("3")))

        assert isinstance(update, schema.ToolCallProgress)
        assert update.status == "completed"
        assert update.content[0].content.text == "3"

    def test_a_tool_error_is_reported_failed(self) -> None:
        event = ToolErrorEvent(parent_id="c1", name="add", result=ToolResult("x"), error=ValueError("kaboom"))

        update = event_to_session_update(event)

        assert isinstance(update, schema.ToolCallProgress)
        assert update.status == "failed"
        assert "kaboom" in update.content[0].content.text

    def test_a_tool_error_is_not_mistaken_for_a_success(self) -> None:
        """``ToolErrorEvent`` subclasses ``ToolResultEvent`` — order matters."""
        event = ToolErrorEvent(parent_id="c1", name="add", result=ToolResult("x"), error=ValueError("kaboom"))

        assert event_to_session_update(event).status == "failed"

    def test_the_final_response_is_not_projected(self) -> None:
        """Its text already went out as chunks; re-sending would duplicate the reply."""
        assert event_to_session_update(ModelResponse(ModelMessage("the whole answer"))) is None


class TestToolResultText:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("plain", "plain"),
            (200, "200"),
            ({"total": 200}, '{"total": 200}'),
            ([1, 2], "[1, 2]"),
        ],
    )
    def test_non_string_results_render_as_text(self, value: object, expected: str) -> None:
        assert tool_result_text(ToolResult(value)) == expected

    def test_binary_parts_get_a_placeholder_not_the_bytes(self) -> None:
        result = ToolResult(BinaryInput(b"\x89PNG", media_type="image/png"))

        rendered = tool_result_text(result)

        assert "PNG" not in rendered
        assert rendered.startswith("[")
