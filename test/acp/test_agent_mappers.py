# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the AG2 -> ACP direction of :mod:`ag2.acp.mappers`."""

import base64

import acp
import pytest
from acp import schema

from ag2.acp.mappers import (
    event_to_session_update,
    history_to_session_updates,
    input_to_block,
    prompt_to_inputs,
    tool_result_text,
)
from ag2.compact import CompactionSummary
from ag2.events import (
    BinaryInput,
    DataInput,
    HumanInputRequest,
    ImageInput,
    ModelMessageChunk,
    ModelReasoning,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolResultEvent,
    ToolResultsEvent,
    UsageEvent,
)
from ag2.events.tool_events import ToolResult
from ag2.events.types import HumanMessage, ModelMessage


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


def _call(call_id: str = "c1") -> ToolCallEvent:
    return ToolCallEvent(id=call_id, name="add", arguments='{"a": 1}')


def _result(call_id: str = "c1") -> ToolResultEvent:
    return ToolResultEvent(parent_id=call_id, name="add", result=ToolResult("3"))


class TestHistoryToSessionUpdates:
    """Storage holds what the loop persisted, not what the Client saw; replay bridges the two."""

    def test_user_text_parts_become_user_chunks_sharing_a_message_id(self) -> None:
        updates = history_to_session_updates([ModelRequest([TextInput("hi"), TextInput("there")])], session_id="s")

        assert [type(u) for u in updates] == [schema.UserMessageChunk, schema.UserMessageChunk]
        assert [u.content.text for u in updates] == ["hi", "there"]
        assert {u.message_id for u in updates} == {"s:u1"}

    def test_agent_text_comes_from_the_response_not_from_chunks(self) -> None:
        """Chunks are transient and never stored; the response carries the whole message."""
        [update] = history_to_session_updates([ModelResponse(ModelMessage("the answer"))], session_id="s")

        assert isinstance(update, schema.AgentMessageChunk)
        assert (update.content.text, update.message_id) == ("the answer", "s:a1")

    def test_agent_text_precedes_its_tool_calls(self) -> None:
        response = ModelResponse(ModelMessage("let me add"), tool_calls=ToolCallsEvent([_call()]))

        updates = history_to_session_updates([response], session_id="s")

        assert [type(u) for u in updates] == [schema.AgentMessageChunk, schema.ToolCallStart]
        assert updates[1].tool_call_id == "c1"

    def test_a_tool_call_persisted_three_times_starts_once(self) -> None:
        """The response, the ``ToolCallsEvent`` and the loose ``ToolCallEvent`` all name the same call."""
        events = [ModelResponse(tool_calls=ToolCallsEvent([_call()])), ToolCallsEvent([_call()]), _call()]

        updates = history_to_session_updates(events, session_id="s")

        assert [type(u) for u in updates] == [schema.ToolCallStart]

    def test_a_result_persisted_twice_settles_once(self) -> None:
        events = [_call(), _result(), ToolResultsEvent([_result()])]

        updates = history_to_session_updates(events, session_id="s")

        assert [type(u) for u in updates] == [schema.ToolCallStart, schema.ToolCallProgress]
        assert updates[1].status == "completed"

    def test_a_wrapped_result_alone_settles_the_call(self) -> None:
        """A repaired batch carries its results only inside the wrapper."""
        updates = history_to_session_updates([_call(), ToolResultsEvent([_result()])], session_id="s")

        assert [type(u) for u in updates] == [schema.ToolCallStart, schema.ToolCallProgress]

    def test_an_error_result_is_failed(self) -> None:
        error = ToolErrorEvent(parent_id="c1", name="add", result=ToolResult("x"), error=ValueError("kaboom"))

        _, update = history_to_session_updates([_call(), error], session_id="s")

        assert update.status == "failed"
        assert "kaboom" in update.content[0].content.text

    def test_a_result_for_a_call_never_started_is_dropped(self) -> None:
        assert history_to_session_updates([_result()], session_id="s") == []

    def test_human_input_round_trips_as_a_question_and_an_answer(self) -> None:
        request = HumanInputRequest("which one?")
        answer = HumanMessage("the second", parent_id=request.id)

        updates = history_to_session_updates([request, answer], session_id="s")

        assert [type(u) for u in updates] == [schema.AgentMessageChunk, schema.UserMessageChunk]
        assert [u.content.text for u in updates] == ["which one?", "the second"]

    def test_telemetry_and_compaction_are_not_replayed(self) -> None:
        events = [UsageEvent(), CompactionSummary(summary="earlier talk", event_count=9)]

        assert history_to_session_updates(events, session_id="s") == []

    def test_a_response_without_a_message_emits_no_chunk(self) -> None:
        assert history_to_session_updates([ModelResponse()], session_id="s") == []

    def test_message_ids_count_per_role_from_the_start(self) -> None:
        events = [
            ModelRequest([TextInput("one")]),
            ModelResponse(ModelMessage("1")),
            ModelRequest([TextInput("two")]),
            ModelResponse(ModelMessage("2")),
        ]

        updates = history_to_session_updates(events, session_id="s")

        assert [u.message_id for u in updates] == ["s:u1", "s:a1", "s:u2", "s:a2"]

    def test_replaying_the_same_history_twice_yields_the_same_updates(self) -> None:
        events = [
            ModelRequest([TextInput("q")]),
            ModelResponse(ModelMessage("a"), tool_calls=ToolCallsEvent([_call()])),
        ]

        first = history_to_session_updates(events, session_id="s")
        second = history_to_session_updates(events, session_id="s")

        assert first == second


class TestInputToBlock:
    def test_an_image_goes_back_as_an_image_block(self) -> None:
        block = input_to_block(ImageInput(data=b"png-bytes", media_type="image/png"))

        assert isinstance(block, schema.ImageContentBlock)
        assert (base64.b64decode(block.data), block.mime_type) == (b"png-bytes", "image/png")

    def test_data_renders_as_json_text(self) -> None:
        assert input_to_block(DataInput({"total": 2})).text == '{"total": 2}'

    def test_a_document_is_named_not_reproduced(self) -> None:
        block = input_to_block(BinaryInput(b"%PDF-1.4", media_type="application/pdf", kind="document"))

        assert isinstance(block, schema.TextContentBlock)
        assert "PDF" not in block.text
        assert "document" in block.text

    def test_a_url_is_named_with_its_kind(self) -> None:
        assert input_to_block(ImageInput("https://example.com/a.png")).text == "[image] https://example.com/a.png"


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
