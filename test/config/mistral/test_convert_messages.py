# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest
from fast_depends.use import SerializerCls
from mistralai.client.models import (
    AssistantMessage,
    DocumentURLChunk,
    FileChunk,
    ImageURLChunk,
    SystemMessage,
    TextChunk,
    ToolMessage,
    UserMessage,
)

from ag2 import ToolResult
from ag2.compact import CompactionSummary
from ag2.config.mistral.mappers import convert_messages
from ag2.events import (
    AudioInput,
    DataInput,
    DocumentInput,
    FileIdInput,
    ImageInput,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolErrorEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from ag2.exceptions import UnsupportedInputError

PNG = b"\x89PNG\r\n\x1a\n"


def test_system_prompt_is_joined() -> None:
    result = convert_messages(["You are helpful.", "Be brief."], [], SerializerCls)

    assert result == [SystemMessage(content="You are helpful.\nBe brief.")]


def test_empty_system_prompt_is_omitted() -> None:
    assert convert_messages(["", ""], [], SerializerCls) == []


def test_user_text_input_collapses_to_string() -> None:
    result = convert_messages([], [ModelRequest([TextInput("hello")])], SerializerCls)

    assert result == [UserMessage(content="hello")]


def test_data_input_is_serialized_as_text() -> None:
    data = {"category": "books", "limit": 3}

    result = convert_messages([], [ModelRequest([DataInput(data)])], SerializerCls)

    assert result == [UserMessage(content=SerializerCls.encode(data).decode())]


def test_multiple_parts_stay_as_chunks() -> None:
    result = convert_messages(
        [],
        [ModelRequest([TextInput("look:"), TextInput("here")])],
        SerializerCls,
    )

    assert result == [UserMessage(content=[TextChunk(text="look:"), TextChunk(text="here")])]


def test_compaction_summary_becomes_user_turn() -> None:
    result = convert_messages([], [CompactionSummary(summary="we discussed books")], SerializerCls)

    assert result == [UserMessage(content="[Summary of earlier conversation]\nwe discussed books")]


def test_assistant_text_and_tool_call() -> None:
    response = ModelResponse(
        message=ModelMessage("Let me check."),
        tool_calls=ToolCallsEvent([ToolCallEvent(id="tc_1", name="list_items", arguments='{"category": "books"}')]),
    )

    [message] = convert_messages([], [response], SerializerCls)

    assert isinstance(message, AssistantMessage)
    assert message.content == "Let me check."
    assert [(c.id, c.function.name, c.function.arguments) for c in message.tool_calls] == [
        ("tc_1", "list_items", '{"category": "books"}')
    ]


def test_assistant_without_tool_calls_omits_them() -> None:
    response = ModelResponse(message=ModelMessage("done"), tool_calls=ToolCallsEvent([]))

    [message] = convert_messages([], [response], SerializerCls)

    assert message.content == "done"
    assert not message.tool_calls


def test_tool_results_event() -> None:
    event = ToolResultsEvent([
        ToolResultEvent(parent_id="tc_1", name="list_items", result=ToolResult("42")),
    ])

    result = convert_messages([], [event], SerializerCls)

    assert result == [ToolMessage(content="42", tool_call_id="tc_1")]


def test_tool_error_is_sent_as_a_tool_message() -> None:
    event = ToolResultsEvent([
        ToolErrorEvent(parent_id="tc_2", name="fail", error=ValueError("boom"), result=ToolResult("boom")),
    ])

    result = convert_messages([], [event], SerializerCls)

    assert result == [ToolMessage(content="boom", tool_call_id="tc_2")]


def test_constituent_tool_events_do_not_duplicate_the_batch() -> None:
    """History holds the batch and its constituents; only the batch is mapped."""
    inner = ToolResultEvent(parent_id="tc_1", name="list_items", result=ToolResult("42"))

    result = convert_messages([], [ToolResultsEvent([inner]), inner], SerializerCls)

    assert result == [ToolMessage(content="42", tool_call_id="tc_1")]


class TestImageInput:
    def test_url(self) -> None:
        request = ModelRequest([ImageInput(url="https://example.com/cat.png")])

        [message] = convert_messages([], [request], SerializerCls)

        assert message.content == [ImageURLChunk(image_url="https://example.com/cat.png")]

    def test_binary_becomes_data_url(self) -> None:
        request = ModelRequest([ImageInput(data=PNG, media_type="image/png")])

        [message] = convert_messages([], [request], SerializerCls)

        expected = f"data:image/png;base64,{base64.b64encode(PNG).decode()}"
        assert message.content == [ImageURLChunk(image_url=expected)]


class TestDocumentInput:
    def test_url(self) -> None:
        request = ModelRequest([DocumentInput(url="https://example.com/report.pdf")])

        [message] = convert_messages([], [request], SerializerCls)

        assert message.content == [DocumentURLChunk(document_url="https://example.com/report.pdf", document_name=None)]

    def test_binary_becomes_data_url(self) -> None:
        request = ModelRequest([DocumentInput(data=b"%PDF-", media_type="application/pdf")])

        [message] = convert_messages([], [request], SerializerCls)

        expected = f"data:application/pdf;base64,{base64.b64encode(b'%PDF-').decode()}"
        assert message.content == [DocumentURLChunk(document_url=expected, document_name=None)]

    def test_binary_from_path_carries_filename(self, tmp_path) -> None:
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-")
        request = ModelRequest([DocumentInput(path=pdf)])

        [message] = convert_messages([], [request], SerializerCls)

        expected = f"data:application/pdf;base64,{base64.b64encode(b'%PDF-').decode()}"
        assert message.content == [DocumentURLChunk(document_url=expected, document_name="report.pdf")]


def test_file_id_input() -> None:
    request = ModelRequest([FileIdInput(file_id="file_123")])

    [message] = convert_messages([], [request], SerializerCls)

    assert message.content == [FileChunk(file_id="file_123")]


def test_audio_input_is_unsupported() -> None:
    request = ModelRequest([AudioInput(data=b"\x00", media_type="audio/wav")])

    with pytest.raises(UnsupportedInputError, match="mistral"):
        convert_messages([], [request], SerializerCls)
