# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest
from dirty_equals import IsPartialDict
from fast_depends.pydantic import PydanticSerializer

from ag2.compact import CompactionSummary
from ag2.config.ollama.mappers import convert_messages
from ag2.events import (
    AudioInput,
    BinaryInput,
    DocumentInput,
    FileIdInput,
    ImageInput,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolNotFoundEvent,
    ToolResultsEvent,
)
from ag2.exceptions import ToolNotFoundError, UnsupportedInputError


def _model_response_with_tool_call(arguments: str) -> ModelResponse:
    return ModelResponse(
        message=None,
        tool_calls=ToolCallsEvent(
            calls=[ToolCallEvent(id="tc_1", name="list_items", arguments=arguments)],
        ),
    )


class TestConvertMessagesEmptyArguments:
    """json.loads must not crash on empty tool call arguments."""

    def test_empty_arguments_produce_empty_dict(self) -> None:
        result = convert_messages([], [_model_response_with_tool_call("")], PydanticSerializer())

        assert result[0] == IsPartialDict({
            "role": "assistant",
            "tool_calls": [IsPartialDict({"function": IsPartialDict({"name": "list_items", "arguments": {}})})],
        })

    def test_valid_arguments_are_preserved(self) -> None:
        result = convert_messages([], [_model_response_with_tool_call('{"category": "books"}')], PydanticSerializer())

        assert result[0] == IsPartialDict({
            "tool_calls": [IsPartialDict({"function": IsPartialDict({"arguments": {"category": "books"}})})],
        })


def test_audio_url_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="UrlInput.*ollama"):
        convert_messages([], [ModelRequest([AudioInput(url="https://example.com/audio.wav")])], PydanticSerializer())


def test_image_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="UrlInput.*ollama"):
        convert_messages([], [ModelRequest([ImageInput(url="https://example.com/img.png")])], PydanticSerializer())


def test_document_url_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="UrlInput.*ollama"):
        convert_messages([], [ModelRequest([DocumentInput(url="https://example.com/doc.pdf")])], PydanticSerializer())


def test_file_id_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="FileIdInput.*ollama"):
        convert_messages([], [ModelRequest([FileIdInput(file_id="file-abc123")])], PydanticSerializer())


def test_binary_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="BinaryInput.*ollama"):
        convert_messages([], [ModelRequest([BinaryInput(data=b"data", media_type="image/png")])], PydanticSerializer())


class TestImageBinaryInput:
    """Ollama multimodal (llava) — images travel via the `images` field on the user message."""

    PNG = b"\x89PNG\r\n"

    def test_image_only(self) -> None:
        result = convert_messages(
            [], [ModelRequest([ImageInput(data=self.PNG, media_type="image/png")])], PydanticSerializer()
        )

        b64 = base64.b64encode(self.PNG).decode()
        assert result == [{"role": "user", "content": "", "images": [b64]}]

    def test_text_plus_image(self) -> None:
        result = convert_messages(
            [],
            [ModelRequest([TextInput("what is in this image?"), ImageInput(data=self.PNG, media_type="image/png")])],
            PydanticSerializer(),
        )

        b64 = base64.b64encode(self.PNG).decode()
        assert result == [
            {"role": "user", "content": "what is in this image?", "images": [b64]},
        ]

    def test_multiple_images(self) -> None:
        png2 = b"\x89PNG\r\nB"
        result = convert_messages(
            [],
            [
                ModelRequest([
                    ImageInput(data=self.PNG, media_type="image/png"),
                    ImageInput(data=png2, media_type="image/png"),
                ])
            ],
            PydanticSerializer(),
        )

        assert result == [
            {
                "role": "user",
                "content": "",
                "images": [base64.b64encode(self.PNG).decode(), base64.b64encode(png2).decode()],
            }
        ]

    def test_text_without_image_stays_plain(self) -> None:
        """Regression: text-only message must not acquire an `images` key."""
        result = convert_messages([], [ModelRequest([TextInput("hello")])], PydanticSerializer())

        assert result == [{"role": "user", "content": "hello"}]


def test_multiple_text_inputs_emit_separate_messages() -> None:
    """Multiple TextInput in one turn must not be joined; emit one user message each."""
    result = convert_messages(
        [], [ModelRequest([TextInput("first"), TextInput("second"), TextInput("third")])], PydanticSerializer()
    )

    assert result == [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "user", "content": "third"},
    ]


def test_multiple_text_inputs_with_images_attach_to_last() -> None:
    """Images must attach to the last TextInput to stay within the same turn."""
    png = b"\x89PNG\r\n"
    result = convert_messages(
        [],
        [ModelRequest([TextInput("intro"), TextInput("look at this"), ImageInput(data=png, media_type="image/png")])],
        PydanticSerializer(),
    )

    b64 = base64.b64encode(png).decode()
    assert result == [
        {"role": "user", "content": "intro"},
        {"role": "user", "content": "look at this", "images": [b64]},
    ]


def test_hallucinated_tool_call_maps_with_error_text() -> None:
    # Regression: a not-found tool call used to leave result=None and crash on r.result.parts.
    call = ToolCallEvent(id="tc_1", name="ghost_tool")
    event = ToolResultsEvent(results=[ToolNotFoundEvent.from_call(call, ToolNotFoundError("ghost_tool"))])

    result = convert_messages([], [event], PydanticSerializer())

    assert result == [{"role": "tool", "content": "ag2.exceptions.ToolNotFoundError: Tool `ghost_tool` not found\n"}]


def test_compaction_summary_renders_as_user_turn() -> None:
    summary = CompactionSummary(summary="Looked up Paris and Tokyo.", event_count=6)

    result = convert_messages([], [summary], PydanticSerializer())

    assert result == [{"role": "user", "content": "[Summary of earlier conversation]\nLooked up Paris and Tokyo."}]
