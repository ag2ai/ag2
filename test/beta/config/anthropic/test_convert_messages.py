# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict

from autogen.beta import ToolResult
from autogen.beta.config.anthropic.mappers import convert_messages
from autogen.beta.events import (
    AudioUrlInput,
    BinaryInput,
    DocumentUrlInput,
    FileIdInput,
    ImageUrlInput,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from autogen.beta.exceptions import UnsupportedInputError


def _model_response_with_tool_call(arguments: str | None) -> ModelResponse:
    """Helper to build a ModelResponse containing a single tool call."""
    return ModelResponse(
        message=None,
        tool_calls=ToolCallsEvent(
            calls=[ToolCallEvent(id="tc_1", name="list_items", arguments=arguments)],
        ),
    )


class TestConvertMessagesEmptyArguments:
    """json.loads must not crash on empty or None tool call arguments."""

    @pytest.mark.parametrize("arguments", ["", None])
    def test_empty_arguments_produce_empty_dict(self, arguments: str | None) -> None:
        response = _model_response_with_tool_call(arguments)
        result = convert_messages([response])

        assert result == [
            IsPartialDict({
                "role": "assistant",
                "content": [IsPartialDict({"type": "tool_use", "id": "tc_1", "name": "list_items", "input": {}})],
            }),
        ]

    def test_valid_arguments_are_preserved(self) -> None:
        response = _model_response_with_tool_call('{"category": "books"}')
        result = convert_messages([response])

        assert result == [
            IsPartialDict({
                "content": [IsPartialDict({"type": "tool_use", "input": {"category": "books"}})],
            }),
        ]

    def test_empty_object_arguments(self) -> None:
        response = _model_response_with_tool_call("{}")
        result = convert_messages([response])

        assert result == [
            IsPartialDict({
                "content": [IsPartialDict({"type": "tool_use", "input": {}})],
            }),
        ]


def test_full_sequence_with_empty_args() -> None:
    """A request -> response-with-tool-call -> tool-result sequence should convert cleanly."""
    events = [
        ModelRequest([TextInput("What items do we have?")]),
        _model_response_with_tool_call(""),
        ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="list_items",
                    result=ToolResult(content="apple, banana"),
                )
            ],
        ),
    ]
    result = convert_messages(events)

    assert result[0] == IsPartialDict({"role": "user"})
    assert result[1] == IsPartialDict({
        "role": "assistant",
        "content": [IsPartialDict({"input": {}})],
    })
    assert result[2] == IsPartialDict({
        "role": "user",
        "content": [IsPartialDict({"type": "tool_result"})],
    })


def test_audio_url_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="AudioUrlInput.*anthropic"):
        convert_messages([ModelRequest([AudioUrlInput(url="https://example.com/audio.wav")])])


def test_image_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="ImageUrlInput.*anthropic"):
        convert_messages([ModelRequest([ImageUrlInput(url="https://example.com/img.png")])])


def test_file_id_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="FileIdInput.*anthropic"):
        convert_messages([ModelRequest([FileIdInput(file_id="file-abc123")])])


def test_binary_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="BinaryInput.*anthropic"):
        convert_messages([ModelRequest([BinaryInput(data=b"data", media_type="image/png")])])


def test_document_url_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="DocumentUrlInput.*anthropic"):
        convert_messages([ModelRequest([DocumentUrlInput(url="https://example.com/doc.pdf")])])
