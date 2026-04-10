# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict

from autogen.beta.config.ollama.mappers import convert_messages
from autogen.beta.events import (
    AudioUrlInput,
    BinaryInput,
    DocumentUrlInput,
    FileIdInput,
    ImageUrlInput,
    ModelRequest,
    ModelResponse,
)
from autogen.beta.events.tool_events import ToolCallEvent, ToolCallsEvent
from autogen.beta.exceptions import UnsupportedInputError


def _model_response_with_tool_call(arguments: str | None) -> ModelResponse:
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
        result = convert_messages([], [_model_response_with_tool_call(arguments)])

        assert result[0] == IsPartialDict({
            "role": "assistant",
            "tool_calls": [IsPartialDict({"function": IsPartialDict({"name": "list_items", "arguments": {}})})],
        })

    def test_valid_arguments_are_preserved(self) -> None:
        result = convert_messages([], [_model_response_with_tool_call('{"category": "books"}')])

        assert result[0] == IsPartialDict({
            "tool_calls": [IsPartialDict({"function": IsPartialDict({"arguments": {"category": "books"}})})],
        })


def test_audio_url_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="AudioUrlInput.*ollama"):
        convert_messages([], [ModelRequest([AudioUrlInput(url="https://example.com/audio.wav")])])


def test_image_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="ImageUrlInput.*ollama"):
        convert_messages([], [ModelRequest([ImageUrlInput(url="https://example.com/img.png")])])


def test_file_id_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="FileIdInput.*ollama"):
        convert_messages([], [ModelRequest([FileIdInput(file_id="file-abc123")])])


def test_binary_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="BinaryInput.*ollama"):
        convert_messages([], [ModelRequest([BinaryInput(data=b"data", media_type="image/png")])])


def test_document_url_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="DocumentUrlInput.*ollama"):
        convert_messages([], [ModelRequest([DocumentUrlInput(url="https://example.com/doc.pdf")])])
