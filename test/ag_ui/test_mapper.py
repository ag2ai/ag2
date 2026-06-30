# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from base64 import b64encode
from typing import Any

import pytest
from ag_ui.core import (
    AssistantMessage,
    AudioInputContent,
    BinaryInputContent,
    DocumentInputContent,
    FunctionCall,
    ImageInputContent,
    InputContentDataSource,
    InputContentUrlSource,
    SystemMessage,
    TextInputContent,
    ToolCall,
    ToolMessage,
    UserMessage,
    VideoInputContent,
)

from ag2 import ToolResult
from ag2.ag_ui.stream import AGStreamInput, map_agui_messages_to_events
from ag2.events import (
    AudioInput,
    BinaryInput,
    BinaryType,
    DocumentInput,
    ImageInput,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
    UrlInput,
    VideoInput,
)

from .utils import create_run_input

RAW_BYTES = b"\xff\xd8\xff\xe0"
B64_VALUE = b64encode(RAW_BYTES).decode()


def _command(*messages: object) -> AGStreamInput:
    return AGStreamInput(incoming=create_run_input(*messages), variables={})


class TestUserMessageString:
    def test_plain_string_becomes_text_input(self) -> None:
        command = _command(UserMessage(id="m1", content="hello"))

        prompt, messages, current_turn = map_agui_messages_to_events(command)

        assert prompt == []
        assert messages == []
        assert current_turn == [TextInput("hello")]

    def test_text_content_becomes_text_input(self) -> None:
        command = _command(UserMessage(id="m1", content=[TextInputContent(text="hi")]))

        _, messages, current_turn = map_agui_messages_to_events(command)

        assert messages == []
        assert current_turn == [TextInput("hi")]


@pytest.mark.parametrize(
    "content_cls,factory,kind,mime",
    [
        (ImageInputContent, ImageInput, BinaryType.IMAGE, "image/jpeg"),
        (AudioInputContent, AudioInput, BinaryType.AUDIO, "audio/wav"),
        (VideoInputContent, VideoInput, BinaryType.VIDEO, "video/mp4"),
        (DocumentInputContent, DocumentInput, BinaryType.DOCUMENT, "application/pdf"),
    ],
)
class TestMediaContentMapping:
    def test_url_source_maps_to_url_input(self, content_cls: type, factory: Any, kind: BinaryType, mime: str) -> None:
        url = "https://example.com/file"
        content = content_cls(source=InputContentUrlSource(value=url))
        command = _command(UserMessage(id="m1", content=[content]))

        _, messages, current_turn = map_agui_messages_to_events(command)

        assert messages == []
        assert current_turn == [factory(url)]
        [part] = current_turn
        assert isinstance(part, UrlInput)
        assert part.kind == kind

    def test_data_source_maps_to_binary_input(
        self, content_cls: type, factory: Any, kind: BinaryType, mime: str
    ) -> None:
        content = content_cls(source=InputContentDataSource(value=B64_VALUE, mime_type=mime))
        command = _command(UserMessage(id="m1", content=[content]))

        _, messages, current_turn = map_agui_messages_to_events(command)

        assert messages == []
        assert current_turn == [factory(data=RAW_BYTES, media_type=mime)]
        [part] = current_turn
        assert isinstance(part, BinaryInput)
        assert part.kind == kind


class TestBinaryContentRejected:
    def test_binary_type_raises(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            binary = BinaryInputContent(mime_type="application/octet-stream", url="https://x/blob")
        command = _command(UserMessage(id="m1", content=[binary]))

        with pytest.raises(ValueError, match="deprecated"):
            map_agui_messages_to_events(command)


class TestMetadata:
    def test_content_metadata_propagates_to_input_metadata(self) -> None:
        command = _command(
            UserMessage(
                id="m1",
                content=[
                    TextInputContent(text="hi"),
                    ImageInputContent(
                        source=InputContentUrlSource(value="https://x/i.png"),
                        metadata={"alt": "cat"},
                    ),
                ],
            )
        )

        _, messages, current_turn = map_agui_messages_to_events(command)

        assert messages == []
        text_part, image_part = current_turn
        assert text_part.metadata == {}
        assert image_part.metadata == {"alt": "cat"}


class TestNonUserRoles:
    def test_system_message_goes_to_prompt(self) -> None:
        command = _command(
            SystemMessage(id="s1", content="be brief"),
            UserMessage(id="u1", content="hi"),
        )

        prompt, messages, current_turn = map_agui_messages_to_events(command)

        assert prompt == ["be brief"]
        assert messages == []
        assert current_turn == [TextInput("hi")]

    def test_assistant_message_becomes_model_response(self) -> None:
        command = _command(
            UserMessage(id="u1", content="hi"),
            AssistantMessage(id="a1", content="hello!"),
        )

        _, messages, current_turn = map_agui_messages_to_events(command)

        assert current_turn == []
        assert messages == [
            ModelRequest([TextInput("hi")]),
            ModelResponse(ModelMessage("hello!"), tool_calls=ToolCallsEvent([])),
        ]

    def test_tool_message_becomes_tool_result(self) -> None:
        command = _command(
            UserMessage(id="u1", content="run tool"),
            AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[
                    ToolCall(id="t1", type="function", function=FunctionCall(name="do", arguments="{}")),
                ],
            ),
            ToolMessage(id="tm1", tool_call_id="t1", content="42"),
        )

        _, messages, current_turn = map_agui_messages_to_events(command)

        assert current_turn == []
        assert messages == [
            ModelRequest([TextInput("run tool")]),
            ModelResponse(
                None,
                tool_calls=ToolCallsEvent([ToolCallEvent(id="t1", name="do", arguments="{}")]),
            ),
            ToolResultsEvent([ToolResultEvent(parent_id="t1", result=ToolResult(["42"]))]),
        ]


class TestCurrentTurnSplit:
    def test_trailing_user_messages_split_from_history(self) -> None:
        command = _command(
            UserMessage(id="u1", content="first turn"),
            AssistantMessage(id="a1", content="reply"),
            UserMessage(id="u2", content="follow-up"),
        )

        prompt, messages, current_turn = map_agui_messages_to_events(command)

        assert prompt == []
        assert messages == [
            ModelRequest([TextInput("first turn")]),
            ModelResponse(ModelMessage("reply"), tool_calls=ToolCallsEvent([])),
        ]
        assert current_turn == [TextInput("follow-up")]

    def test_consecutive_trailing_user_messages_accumulate_in_current_turn(self) -> None:
        command = _command(
            UserMessage(id="u1", content="hello"),
            UserMessage(id="u2", content="and more"),
        )

        _, messages, current_turn = map_agui_messages_to_events(command)

        assert messages == []
        assert current_turn == [TextInput("hello"), TextInput("and more")]
