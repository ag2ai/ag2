# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from base64 import b64encode
from typing import Any

import pytest
from ag_ui.core import (
    AssistantMessage,
    AudioPart,
    DataSource,
    DocumentPart,
    FileSource,
    FunctionCall,
    ImagePart,
    ReasoningMessage,
    SystemMessage,
    TextPart,
    ToolCall,
    ToolMessage,
    UrlSource,
    UserMessage,
    VideoPart,
)

from ag2 import ToolResult
from ag2.ag_ui.stream import AGStreamInput, map_agui_messages_to_events
from ag2.events import (
    AudioInput,
    BinaryInput,
    BinaryType,
    DocumentInput,
    FileIdInput,
    ImageInput,
    ModelMessage,
    ModelReasoning,
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
from test.ag_ui.harness import run_input

RAW_BYTES = b"\xff\xd8\xff\xe0"
B64_VALUE = b64encode(RAW_BYTES).decode()


def _command(*messages: object) -> AGStreamInput:
    """The command `run_stream` builds from an AG-UI request, for `messages`."""
    return AGStreamInput(incoming=run_input(*messages), variables={})


class TestUserMessageString:
    def test_plain_string_becomes_text_input(self) -> None:
        command = _command(UserMessage(id="m1", content="hello"))

        prompt, messages, current_turn = map_agui_messages_to_events(command)

        assert prompt == []
        assert messages == []
        assert current_turn == [TextInput("hello")]

    def test_text_content_becomes_text_input(self) -> None:
        command = _command(UserMessage(id="m1", content=[TextPart(text="hi")]))

        _, messages, current_turn = map_agui_messages_to_events(command)

        assert messages == []
        assert current_turn == [TextInput("hi")]


@pytest.mark.parametrize(
    "content_cls,factory,kind,mime",
    [
        (ImagePart, ImageInput, BinaryType.IMAGE, "image/jpeg"),
        (AudioPart, AudioInput, BinaryType.AUDIO, "audio/wav"),
        (VideoPart, VideoInput, BinaryType.VIDEO, "video/mp4"),
        (DocumentPart, DocumentInput, BinaryType.DOCUMENT, "application/pdf"),
    ],
)
class TestMediaContentMapping:
    def test_url_source_maps_to_url_input(self, content_cls: type, factory: Any, kind: BinaryType, mime: str) -> None:
        url = "https://example.com/file"
        content = content_cls(source=UrlSource(value=url))
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
        content = content_cls(source=DataSource(value=B64_VALUE, mime_type=mime))
        command = _command(UserMessage(id="m1", content=[content]))

        _, messages, current_turn = map_agui_messages_to_events(command)

        assert messages == []
        assert current_turn == [factory(data=RAW_BYTES, media_type=mime)]
        [part] = current_turn
        assert isinstance(part, BinaryInput)
        assert part.kind == kind


class TestMetadata:
    def test_content_metadata_propagates_to_input_metadata(self) -> None:
        command = _command(
            UserMessage(
                id="m1",
                content=[
                    TextPart(text="hi"),
                    ImagePart(
                        source=UrlSource(value="https://x/i.png"),
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

    def test_content_metadata_survives_base64_decoding(self) -> None:
        """A part carried as data, not as a URL, keeps its metadata through the decode."""
        command = _command(
            UserMessage(
                id="m1",
                content=[
                    DocumentPart(
                        source=DataSource(value=B64_VALUE, mime_type="application/pdf"),
                        metadata={"source_filename": "report.pdf"},
                    ),
                ],
            )
        )

        _, _, current_turn = map_agui_messages_to_events(command)

        expected = DocumentInput(data=RAW_BYTES, media_type="application/pdf")
        expected.metadata = {"source_filename": "report.pdf"}
        assert current_turn == [expected]


class TestReasoningMessages:
    """What the model was thinking last turn is history, not part of this turn."""

    def test_reasoning_message_becomes_a_model_reasoning_event(self) -> None:
        command = _command(
            UserMessage(id="u1", content="Hi"),
            ReasoningMessage(id="r1", content="user is greeting me"),
        )

        _, messages, current_turn = map_agui_messages_to_events(command)

        assert current_turn == []
        assert messages == [
            ModelRequest([TextInput("Hi")]),
            ModelReasoning("user is greeting me"),
        ]

    def test_an_empty_reasoning_message_is_dropped(self) -> None:
        """Rather than handed to the LLM as a thought with nothing in it."""
        command = _command(
            UserMessage(id="u1", content="Hi"),
            ReasoningMessage(id="r1", content=""),
        )

        _, messages, _ = map_agui_messages_to_events(command)

        assert messages == [ModelRequest([TextInput("Hi")])]


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
            ToolResultsEvent([ToolResultEvent(parent_id="t1", result=ToolResult("42"))]),
        ]

    def test_a_tool_message_in_parts_becomes_a_tool_result_of_those_parts(self) -> None:
        """A screenshot beside its caption reaches the model as both, not as a string."""
        command = _command(
            ToolMessage(
                id="tm1",
                tool_call_id="t1",
                content=[
                    TextPart(text="the page"),
                    ImagePart(source=DataSource(value=B64_VALUE, mime_type="image/jpeg")),
                ],
            ),
        )

        _, messages, _ = map_agui_messages_to_events(command)

        assert messages == [
            ToolResultsEvent([
                ToolResultEvent(
                    parent_id="t1",
                    result=ToolResult(TextInput("the page"), ImageInput(data=RAW_BYTES, media_type="image/jpeg")),
                )
            ]),
        ]

    def test_a_tool_message_s_error_is_what_the_model_hears(self) -> None:
        command = _command(
            ToolMessage(id="tm1", tool_call_id="t1", content=[TextPart(text="partial")], error="it broke"),
        )

        _, messages, _ = map_agui_messages_to_events(command)

        assert messages == [
            ToolResultsEvent([ToolResultEvent(parent_id="t1", result=ToolResult("it broke"))]),
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


class TestProviderFileHandles:
    """A handle only the provider that minted it can resolve, and never fetched or parsed."""

    def test_an_untagged_handle_is_taken_to_be_the_run_s_own(self) -> None:
        command = _command(UserMessage(id="m1", content=[DocumentPart(source=FileSource(value="file-abc"))]))

        _, _, current_turn = map_agui_messages_to_events(command, provider="anthropic")

        assert current_turn == [FileIdInput("file-abc")]

    def test_a_handle_tagged_with_the_run_s_provider_reaches_it(self) -> None:
        command = _command(
            UserMessage(id="m1", content=[ImagePart(source=FileSource(value="file-abc", provider="anthropic"))])
        )

        _, _, current_turn = map_agui_messages_to_events(command, provider="anthropic")

        assert current_turn == [FileIdInput("file-abc")]

    def test_another_provider_s_handle_is_skipped_and_said_so(self, caplog: pytest.LogCaptureFixture) -> None:
        command = _command(
            UserMessage(
                id="m1",
                content=[
                    TextPart(text="look"),
                    DocumentPart(source=FileSource(value="file-secret", provider="openai")),
                ],
            )
        )

        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            _, _, current_turn = map_agui_messages_to_events(command, provider="anthropic")

        assert current_turn == [TextInput("look")]
        [record] = caplog.records
        assert "openai" in record.getMessage()
        assert "anthropic" in record.getMessage()
        assert "file-secret" not in record.getMessage()

    def test_a_skipped_handle_in_a_tool_message_leaves_the_rest_of_the_result(self) -> None:
        command = _command(
            ToolMessage(
                id="tm1",
                tool_call_id="t1",
                content=[TextPart(text="done"), DocumentPart(source=FileSource(value="f", provider="openai"))],
            ),
        )

        _, messages, _ = map_agui_messages_to_events(command, provider="gemini")

        assert messages == [
            ToolResultsEvent([ToolResultEvent(parent_id="t1", result=ToolResult("done"))]),
        ]
