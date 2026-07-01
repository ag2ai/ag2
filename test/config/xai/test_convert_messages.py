# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest
from fast_depends.use import SerializerCls
from xai_sdk.chat import chat_pb2

from ag2 import ToolResult
from ag2.compact import CompactionSummary
from ag2.config.xai.events import XAIAssistantEvent
from ag2.config.xai.mappers import convert_messages
from ag2.events import (
    AudioInput,
    BinaryInput,
    BinaryType,
    DataInput,
    DocumentInput,
    FileIdInput,
    ImageInput,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolNotFoundEvent,
    ToolResultEvent,
    ToolResultsEvent,
    VideoInput,
)
from ag2.exceptions import ToolNotFoundError, UnsupportedInputError


def _content_texts(msg: chat_pb2.Message) -> list[str]:
    return [c.text for c in msg.content if c.HasField("text")]


def _content_kinds(msg: chat_pb2.Message) -> list[str]:
    return [c.WhichOneof("content") for c in msg.content]


class TestSystemPrompt:
    def test_no_system_messages_when_prompt_empty(self) -> None:
        result, replays = convert_messages([], [], SerializerCls)
        assert result == []
        assert replays == []

    def test_system_prompts_joined_with_newline(self) -> None:
        result, replays = convert_messages(["be helpful", "be terse"], [], SerializerCls)

        assert len(result) == 1
        assert result[0].role == chat_pb2.ROLE_SYSTEM
        assert _content_texts(result[0]) == ["be helpful\nbe terse"]
        assert replays == []


class TestUserContent:
    IMG_URL = "https://example.com/image.png"
    PNG = b"\x89PNG\r\n"

    def test_text_only_request(self) -> None:
        [msg], _ = convert_messages([], [ModelRequest([TextInput("hello")])], SerializerCls)

        assert msg.role == chat_pb2.ROLE_USER
        assert _content_texts(msg) == ["hello"]

    def test_multiple_text_parts(self) -> None:
        [msg], _ = convert_messages([], [ModelRequest([TextInput("hi"), TextInput("there")])], SerializerCls)

        assert _content_texts(msg) == ["hi", "there"]

    def test_data_input_serialized_via_serializer(self) -> None:
        [msg], _ = convert_messages([], [ModelRequest([DataInput({"key": "value"})])], SerializerCls)

        assert _content_texts(msg) == ['{"key":"value"}']

    def test_image_url(self) -> None:
        [msg], _ = convert_messages([], [ModelRequest([ImageInput(url=self.IMG_URL)])], SerializerCls)

        assert msg.role == chat_pb2.ROLE_USER
        # Image is a Content oneof — not text
        kinds = _content_kinds(msg)
        assert "image_url" in kinds

    def test_image_binary_base64(self) -> None:
        [msg], _ = convert_messages(
            [], [ModelRequest([ImageInput(data=self.PNG, media_type="image/png")])], SerializerCls
        )

        kinds = _content_kinds(msg)
        assert "image_url" in kinds

        # Confirm the data URL ended up in the proto
        b64 = base64.b64encode(self.PNG).decode()
        image_block = next(c for c in msg.content if c.WhichOneof("content") == "image_url")
        assert f"data:image/png;base64,{b64}" in image_block.image_url.image_url

    def test_pdf_binary_becomes_file(self) -> None:
        [msg], _ = convert_messages(
            [], [ModelRequest([DocumentInput(data=b"%PDF", media_type="application/pdf")])], SerializerCls
        )

        kinds = _content_kinds(msg)
        assert "file" in kinds
        file_block = next(c for c in msg.content if c.WhichOneof("content") == "file")
        assert file_block.file.data == b"%PDF"
        assert file_block.file.mime_type == "application/pdf"

    def test_text_plus_image_interleaved(self) -> None:
        [msg], _ = convert_messages(
            [],
            [ModelRequest([TextInput("describe"), ImageInput(url=self.IMG_URL)])],
            SerializerCls,
        )

        kinds = _content_kinds(msg)
        assert kinds == ["text", "image_url"]

    def test_audio_input_unsupported(self) -> None:
        with pytest.raises(UnsupportedInputError, match="xai"):
            convert_messages(
                [],
                [
                    ModelRequest([
                        BinaryInput(b"\x00", media_type="audio/wav", kind=BinaryType.AUDIO),
                    ])
                ],
                SerializerCls,
            )

    def test_video_input_unsupported(self) -> None:
        with pytest.raises(UnsupportedInputError, match="xai"):
            convert_messages(
                [],
                [ModelRequest([VideoInput(url="https://example.com/clip.mp4")])],
                SerializerCls,
            )

    def test_audio_url_input_unsupported(self) -> None:
        with pytest.raises(UnsupportedInputError, match="xai"):
            convert_messages(
                [],
                [ModelRequest([AudioInput(url="https://example.com/a.wav")])],
                SerializerCls,
            )

    def test_file_id_input_passes_through(self) -> None:
        [msg], _ = convert_messages([], [ModelRequest([FileIdInput(file_id="file-abc")])], SerializerCls)

        kinds = _content_kinds(msg)
        assert "file" in kinds
        file_block = next(c for c in msg.content if c.WhichOneof("content") == "file")
        assert file_block.file.file_id == "file-abc"


class TestToolResult:
    def test_text_only_tool_result(self) -> None:
        event = ToolResultsEvent(results=[ToolResultEvent(parent_id="tc_1", name="t", result=ToolResult("ok"))])

        [msg], _ = convert_messages([], [event], SerializerCls)

        assert msg.role == chat_pb2.ROLE_TOOL
        assert msg.tool_call_id == "tc_1"
        assert _content_texts(msg) == ["ok"]

    def test_multi_text_tool_result_serialises_to_json_array(self) -> None:
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="t",
                    result=ToolResult(TextInput("first"), TextInput("second")),
                )
            ]
        )

        [msg], _ = convert_messages([], [event], SerializerCls)

        assert _content_texts(msg) == ['["first", "second"]']

    def test_binary_in_tool_result_unsupported(self) -> None:
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="t",
                    result=ToolResult(ImageInput(url="https://x/img.png")),
                )
            ]
        )

        with pytest.raises(UnsupportedInputError, match="tool_result"):
            convert_messages([], [event], SerializerCls)

    def test_hallucinated_tool_call_maps_with_error_text(self) -> None:
        # Regression: a not-found tool call used to leave result=None and crash on r.result.parts.
        call = ToolCallEvent(id="tc_1", name="ghost_tool")
        event = ToolResultsEvent(results=[ToolNotFoundEvent.from_call(call, ToolNotFoundError("ghost_tool"))])

        [msg], _ = convert_messages([], [event], SerializerCls)

        assert msg.role == chat_pb2.ROLE_TOOL
        assert msg.tool_call_id == "tc_1"
        assert _content_texts(msg) == ["ag2.exceptions.ToolNotFoundError: Tool `ghost_tool` not found\n"]


class TestAssistantRoundTrip:
    def test_xai_assistant_event_round_trips_via_replay_list(self) -> None:
        proto = chat_pb2.GetChatCompletionResponse()
        event = XAIAssistantEvent(
            proto_bytes=proto.SerializeToString(),
            model="grok-4-fast",
            finish_reason="stop",
        )

        messages, replays = convert_messages([], [event], SerializerCls)

        assert messages == []
        assert len(replays) == 1
        # The replay is an xai-sdk Response wrapping the proto
        assert replays[0].proto.SerializeToString() == proto.SerializeToString()

    def test_companion_model_response_after_assistant_event_is_skipped(self) -> None:
        """An ``XAIAssistantEvent`` shadows the next ``ModelResponse`` for the same turn."""
        proto = chat_pb2.GetChatCompletionResponse()
        event = XAIAssistantEvent(proto_bytes=proto.SerializeToString())
        mr = ModelResponse(message=ModelMessage("Hello"))

        messages, replays = convert_messages([], [event, mr], SerializerCls)

        assert messages == []  # neither the proto nor the synthesized assistant
        assert len(replays) == 1

    def test_orphan_model_response_falls_back_to_assistant_text(self) -> None:
        """When no XAIAssistantEvent precedes it, ModelResponse becomes ``assistant(text)``."""
        mr = ModelResponse(message=ModelMessage("Hi"))

        [msg], replays = convert_messages([], [mr], SerializerCls)

        assert msg.role == chat_pb2.ROLE_ASSISTANT
        assert _content_texts(msg) == ["Hi"]
        assert replays == []


def test_compaction_summary_renders_as_user_turn() -> None:
    summary = CompactionSummary(summary="Looked up Paris and Tokyo.", event_count=6)

    [msg], replays = convert_messages([], [summary], SerializerCls)

    assert msg.role == chat_pb2.ROLE_USER
    assert _content_texts(msg) == ["[Summary of earlier conversation]\nLooked up Paris and Tokyo."]
    assert replays == []
