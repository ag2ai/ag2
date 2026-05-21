# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest
from fast_depends.use import SerializerCls

from autogen.beta import ToolResult
from autogen.beta.config.dashscope.mappers import convert_messages
from autogen.beta.events import (
    AudioInput,
    BinaryInput,
    DocumentInput,
    FileIdInput,
    ImageInput,
    ModelRequest,
    TextInput,
    ToolResultEvent,
    ToolResultsEvent,
)
from autogen.beta.exceptions import UnsupportedInputError


def test_audio_url_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="UrlInput.*dashscope"):
        convert_messages([], [ModelRequest([AudioInput(url="https://example.com/audio.wav")])], SerializerCls)


def test_document_url_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="UrlInput.*dashscope"):
        convert_messages([], [ModelRequest([DocumentInput(url="https://example.com/doc.pdf")])], SerializerCls)


def test_file_id_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="FileIdInput.*dashscope"):
        convert_messages([], [ModelRequest([FileIdInput(file_id="file-abc123")])], SerializerCls)


def test_non_image_binary_raises() -> None:
    """BinaryInput with the default (binary) kind is still unsupported."""
    with pytest.raises(UnsupportedInputError, match="BinaryInput.*dashscope"):
        convert_messages(
            [],
            [ModelRequest([BinaryInput(data=b"data", media_type="application/octet-stream")])],
            SerializerCls,
        )


class TestQwenVLImage:
    """Qwen-VL multimodal content block format."""

    IMG_URL = "https://example.com/image.png"
    PNG = b"\x89PNG\r\n"

    def test_text_only_stays_string(self) -> None:
        result = convert_messages([], [ModelRequest([TextInput("hello")])], SerializerCls)

        assert result == [{"role": "user", "content": "hello"}]

    def test_image_url(self) -> None:
        result = convert_messages([], [ModelRequest([ImageInput(url=self.IMG_URL)])], SerializerCls)

        assert result == [{"role": "user", "content": [{"image": self.IMG_URL}]}]

    def test_image_binary(self) -> None:
        result = convert_messages(
            [], [ModelRequest([ImageInput(data=self.PNG, media_type="image/png")])], SerializerCls
        )

        b64 = base64.b64encode(self.PNG).decode()
        assert result == [{"role": "user", "content": [{"image": f"data:image/png;base64,{b64}"}]}]

    def test_text_plus_image(self) -> None:
        result = convert_messages(
            [], [ModelRequest([TextInput("describe"), ImageInput(url=self.IMG_URL)])], SerializerCls
        )

        assert result == [{"role": "user", "content": [{"text": "describe"}, {"image": self.IMG_URL}]}]


class TestToolResult:
    """Tool results support text and image blocks (DashScope multimodal tool messages)."""

    IMG_URL = "https://example.com/image.png"
    PNG = b"\x89PNG\r\n"

    def test_text_only_stays_string(self) -> None:
        event = ToolResultsEvent(results=[ToolResultEvent(parent_id="tc_1", name="t", result=ToolResult("hello"))])
        result = convert_messages([], [event], SerializerCls)

        assert result == [{"role": "tool", "tool_call_id": "tc_1", "content": "hello"}]

    def test_image_url(self) -> None:
        event = ToolResultsEvent(
            results=[ToolResultEvent(parent_id="tc_1", name="t", result=ToolResult(ImageInput(url=self.IMG_URL)))]
        )
        result = convert_messages([], [event], SerializerCls)

        assert result == [{"role": "tool", "tool_call_id": "tc_1", "content": [{"image": self.IMG_URL}]}]

    def test_image_binary(self) -> None:
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="t",
                    result=ToolResult(ImageInput(data=self.PNG, media_type="image/png")),
                )
            ]
        )
        result = convert_messages([], [event], SerializerCls)

        b64 = base64.b64encode(self.PNG).decode()
        assert result == [
            {"role": "tool", "tool_call_id": "tc_1", "content": [{"image": f"data:image/png;base64,{b64}"}]}
        ]

    def test_mixed_text_and_image(self) -> None:
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="t",
                    result=ToolResult("here", ImageInput(url=self.IMG_URL)),
                )
            ]
        )
        result = convert_messages([], [event], SerializerCls)

        assert result == [
            {
                "role": "tool",
                "tool_call_id": "tc_1",
                "content": [{"text": "here"}, {"image": self.IMG_URL}],
            }
        ]

    def test_document_in_tool_result_raises(self) -> None:
        """DashScope Generation tool message accepts only text and images."""
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="t",
                    result=ToolResult(DocumentInput(data=b"%PDF", media_type="application/pdf")),
                )
            ]
        )
        with pytest.raises(UnsupportedInputError, match="BinaryInput.*dashscope"):
            convert_messages([], [event], SerializerCls)
