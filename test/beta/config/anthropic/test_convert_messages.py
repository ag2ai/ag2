# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest
from dirty_equals import IsPartialDict

from autogen.beta import ToolResult
from autogen.beta.config.anthropic.mappers import convert_messages
from autogen.beta.events import (
    AudioInput,
    BinaryInput,
    BinaryType,
    DocumentInput,
    FileIdInput,
    ImageInput,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
    VideoInput,
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


class TestImageUrlInput:
    IMAGE_URL = "https://example.com/image.png"

    def test_converts_to_image_url_block(self) -> None:
        result = convert_messages([ModelRequest([ImageInput(url=self.IMAGE_URL)])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "image", "source": {"type": "url", "url": self.IMAGE_URL}}],
            }
        ]


class TestImageBinaryInput:
    SAMPLE_BYTES = b"\x89PNG\r\n\x1a\nfake"

    def test_converts_to_image_base64_block(self) -> None:
        result = convert_messages([ModelRequest([ImageInput(data=self.SAMPLE_BYTES, media_type="image/png")])])

        expected_b64 = base64.b64encode(self.SAMPLE_BYTES).decode()
        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": expected_b64},
                    }
                ],
            }
        ]

    def test_vendor_metadata_cache_control_merges(self) -> None:
        result = convert_messages([
            ModelRequest([
                BinaryInput(
                    data=self.SAMPLE_BYTES,
                    media_type="image/png",
                    vendor_metadata={"cache_control": {"type": "ephemeral"}},
                    kind=BinaryType.IMAGE,
                )
            ])
        ])

        assert result == [
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "image", "cache_control": {"type": "ephemeral"}})],
            })
        ]

    def test_vendor_metadata_filename_filtered_out(self) -> None:
        result = convert_messages([
            ModelRequest([
                BinaryInput(
                    data=self.SAMPLE_BYTES,
                    media_type="image/png",
                    vendor_metadata={"filename": "photo.png"},
                    kind=BinaryType.IMAGE,
                )
            ])
        ])

        content = result[0]["content"][0]
        assert "filename" not in content


class TestDocumentUrlInput:
    DOC_URL = "https://example.com/doc.pdf"

    def test_converts_to_document_url_block(self) -> None:
        result = convert_messages([ModelRequest([DocumentInput(url=self.DOC_URL)])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "document", "source": {"type": "url", "url": self.DOC_URL}}],
            }
        ]


class TestDocumentBinaryInput:
    SAMPLE_BYTES = b"%PDF-1.4"

    def test_converts_to_document_base64_block(self) -> None:
        result = convert_messages([ModelRequest([DocumentInput(data=self.SAMPLE_BYTES, media_type="application/pdf")])])

        expected_b64 = base64.b64encode(self.SAMPLE_BYTES).decode()
        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {"type": "base64", "media_type": "application/pdf", "data": expected_b64},
                    }
                ],
            }
        ]

    def test_vendor_metadata_merges(self) -> None:
        result = convert_messages([
            ModelRequest([
                BinaryInput(
                    data=self.SAMPLE_BYTES,
                    media_type="application/pdf",
                    vendor_metadata={"cache_control": {"type": "ephemeral"}},
                    kind=BinaryType.DOCUMENT,
                )
            ])
        ])

        assert result == [
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "document", "cache_control": {"type": "ephemeral"}})],
            })
        ]


class TestFileIdInput:
    FILE_ID = "file_011CNha8iCJcU1wXNR6q4V8w"

    def test_no_filename_defaults_to_document(self) -> None:
        result = convert_messages([ModelRequest([FileIdInput(file_id=self.FILE_ID)])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "document", "source": {"type": "file", "file_id": self.FILE_ID}}],
            }
        ]

    def test_image_filename_uses_image_block(self) -> None:
        result = convert_messages([ModelRequest([FileIdInput(file_id=self.FILE_ID, filename="photo.jpg")])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "image", "source": {"type": "file", "file_id": self.FILE_ID}}],
            }
        ]

    def test_pdf_filename_uses_document_block(self) -> None:
        result = convert_messages([ModelRequest([FileIdInput(file_id=self.FILE_ID, filename="report.pdf")])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "document", "source": {"type": "file", "file_id": self.FILE_ID}}],
            }
        ]


class TestMultipleInputs:
    def test_multiple_inputs_grouped_into_one_message(self) -> None:
        result = convert_messages([
            ModelRequest([
                TextInput("Describe these images."),
                ImageInput(url="https://example.com/a.png"),
                ImageInput(url="https://example.com/b.jpg"),
            ])
        ])

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 3
        assert result[0]["content"][0] == {"type": "text", "text": "Describe these images."}
        assert result[0]["content"][1] == IsPartialDict({"type": "image"})
        assert result[0]["content"][2] == IsPartialDict({"type": "image"})


class TestUnsupportedInputs:
    def test_audio_url_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="UrlInput.*audio.*anthropic"):
            convert_messages([ModelRequest([AudioInput(url="https://example.com/audio.wav")])])

    def test_video_url_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="UrlInput.*video.*anthropic"):
            convert_messages([ModelRequest([VideoInput(url="https://example.com/video.mp4")])])

    def test_audio_binary_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="BinaryInput.*audio.*anthropic"):
            convert_messages([ModelRequest([AudioInput(data=b"\x00audio", media_type="audio/wav")])])

    def test_video_binary_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="BinaryInput.*video.*anthropic"):
            convert_messages([ModelRequest([VideoInput(data=b"\x00video", media_type="video/mp4")])])

    def test_generic_binary_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="BinaryInput.*binary.*anthropic"):
            convert_messages([
                ModelRequest([BinaryInput(data=b"\x00", media_type="application/octet-stream", kind=BinaryType.BINARY)])
            ])
