# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls

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
from autogen.beta.files.types import FileProvider, UploadedFile


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
        result = convert_messages([response], SerializerCls)

        assert result == [
            IsPartialDict({
                "role": "assistant",
                "content": [IsPartialDict({"type": "tool_use", "id": "tc_1", "name": "list_items", "input": {}})],
            }),
        ]

    def test_valid_arguments_are_preserved(self) -> None:
        response = _model_response_with_tool_call('{"category": "books"}')
        result = convert_messages([response], SerializerCls)

        assert result == [
            IsPartialDict({
                "content": [IsPartialDict({"type": "tool_use", "input": {"category": "books"}})],
            }),
        ]

    def test_empty_object_arguments(self) -> None:
        response = _model_response_with_tool_call("{}")
        result = convert_messages([response], SerializerCls)

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
                    result=ToolResult("apple, banana"),
                )
            ],
        ),
    ]
    result = convert_messages(events, SerializerCls)

    assert result[0] == IsPartialDict({"role": "user"})
    assert result[1] == IsPartialDict({
        "role": "assistant",
        "content": [IsPartialDict({"input": {}})],
    })
    assert result[2] == IsPartialDict({
        "role": "user",
        "content": [IsPartialDict({"type": "tool_result"})],
    })


def test_image_url_input_converts_to_url_block() -> None:
    image_url = "https://example.com/image.png"
    result = convert_messages([ModelRequest([ImageInput(url=image_url)])], SerializerCls)

    assert result == [
        {
            "role": "user",
            "content": [{"type": "image", "source": {"type": "url", "url": image_url}}],
        }
    ]


class TestImageBinaryInput:
    SAMPLE_BYTES = b"\x89PNG\r\n\x1a\nfake"

    def test_converts_to_image_base64_block(self) -> None:
        result = convert_messages(
            [ModelRequest([ImageInput(data=self.SAMPLE_BYTES, media_type="image/png")])], SerializerCls
        )

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
        result = convert_messages(
            [
                ModelRequest([
                    BinaryInput(
                        data=self.SAMPLE_BYTES,
                        media_type="image/png",
                        vendor_metadata={"cache_control": {"type": "ephemeral"}},
                        kind=BinaryType.IMAGE,
                    )
                ])
            ],
            SerializerCls,
        )

        assert result == [
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "image", "cache_control": {"type": "ephemeral"}})],
            })
        ]

    def test_vendor_metadata_filename_filtered_out(self) -> None:
        result = convert_messages(
            [
                ModelRequest([
                    BinaryInput(
                        data=self.SAMPLE_BYTES,
                        media_type="image/png",
                        vendor_metadata={"filename": "photo.png"},
                        kind=BinaryType.IMAGE,
                    )
                ])
            ],
            SerializerCls,
        )

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


def test_document_url_input_converts_to_url_block() -> None:
    doc_url = "https://example.com/doc.pdf"
    result = convert_messages([ModelRequest([DocumentInput(url=doc_url)])], SerializerCls)

    assert result == [
        {
            "role": "user",
            "content": [{"type": "document", "source": {"type": "url", "url": doc_url}}],
        }
    ]


class TestDocumentBinaryInput:
    SAMPLE_BYTES = b"%PDF-1.4"

    def test_converts_to_document_base64_block(self) -> None:
        result = convert_messages(
            [ModelRequest([DocumentInput(data=self.SAMPLE_BYTES, media_type="application/pdf")])], SerializerCls
        )

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
        result = convert_messages(
            [
                ModelRequest([
                    BinaryInput(
                        data=self.SAMPLE_BYTES,
                        media_type="application/pdf",
                        vendor_metadata={"cache_control": {"type": "ephemeral"}},
                        kind=BinaryType.DOCUMENT,
                    )
                ])
            ],
            SerializerCls,
        )

        assert result == [
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "document", "cache_control": {"type": "ephemeral"}})],
            })
        ]


class TestFileIdInput:
    FILE_ID = "file_011CNha8iCJcU1wXNR6q4V8w"

    def test_no_filename_defaults_to_document(self) -> None:
        result = convert_messages([ModelRequest([FileIdInput(file_id=self.FILE_ID)])], SerializerCls)

        assert result == [
            {
                "role": "user",
                "content": [{"type": "document", "source": {"type": "file", "file_id": self.FILE_ID}}],
            }
        ]

    def test_image_filename_uses_image_block(self) -> None:
        result = convert_messages(
            [ModelRequest([FileIdInput(file_id=self.FILE_ID, filename="photo.jpg")])], SerializerCls
        )

        assert result == [
            {
                "role": "user",
                "content": [{"type": "image", "source": {"type": "file", "file_id": self.FILE_ID}}],
            }
        ]

    def test_pdf_filename_uses_document_block(self) -> None:
        result = convert_messages(
            [ModelRequest([FileIdInput(file_id=self.FILE_ID, filename="report.pdf")])], SerializerCls
        )

        assert result == [
            {
                "role": "user",
                "content": [{"type": "document", "source": {"type": "file", "file_id": self.FILE_ID}}],
            }
        ]

    def test_foreign_provider_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="'openai'.*anthropic"):
            convert_messages(
                [ModelRequest([UploadedFile(file_id="file-abc", provider=FileProvider.OPENAI)])],
                SerializerCls,
            )

    def test_matching_provider_passes(self) -> None:
        result = convert_messages(
            [ModelRequest([UploadedFile(file_id=self.FILE_ID, provider=FileProvider.ANTHROPIC)])],
            SerializerCls,
        )

        assert result == [
            {
                "role": "user",
                "content": [{"type": "document", "source": {"type": "file", "file_id": self.FILE_ID}}],
            }
        ]


def test_multiple_inputs_grouped_into_one_message() -> None:
    result = convert_messages(
        [
            ModelRequest([
                TextInput("Describe these images."),
                ImageInput(url="https://example.com/a.png"),
                ImageInput(url="https://example.com/b.jpg"),
            ])
        ],
        SerializerCls,
    )

    assert result == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe these images."},
                {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
                {"type": "image", "source": {"type": "url", "url": "https://example.com/b.jpg"}},
            ],
        }
    ]


class TestToolResult:
    """Tool results can include image content blocks (binary + url)."""

    PNG = b"\x89PNG\r\n"

    def test_text_only_stays_string(self) -> None:
        event = ToolResultsEvent(results=[ToolResultEvent(parent_id="tc_1", name="t", result=ToolResult("hello"))])
        result = convert_messages([event], SerializerCls)

        assert result == [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tc_1", "content": "hello"}],
            }
        ]

    def test_binary_image(self) -> None:
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="t",
                    result=ToolResult(ImageInput(data=self.PNG, media_type="image/png")),
                )
            ]
        )
        result = convert_messages([event], SerializerCls)

        expected_b64 = base64.b64encode(self.PNG).decode()
        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tc_1",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/png", "data": expected_b64},
                            }
                        ],
                    }
                ],
            }
        ]

    def test_url_image(self) -> None:
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="t",
                    result=ToolResult(ImageInput(url="https://example.com/a.png")),
                )
            ]
        )
        result = convert_messages([event], SerializerCls)

        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tc_1",
                        "content": [{"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}],
                    }
                ],
            }
        ]

    def test_mixed_text_and_image(self) -> None:
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="t",
                    result=ToolResult("here you go", ImageInput(data=self.PNG, media_type="image/png")),
                )
            ]
        )
        result = convert_messages([event], SerializerCls)

        assert result == [
            IsPartialDict({
                "role": "user",
                "content": [
                    IsPartialDict({
                        "type": "tool_result",
                        "tool_use_id": "tc_1",
                        "content": [
                            {"type": "text", "text": "here you go"},
                            IsPartialDict({"type": "image"}),
                        ],
                    })
                ],
            })
        ]

    def test_binary_document(self) -> None:
        pdf = b"%PDF-1.4 fake"
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="t",
                    result=ToolResult(DocumentInput(data=pdf, media_type="application/pdf")),
                )
            ]
        )
        result = convert_messages([event], SerializerCls)

        expected_b64 = base64.b64encode(pdf).decode()
        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tc_1",
                        "content": [
                            {
                                "type": "document",
                                "source": {"type": "base64", "media_type": "application/pdf", "data": expected_b64},
                            }
                        ],
                    }
                ],
            }
        ]

    def test_url_document(self) -> None:
        doc_url = "https://example.com/report.pdf"
        event = ToolResultsEvent(
            results=[ToolResultEvent(parent_id="tc_1", name="t", result=ToolResult(DocumentInput(url=doc_url)))]
        )
        result = convert_messages([event], SerializerCls)

        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tc_1",
                        "content": [{"type": "document", "source": {"type": "url", "url": doc_url}}],
                    }
                ],
            }
        ]

    def test_file_id_image(self) -> None:
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="t",
                    result=ToolResult(FileIdInput(file_id="file_abc", filename="photo.jpg")),
                )
            ]
        )
        result = convert_messages([event], SerializerCls)

        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tc_1",
                        "content": [{"type": "image", "source": {"type": "file", "file_id": "file_abc"}}],
                    }
                ],
            }
        ]

    def test_file_id_document(self) -> None:
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="t",
                    result=ToolResult(FileIdInput(file_id="file_xyz", filename="report.pdf")),
                )
            ]
        )
        result = convert_messages([event], SerializerCls)

        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tc_1",
                        "content": [{"type": "document", "source": {"type": "file", "file_id": "file_xyz"}}],
                    }
                ],
            }
        ]

    def test_file_id_no_filename_defaults_to_document(self) -> None:
        event = ToolResultsEvent(
            results=[ToolResultEvent(parent_id="tc_1", name="t", result=ToolResult(FileIdInput(file_id="file_plain")))]
        )
        result = convert_messages([event], SerializerCls)

        assert result == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tc_1",
                        "content": [{"type": "document", "source": {"type": "file", "file_id": "file_plain"}}],
                    }
                ],
            }
        ]

    def test_audio_in_tool_result_raises(self) -> None:
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="t",
                    result=ToolResult(AudioInput(data=b"\x00audio", media_type="audio/wav")),
                )
            ]
        )
        with pytest.raises(UnsupportedInputError, match="BinaryInput.*audio.*anthropic"):
            convert_messages([event], SerializerCls)


class TestUnsupportedInputs:
    def test_audio_url_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="UrlInput.*audio.*anthropic"):
            convert_messages([ModelRequest([AudioInput(url="https://example.com/audio.wav")])], SerializerCls)

    def test_video_url_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="UrlInput.*video.*anthropic"):
            convert_messages([ModelRequest([VideoInput(url="https://example.com/video.mp4")])], SerializerCls)

    def test_audio_binary_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="BinaryInput.*audio.*anthropic"):
            convert_messages([ModelRequest([AudioInput(data=b"\x00audio", media_type="audio/wav")])], SerializerCls)

    def test_video_binary_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="BinaryInput.*video.*anthropic"):
            convert_messages([ModelRequest([VideoInput(data=b"\x00video", media_type="video/mp4")])], SerializerCls)

    def test_generic_binary_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="BinaryInput.*binary.*anthropic"):
            convert_messages(
                [
                    ModelRequest([
                        BinaryInput(data=b"\x00", media_type="application/octet-stream", kind=BinaryType.BINARY)
                    ])
                ],
                SerializerCls,
            )
