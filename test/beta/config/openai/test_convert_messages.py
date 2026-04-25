# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest
from dirty_equals import IsPartialDict
from fast_depends.use import SerializerCls

from autogen.beta import ToolResult
from autogen.beta.config.openai.mappers import convert_messages, events_to_responses_input
from autogen.beta.events import (
    AudioInput,
    BinaryInput,
    BinaryType,
    DocumentInput,
    FileIdInput,
    ImageInput,
    ModelRequest,
    TextInput,
    ToolResultEvent,
    ToolResultsEvent,
)
from autogen.beta.events.input_events import DataInput
from autogen.beta.exceptions import UnsupportedInputError
from autogen.beta.files.types import FileProvider, UploadedFile


class TestTextInput:
    def test_completions(self) -> None:
        result = convert_messages([], [ModelRequest([TextInput("hello")])], SerializerCls)

        assert result[1] == {"role": "user", "content": "hello"}

    def test_responses(self) -> None:
        result = events_to_responses_input([ModelRequest([TextInput("hello")])], SerializerCls)

        assert result == [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]

    def test_completions_text_with_image_url(self) -> None:
        """Text + image in one ModelRequest must produce a single message with content array."""
        image_url = "https://example.com/image.png"
        result = convert_messages(
            [],
            [
                ModelRequest([TextInput("describe this"), ImageInput(url=image_url)]),
            ],
            SerializerCls,
        )

        assert result == [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]


class TestImageUrlInput:
    IMAGE_URL = "https://example.com/image.png"

    def test_completions(self) -> None:
        result = convert_messages([], [ModelRequest([ImageInput(url=self.IMAGE_URL)])], SerializerCls)

        assert result[1] == {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": self.IMAGE_URL}}],
        }

    def test_responses(self) -> None:
        result = events_to_responses_input([ModelRequest([ImageInput(url=self.IMAGE_URL)])], SerializerCls)

        assert result == [
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": self.IMAGE_URL}],
            }
        ]


class TestFileIdInput:
    FILE_ID = "file-6F2ksmvXxt4VdoqmHRw6kL"

    def test_completions(self) -> None:
        result = convert_messages([], [ModelRequest([FileIdInput(file_id=self.FILE_ID)])], SerializerCls)

        assert result[1] == {
            "role": "user",
            "content": [{"type": "file", "file": {"file_id": self.FILE_ID}}],
        }

    def test_completions_filename_forbidden_with_file_id(self) -> None:
        """Chat Completions API rejects `filename` when `file_id` is present."""
        result = convert_messages(
            [], [ModelRequest([FileIdInput(file_id=self.FILE_ID, filename="report.pdf")])], SerializerCls
        )

        assert result[1] == {
            "role": "user",
            "content": [{"type": "file", "file": {"file_id": self.FILE_ID}}],
        }

    def test_responses(self) -> None:
        result = events_to_responses_input([ModelRequest([FileIdInput(file_id=self.FILE_ID)])], SerializerCls)

        assert result == [
            {
                "role": "user",
                "content": [{"type": "input_file", "file_id": self.FILE_ID}],
            }
        ]

    def test_responses_with_filename_ignores_filename(self) -> None:
        # OpenAI Responses API rejects file_id + filename together (mutually exclusive).
        result = events_to_responses_input(
            [ModelRequest([FileIdInput(file_id=self.FILE_ID, filename="report.pdf")])], SerializerCls
        )

        assert result == [
            {
                "role": "user",
                "content": [{"type": "input_file", "file_id": self.FILE_ID}],
            }
        ]

    def test_responses_foreign_provider_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="'anthropic'.*openai"):
            events_to_responses_input(
                [ModelRequest([UploadedFile(file_id="file_011CNha8", provider=FileProvider.ANTHROPIC)])],
                SerializerCls,
            )

    def test_responses_matching_provider_passes(self) -> None:
        result = events_to_responses_input(
            [ModelRequest([UploadedFile(file_id=self.FILE_ID, provider=FileProvider.OPENAI)])],
            SerializerCls,
        )

        assert result == [
            {
                "role": "user",
                "content": [{"type": "input_file", "file_id": self.FILE_ID}],
            }
        ]


class TestAudioUrlInput:
    AUDIO_URL = "https://example.com/audio.wav"

    def test_completions_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="UrlInput.*audio.*openai-completions"):
            convert_messages([], [ModelRequest([AudioInput(url=self.AUDIO_URL)])], SerializerCls)

    def test_responses_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="UrlInput.*audio.*openai-responses"):
            events_to_responses_input([ModelRequest([AudioInput(url=self.AUDIO_URL)])], SerializerCls)


class TestAudioBinaryInput:
    SAMPLE_BYTES = b"\x00\x01\x02audio"

    def test_completions(self) -> None:
        result = convert_messages(
            [], [ModelRequest([AudioInput(data=self.SAMPLE_BYTES, media_type="audio/wav")])], SerializerCls
        )

        expected_b64 = base64.b64encode(self.SAMPLE_BYTES).decode()
        assert result[1] == {
            "role": "user",
            "content": [{"type": "input_audio", "input_audio": {"data": expected_b64, "format": "wav"}}],
        }

    def test_completions_mp3(self) -> None:
        result = convert_messages(
            [], [ModelRequest([AudioInput(data=self.SAMPLE_BYTES, media_type="audio/mpeg")])], SerializerCls
        )

        expected_b64 = base64.b64encode(self.SAMPLE_BYTES).decode()
        assert result[1] == {
            "role": "user",
            "content": [{"type": "input_audio", "input_audio": {"data": expected_b64, "format": "mp3"}}],
        }


class TestBinaryInput:
    SAMPLE_BYTES = b"\x89PNG\r\n\x1a\nfake"

    def test_completions(self) -> None:
        result = convert_messages(
            [], [ModelRequest([ImageInput(data=self.SAMPLE_BYTES, media_type="image/png")])], SerializerCls
        )

        expected_url = f"data:image/png;base64,{base64.b64encode(self.SAMPLE_BYTES).decode()}"
        assert result[1] == {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": expected_url}}],
        }

    def test_completions_with_vendor_metadata(self) -> None:
        result = convert_messages(
            [],
            [
                ModelRequest([
                    BinaryInput(
                        data=self.SAMPLE_BYTES,
                        media_type="image/png",
                        vendor_metadata={"detail": "low"},
                        kind=BinaryType.IMAGE,
                    )
                ])
            ],
            SerializerCls,
        )

        assert result[1] == IsPartialDict({
            "role": "user",
            "content": [IsPartialDict({"type": "image_url", "detail": "low"})],
        })

    def test_responses(self) -> None:
        result = events_to_responses_input(
            [ModelRequest([BinaryInput(data=self.SAMPLE_BYTES, media_type="image/png")])],
            SerializerCls,
        )

        expected_data = f"data:image/png;base64,{base64.b64encode(self.SAMPLE_BYTES).decode()}"
        assert result == [
            {
                "role": "user",
                "content": [{"type": "input_file", "file_data": expected_data}],
            }
        ]

    def test_responses_with_vendor_metadata(self) -> None:
        result = events_to_responses_input(
            [
                ModelRequest([
                    BinaryInput(
                        data=self.SAMPLE_BYTES, media_type="image/png", vendor_metadata={"filename": "test.png"}
                    )
                ])
            ],
            SerializerCls,
        )

        assert result == [
            IsPartialDict({
                "role": "user",
                "content": [IsPartialDict({"type": "input_file", "filename": "test.png"})],
            })
        ]


class TestDocumentUrlInput:
    DOC_URL = "https://example.com/document.pdf"

    def test_completions_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="UrlInput.*document.*openai-completions"):
            convert_messages([], [ModelRequest([DocumentInput(url=self.DOC_URL)])], SerializerCls)

    def test_responses(self) -> None:
        result = events_to_responses_input([ModelRequest([DocumentInput(url=self.DOC_URL)])], SerializerCls)

        assert result == [
            {
                "role": "user",
                "content": [{"type": "input_file", "file_url": self.DOC_URL}],
            }
        ]


class TestDocumentBinaryInput:
    SAMPLE_BYTES = b"%PDF-1.4 fake"

    def test_completions_infers_filename_from_media_type(self) -> None:
        result = convert_messages(
            [],
            [ModelRequest([DocumentInput(data=self.SAMPLE_BYTES, media_type="application/pdf")])],
            SerializerCls,
        )

        expected_data = f"data:application/pdf;base64,{base64.b64encode(self.SAMPLE_BYTES).decode()}"
        assert result[1] == {
            "role": "user",
            "content": [{"type": "file", "file": {"file_data": expected_data, "filename": "file.pdf"}}],
        }

    def test_completions_uses_vendor_metadata_filename(self) -> None:
        result = convert_messages(
            [],
            [
                ModelRequest([
                    BinaryInput(
                        data=self.SAMPLE_BYTES,
                        media_type="application/pdf",
                        vendor_metadata={"filename": "report.pdf"},
                        kind=BinaryType.DOCUMENT,
                    )
                ])
            ],
            SerializerCls,
        )

        expected_data = f"data:application/pdf;base64,{base64.b64encode(self.SAMPLE_BYTES).decode()}"
        assert result[1] == {
            "role": "user",
            "content": [{"type": "file", "file": {"file_data": expected_data, "filename": "report.pdf"}}],
        }


def test_data_input_in_model_request_becomes_input_text() -> None:
    """DataInput must be serialized to input_text in Responses API ModelRequest."""
    result = events_to_responses_input([ModelRequest([DataInput({"key": "value"})])], SerializerCls)

    assert result == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": '{"key":"value"}'}],
        }
    ]


class TestResponsesToolResult:
    """Tool-result output supports text, image, file and file_id parts."""

    def test_text_only_stays_string(self) -> None:
        event = ToolResultsEvent(results=[ToolResultEvent(parent_id="c1", name="t", result=ToolResult("hello"))])
        result = events_to_responses_input([event], SerializerCls)

        assert result == [{"type": "function_call_output", "call_id": "c1", "output": "hello"}]

    def test_image_binary_becomes_input_image_block(self) -> None:
        png = b"\x89PNG\r\n"
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="c1",
                    name="t",
                    result=ToolResult(ImageInput(data=png, media_type="image/png")),
                )
            ]
        )
        result = events_to_responses_input([event], SerializerCls)

        expected_url = f"data:image/png;base64,{base64.b64encode(png).decode()}"
        assert result == [
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_image", "image_url": expected_url}],
            }
        ]

    def test_image_url_becomes_input_image_block(self) -> None:
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="c1",
                    name="t",
                    result=ToolResult(ImageInput(url="https://example.com/a.png")),
                )
            ]
        )
        result = events_to_responses_input([event], SerializerCls)

        assert result == [
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_image", "image_url": "https://example.com/a.png"}],
            }
        ]

    def test_document_url_becomes_input_file_url_block(self) -> None:
        """UrlInput(DOCUMENT) → input_file with file_url; filename is forbidden by the API here."""
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="c1",
                    name="t",
                    result=ToolResult(DocumentInput(url="https://example.com/d.pdf")),
                )
            ]
        )
        result = events_to_responses_input([event], SerializerCls)

        assert result == [
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_file", "file_url": "https://example.com/d.pdf"}],
            }
        ]

    def test_document_binary_becomes_input_file_with_filename(self) -> None:
        """BinaryInput(DOCUMENT) → input_file with file_data; API requires filename."""
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="c1",
                    name="t",
                    result=ToolResult(DocumentInput(data=b"%PDF-1.4", media_type="application/pdf")),
                )
            ]
        )
        result = events_to_responses_input([event], SerializerCls)

        expected_data = f"data:application/pdf;base64,{base64.b64encode(b'%PDF-1.4').decode()}"
        assert result == [
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_file", "file_data": expected_data, "filename": "file.pdf"}],
            }
        ]

    def test_document_binary_uses_vendor_filename(self) -> None:
        """Custom filename via vendor_metadata is preserved in tool-result output."""
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="c1",
                    name="t",
                    result=ToolResult(
                        BinaryInput(
                            data=b"%PDF-1.4",
                            media_type="application/pdf",
                            kind=BinaryType.DOCUMENT,
                            vendor_metadata={"filename": "report.pdf"},
                        )
                    ),
                )
            ]
        )
        result = events_to_responses_input([event], SerializerCls)

        assert result == [
            IsPartialDict({
                "type": "function_call_output",
                "call_id": "c1",
                "output": [IsPartialDict({"type": "input_file", "filename": "report.pdf"})],
            })
        ]

    def test_file_id_becomes_input_file_block(self) -> None:
        """Only file_id is accepted (filename is rejected as mutually exclusive by the API)."""
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="c1",
                    name="t",
                    result=ToolResult(FileIdInput(file_id="file-xyz", filename="r.pdf")),
                )
            ]
        )
        result = events_to_responses_input([event], SerializerCls)

        assert result == [
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_file", "file_id": "file-xyz"}],
            }
        ]

    def test_mixed_text_and_image(self) -> None:
        png = b"\x89PNG\r\n"
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="c1",
                    name="t",
                    result=ToolResult("here is an image", ImageInput(data=png, media_type="image/png")),
                )
            ]
        )
        result = events_to_responses_input([event], SerializerCls)

        expected_url = f"data:image/png;base64,{base64.b64encode(png).decode()}"
        assert result == [
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [
                    {"type": "output_text", "text": "here is an image"},
                    {"type": "input_image", "image_url": expected_url},
                ],
            }
        ]

    def test_audio_binary_raises(self) -> None:
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="c1",
                    name="t",
                    result=ToolResult(AudioInput(data=b"\x00", media_type="audio/wav")),
                )
            ]
        )
        with pytest.raises(UnsupportedInputError, match="BinaryInput.*openai-responses"):
            events_to_responses_input([event], SerializerCls)
