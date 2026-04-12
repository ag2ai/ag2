# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest
from dirty_equals import IsPartialDict

from autogen.beta.config.openai.mappers import convert_messages, events_to_responses_input
from autogen.beta.events import (
    AudioInput,
    AudioUrlInput,
    BinaryInput,
    BinaryType,
    DocumentUrlInput,
    FileIdInput,
    ImageInput,
    ImageUrlInput,
    ModelRequest,
    TextInput,
)
from autogen.beta.exceptions import UnsupportedInputError


class TestTextInput:
    def test_completions(self) -> None:
        result = convert_messages([], [ModelRequest([TextInput("hello")])])

        assert result[1] == {"role": "user", "content": "hello"}

    def test_responses(self) -> None:
        result = events_to_responses_input([ModelRequest([TextInput("hello")])])

        assert result == [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]

    def test_completions_text_with_image_url(self) -> None:
        """Text + image in one ModelRequest must produce a single message with content array."""
        image_url = "https://example.com/image.png"
        result = convert_messages(
            [],
            [
                ModelRequest([TextInput("describe this"), ImageUrlInput(url=image_url)]),
            ],
        )

        assert len(result) == 2  # system + one user message
        assert result[1] == {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }


class TestImageUrlInput:
    IMAGE_URL = "https://example.com/image.png"

    def test_completions(self) -> None:
        result = convert_messages([], [ModelRequest([ImageUrlInput(url=self.IMAGE_URL)])])

        assert result[1] == {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": self.IMAGE_URL}}],
        }

    def test_responses(self) -> None:
        result = events_to_responses_input([ModelRequest([ImageUrlInput(url=self.IMAGE_URL)])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "input_image", "image_url": self.IMAGE_URL}],
            }
        ]


class TestFileIdInput:
    FILE_ID = "file-6F2ksmvXxt4VdoqmHRw6kL"

    def test_completions_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="FileIdInput.*openai-completions"):
            convert_messages([], [ModelRequest([FileIdInput(file_id=self.FILE_ID)])])

    def test_responses(self) -> None:
        result = events_to_responses_input([ModelRequest([FileIdInput(file_id=self.FILE_ID)])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "input_file", "file_id": self.FILE_ID}],
            }
        ]

    def test_responses_with_filename(self) -> None:
        result = events_to_responses_input([ModelRequest([FileIdInput(file_id=self.FILE_ID, filename="report.pdf")])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "input_file", "file_id": self.FILE_ID, "filename": "report.pdf"}],
            }
        ]


class TestAudioUrlInput:
    AUDIO_URL = "https://example.com/audio.wav"

    def test_completions_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="AudioUrlInput.*openai-completions"):
            convert_messages([], [ModelRequest([AudioUrlInput(url=self.AUDIO_URL)])])

    def test_responses_raises(self) -> None:
        with pytest.raises(UnsupportedInputError, match="AudioUrlInput.*openai-responses"):
            events_to_responses_input([ModelRequest([AudioUrlInput(url=self.AUDIO_URL)])])


class TestAudioBinaryInput:
    SAMPLE_BYTES = b"\x00\x01\x02audio"

    def test_completions(self) -> None:
        result = convert_messages([], [ModelRequest([AudioInput(data=self.SAMPLE_BYTES, media_type="audio/wav")])])

        expected_b64 = base64.b64encode(self.SAMPLE_BYTES).decode()
        assert result[1] == {
            "role": "user",
            "content": [{"type": "input_audio", "input_audio": {"data": expected_b64, "format": "wav"}}],
        }

    def test_completions_mp3(self) -> None:
        result = convert_messages([], [ModelRequest([AudioInput(data=self.SAMPLE_BYTES, media_type="audio/mpeg")])])

        expected_b64 = base64.b64encode(self.SAMPLE_BYTES).decode()
        assert result[1] == {
            "role": "user",
            "content": [{"type": "input_audio", "input_audio": {"data": expected_b64, "format": "mp3"}}],
        }


class TestBinaryInput:
    SAMPLE_BYTES = b"\x89PNG\r\n\x1a\nfake"

    def test_completions(self) -> None:
        result = convert_messages([], [ModelRequest([ImageInput(data=self.SAMPLE_BYTES, media_type="image/png")])])

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
        )

        assert result[1] == IsPartialDict({
            "role": "user",
            "content": [IsPartialDict({"type": "image_url", "detail": "low"})],
        })

    def test_responses(self) -> None:
        result = events_to_responses_input([
            ModelRequest([BinaryInput(data=self.SAMPLE_BYTES, media_type="image/png")])
        ])

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
        with pytest.raises(UnsupportedInputError, match="DocumentUrlInput.*openai-completions"):
            convert_messages([], [ModelRequest([DocumentUrlInput(url=self.DOC_URL)])])

    def test_responses(self) -> None:
        result = events_to_responses_input([ModelRequest([DocumentUrlInput(url=self.DOC_URL)])])

        assert result == [
            {
                "role": "user",
                "content": [{"type": "input_file", "file_url": self.DOC_URL}],
            }
        ]
