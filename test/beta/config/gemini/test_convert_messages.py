# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.config.gemini.mappers import convert_messages
from autogen.beta.events import (
    AudioInput,
    BinaryInput,
    DocumentInput,
    FileIdInput,
    ImageInput,
    ModelRequest,
    ModelResponse,
    TextInput,
    VideoInput,
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
        result = convert_messages([_model_response_with_tool_call(arguments)])

        assert len(result) == 1
        part = result[0].parts[0]
        assert part.function_call is not None
        assert part.function_call.args == {}

    def test_valid_arguments_are_preserved(self) -> None:
        result = convert_messages([_model_response_with_tool_call('{"category": "books"}')])

        part = result[0].parts[0]
        assert part.function_call.args == {"category": "books"}


class TestImageUrlInput:
    IMAGE_URL = "https://example.com/image.png"

    def test_converts_to_part_from_uri(self) -> None:
        result = convert_messages([ModelRequest([ImageInput(url=self.IMAGE_URL)])])

        assert len(result) == 1
        assert result[0].role == "user"
        part = result[0].parts[0]
        assert part.file_data.file_uri == self.IMAGE_URL
        assert part.file_data.mime_type == "image/png"


class TestImageBinaryInput:
    SAMPLE_BYTES = b"\x89PNG\r\n\x1a\nfake"

    def test_converts_to_part_from_bytes(self) -> None:
        result = convert_messages([ModelRequest([ImageInput(data=self.SAMPLE_BYTES, media_type="image/png")])])

        assert len(result) == 1
        part = result[0].parts[0]
        assert part.inline_data.data == self.SAMPLE_BYTES
        assert part.inline_data.mime_type == "image/png"


class TestAudioUrlInput:
    AUDIO_URL = "https://example.com/audio.wav"

    def test_converts_to_part_from_uri(self) -> None:
        result = convert_messages([ModelRequest([AudioInput(url=self.AUDIO_URL)])])

        assert len(result) == 1
        part = result[0].parts[0]
        assert part.file_data.file_uri == self.AUDIO_URL
        assert part.file_data.mime_type == "audio/wav"


class TestAudioBinaryInput:
    SAMPLE_BYTES = b"\x00\x01\x02audio"

    def test_converts_to_part_from_bytes(self) -> None:
        result = convert_messages([ModelRequest([AudioInput(data=self.SAMPLE_BYTES, media_type="audio/wav")])])

        part = result[0].parts[0]
        assert part.inline_data.data == self.SAMPLE_BYTES
        assert part.inline_data.mime_type == "audio/wav"


class TestDocumentUrlInput:
    DOC_URL = "https://example.com/doc.pdf"

    def test_converts_to_part_from_uri(self) -> None:
        result = convert_messages([ModelRequest([DocumentInput(url=self.DOC_URL)])])

        part = result[0].parts[0]
        assert part.file_data.file_uri == self.DOC_URL
        assert part.file_data.mime_type == "application/pdf"


class TestDocumentBinaryInput:
    SAMPLE_BYTES = b"%PDF-1.4"

    def test_converts_to_part_from_bytes(self) -> None:
        result = convert_messages([ModelRequest([DocumentInput(data=self.SAMPLE_BYTES, media_type="application/pdf")])])

        part = result[0].parts[0]
        assert part.inline_data.data == self.SAMPLE_BYTES
        assert part.inline_data.mime_type == "application/pdf"


class TestVideoUrlInput:
    VIDEO_URL = "https://example.com/clip.mp4"

    def test_converts_to_part_from_uri(self) -> None:
        result = convert_messages([ModelRequest([VideoInput(url=self.VIDEO_URL)])])

        part = result[0].parts[0]
        assert part.file_data.file_uri == self.VIDEO_URL
        assert part.file_data.mime_type == "video/mp4"

    def test_youtube_url_has_no_mime_type(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = convert_messages([ModelRequest([VideoInput(url=url)])])

        part = result[0].parts[0]
        assert part.file_data.file_uri == url
        assert part.file_data.mime_type is None


class TestVideoBinaryInput:
    SAMPLE_BYTES = b"\x00\x00\x00\x1cftypisom"

    def test_converts_to_part_from_bytes(self) -> None:
        result = convert_messages([ModelRequest([VideoInput(data=self.SAMPLE_BYTES, media_type="video/mp4")])])

        part = result[0].parts[0]
        assert part.inline_data.data == self.SAMPLE_BYTES
        assert part.inline_data.mime_type == "video/mp4"


class TestVendorMetadata:
    SAMPLE_BYTES = b"\x89PNG\r\n\x1a\nfake"

    def test_media_resolution(self) -> None:
        inp = BinaryInput(
            data=self.SAMPLE_BYTES,
            media_type="image/png",
            vendor_metadata={"media_resolution": "MEDIA_RESOLUTION_LOW"},
        )
        result = convert_messages([ModelRequest([inp])])

        part = result[0].parts[0]
        assert part.media_resolution is not None

    def test_video_metadata_dict(self) -> None:
        inp = BinaryInput(
            data=b"\x00\x00video",
            media_type="video/mp4",
            vendor_metadata={"video_metadata": {"fps": 5, "start_offset": "10s", "end_offset": "30s"}},
        )
        result = convert_messages([ModelRequest([inp])])

        part = result[0].parts[0]
        assert part.video_metadata is not None
        assert part.video_metadata.fps == 5.0
        assert part.video_metadata.start_offset == "10s"
        assert part.video_metadata.end_offset == "30s"

    def test_display_name(self) -> None:
        inp = BinaryInput(
            data=self.SAMPLE_BYTES,
            media_type="image/png",
            vendor_metadata={"display_name": "my_photo.png"},
        )
        result = convert_messages([ModelRequest([inp])])

        part = result[0].parts[0]
        assert part.inline_data.display_name == "my_photo.png"

    def test_empty_metadata_is_noop(self) -> None:
        inp = BinaryInput(data=self.SAMPLE_BYTES, media_type="image/png")
        result = convert_messages([ModelRequest([inp])])

        part = result[0].parts[0]
        assert part.inline_data.data == self.SAMPLE_BYTES


class TestAudioFormatVariants:
    SAMPLE_BYTES = b"\x00\x01\x02audio"

    def test_wav(self) -> None:
        result = convert_messages([ModelRequest([AudioInput(data=self.SAMPLE_BYTES, media_type="audio/wav")])])

        part = result[0].parts[0]
        assert part.inline_data.mime_type == "audio/wav"

    def test_mp3(self) -> None:
        result = convert_messages([ModelRequest([AudioInput(data=self.SAMPLE_BYTES, media_type="audio/mpeg")])])

        part = result[0].parts[0]
        assert part.inline_data.mime_type == "audio/mpeg"

    def test_ogg(self) -> None:
        result = convert_messages([ModelRequest([AudioInput(data=self.SAMPLE_BYTES, media_type="audio/ogg")])])

        part = result[0].parts[0]
        assert part.inline_data.mime_type == "audio/ogg"


class TestMultipleInputs:
    def test_multiple_inputs_grouped_into_one_content(self) -> None:
        result = convert_messages([
            ModelRequest([
                TextInput("Describe these images."),
                ImageInput(url="https://example.com/a.png"),
                ImageInput(url="https://example.com/b.jpg"),
            ])
        ])

        assert len(result) == 1
        assert result[0].role == "user"
        assert len(result[0].parts) == 3
        assert result[0].parts[0].text == "Describe these images."
        assert result[0].parts[1].file_data.file_uri == "https://example.com/a.png"
        assert result[0].parts[2].file_data.file_uri == "https://example.com/b.jpg"

    def test_mixed_text_and_binary(self) -> None:
        result = convert_messages([
            ModelRequest([
                TextInput("What is in this image?"),
                ImageInput(data=b"\x89PNG", media_type="image/png"),
            ])
        ])

        assert len(result) == 1
        assert len(result[0].parts) == 2
        assert result[0].parts[0].text == "What is in this image?"
        assert result[0].parts[1].inline_data.data == b"\x89PNG"


def test_file_id_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="FileIdInput.*gemini"):
        convert_messages([ModelRequest([FileIdInput(file_id="file-abc123")])])
