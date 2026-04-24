# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls

from autogen.beta import ToolResult
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
    ToolResultEvent,
    ToolResultsEvent,
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
        [content] = convert_messages([_model_response_with_tool_call(arguments)], SerializerCls)

        assert content.model_dump(exclude_none=True) == {
            "role": "model",
            "parts": [{"function_call": {"name": "list_items", "args": {}}}],
        }

    def test_valid_arguments_are_preserved(self) -> None:
        [content] = convert_messages([_model_response_with_tool_call('{"category": "books"}')], SerializerCls)

        assert content.model_dump(exclude_none=True) == {
            "role": "model",
            "parts": [{"function_call": {"name": "list_items", "args": {"category": "books"}}}],
        }


def test_image_url() -> None:
    img_url = "https://example.com/image.png"
    [content] = convert_messages([ModelRequest([ImageInput(url=img_url)])], SerializerCls)

    assert content.model_dump(exclude_none=True) == {
        "role": "user",
        "parts": [{"file_data": {"file_uri": img_url, "mime_type": "image/png"}}],
    }


def test_image_binary() -> None:
    png = b"\x89PNG\r\n\x1a\nfake"
    [content] = convert_messages([ModelRequest([ImageInput(data=png, media_type="image/png")])], SerializerCls)

    assert content.model_dump(exclude_none=True) == {
        "role": "user",
        "parts": [{"inline_data": {"data": png, "mime_type": "image/png"}}],
    }


def test_audio_url() -> None:
    audio_url = "https://example.com/audio.wav"
    [content] = convert_messages([ModelRequest([AudioInput(url=audio_url)])], SerializerCls)

    assert content.model_dump(exclude_none=True) == {
        "role": "user",
        "parts": [{"file_data": {"file_uri": audio_url, "mime_type": "audio/wav"}}],
    }


def test_audio_binary() -> None:
    audio = b"\x00\x01\x02audio"
    [content] = convert_messages([ModelRequest([AudioInput(data=audio, media_type="audio/wav")])], SerializerCls)

    assert content.model_dump(exclude_none=True) == {
        "role": "user",
        "parts": [{"inline_data": {"data": audio, "mime_type": "audio/wav"}}],
    }


def test_document_url() -> None:
    doc_url = "https://example.com/doc.pdf"
    [content] = convert_messages([ModelRequest([DocumentInput(url=doc_url)])], SerializerCls)

    assert content.model_dump(exclude_none=True) == {
        "role": "user",
        "parts": [{"file_data": {"file_uri": doc_url, "mime_type": "application/pdf"}}],
    }


def test_document_binary() -> None:
    pdf = b"%PDF-1.4"
    [content] = convert_messages([ModelRequest([DocumentInput(data=pdf, media_type="application/pdf")])], SerializerCls)

    assert content.model_dump(exclude_none=True) == {
        "role": "user",
        "parts": [{"inline_data": {"data": pdf, "mime_type": "application/pdf"}}],
    }


class TestVideoUrl:
    def test_known_extension(self) -> None:
        url = "https://example.com/clip.mp4"
        [content] = convert_messages([ModelRequest([VideoInput(url=url)])], SerializerCls)

        assert content.model_dump(exclude_none=True) == {
            "role": "user",
            "parts": [{"file_data": {"file_uri": url, "mime_type": "video/mp4"}}],
        }

    def test_youtube_url_has_no_mime_type(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        [content] = convert_messages([ModelRequest([VideoInput(url=url)])], SerializerCls)

        assert content.model_dump(exclude_none=True) == {
            "role": "user",
            "parts": [{"file_data": {"file_uri": url}}],
        }


def test_video_binary() -> None:
    video = b"\x00\x00\x00\x1cftypisom"
    [content] = convert_messages([ModelRequest([VideoInput(data=video, media_type="video/mp4")])], SerializerCls)

    assert content.model_dump(exclude_none=True) == {
        "role": "user",
        "parts": [{"inline_data": {"data": video, "mime_type": "video/mp4"}}],
    }


class TestVendorMetadata:
    PNG = b"\x89PNG\r\n\x1a\nfake"

    def test_media_resolution(self) -> None:
        inp = BinaryInput(
            data=self.PNG,
            media_type="image/png",
            vendor_metadata={"media_resolution": "MEDIA_RESOLUTION_LOW"},
        )
        [content] = convert_messages([ModelRequest([inp])], SerializerCls)

        assert content.model_dump(exclude_none=True) == {
            "role": "user",
            "parts": [
                {
                    "media_resolution": "MEDIA_RESOLUTION_LOW",
                    "inline_data": {"data": self.PNG, "mime_type": "image/png"},
                }
            ],
        }

    def test_video_metadata_dict(self) -> None:
        inp = BinaryInput(
            data=b"\x00\x00video",
            media_type="video/mp4",
            vendor_metadata={"video_metadata": {"fps": 5, "start_offset": "10s", "end_offset": "30s"}},
        )
        [content] = convert_messages([ModelRequest([inp])], SerializerCls)

        assert content.model_dump(exclude_none=True) == {
            "role": "user",
            "parts": [
                {
                    "inline_data": {"data": b"\x00\x00video", "mime_type": "video/mp4"},
                    "video_metadata": {"fps": 5.0, "start_offset": "10s", "end_offset": "30s"},
                }
            ],
        }

    def test_display_name(self) -> None:
        inp = BinaryInput(
            data=self.PNG,
            media_type="image/png",
            vendor_metadata={"display_name": "my_photo.png"},
        )
        [content] = convert_messages([ModelRequest([inp])], SerializerCls)

        assert content.model_dump(exclude_none=True) == {
            "role": "user",
            "parts": [{"inline_data": {"data": self.PNG, "mime_type": "image/png", "display_name": "my_photo.png"}}],
        }

    def test_empty_metadata_is_noop(self) -> None:
        inp = BinaryInput(data=self.PNG, media_type="image/png")
        [content] = convert_messages([ModelRequest([inp])], SerializerCls)

        assert content.model_dump(exclude_none=True) == {
            "role": "user",
            "parts": [{"inline_data": {"data": self.PNG, "mime_type": "image/png"}}],
        }


class TestAudioFormatVariants:
    AUDIO = b"\x00\x01\x02audio"

    @pytest.mark.parametrize("media_type", ["audio/wav", "audio/mpeg", "audio/ogg"])
    def test_inline_audio_preserves_media_type(self, media_type: str) -> None:
        [content] = convert_messages(
            [ModelRequest([AudioInput(data=self.AUDIO, media_type=media_type)])], SerializerCls
        )

        assert content.model_dump(exclude_none=True) == {
            "role": "user",
            "parts": [{"inline_data": {"data": self.AUDIO, "mime_type": media_type}}],
        }


class TestMultipleInputs:
    def test_multiple_inputs_grouped_into_one_content(self) -> None:
        a_url = "https://example.com/a.png"
        b_url = "https://example.com/b.jpg"
        [content] = convert_messages(
            [
                ModelRequest([
                    TextInput("Describe these images."),
                    ImageInput(url=a_url),
                    ImageInput(url=b_url),
                ])
            ],
            SerializerCls,
        )

        assert content.model_dump(exclude_none=True) == {
            "role": "user",
            "parts": [
                {"text": "Describe these images."},
                {"file_data": {"file_uri": a_url, "mime_type": "image/png"}},
                {"file_data": {"file_uri": b_url, "mime_type": "image/jpeg"}},
            ],
        }

    def test_mixed_text_and_binary(self) -> None:
        png = b"\x89PNG"
        [content] = convert_messages(
            [
                ModelRequest([
                    TextInput("What is in this image?"),
                    ImageInput(data=png, media_type="image/png"),
                ])
            ],
            SerializerCls,
        )

        assert content.model_dump(exclude_none=True) == {
            "role": "user",
            "parts": [
                {"text": "What is in this image?"},
                {"inline_data": {"data": png, "mime_type": "image/png"}},
            ],
        }


class TestToolResult:
    """Tool results: text via response={'result': ...}, media via parts=[FunctionResponsePart...]."""

    PNG = b"\x89PNG\r\n"
    PDF = b"%PDF-1.4"
    IMG_URL = "https://example.com/image.png"
    PDF_URL = "https://example.com/doc.pdf"

    def test_text_only_goes_into_response(self) -> None:
        event = ToolResultsEvent(results=[ToolResultEvent(parent_id="tc_1", name="t", result=ToolResult("hello"))])
        [content] = convert_messages([event], SerializerCls)

        assert content.model_dump(exclude_none=True) == {
            "role": "user",
            "parts": [{"function_response": {"name": "t", "response": {"result": "hello"}}}],
        }

    def test_multiple_text_chunks_become_list(self) -> None:
        event = ToolResultsEvent(results=[ToolResultEvent(parent_id="tc_1", name="t", result=ToolResult("a", "b"))])
        [content] = convert_messages([event], SerializerCls)

        assert content.model_dump(exclude_none=True) == {
            "role": "user",
            "parts": [{"function_response": {"name": "t", "response": {"result": ["a", "b"]}}}],
        }

    def test_url_image(self) -> None:
        event = ToolResultsEvent(
            results=[ToolResultEvent(parent_id="tc_1", name="t", result=ToolResult(ImageInput(url=self.IMG_URL)))]
        )
        [content] = convert_messages([event], SerializerCls)

        assert content.model_dump(exclude_none=True) == {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "t",
                        "response": {},
                        "parts": [{"file_data": {"file_uri": self.IMG_URL, "mime_type": "image/png"}}],
                    }
                }
            ],
        }

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
        [content] = convert_messages([event], SerializerCls)

        assert content.model_dump(exclude_none=True) == {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "t",
                        "response": {},
                        "parts": [{"inline_data": {"data": self.PNG, "mime_type": "image/png"}}],
                    }
                }
            ],
        }

    def test_url_document(self) -> None:
        event = ToolResultsEvent(
            results=[ToolResultEvent(parent_id="tc_1", name="t", result=ToolResult(DocumentInput(url=self.PDF_URL)))]
        )
        [content] = convert_messages([event], SerializerCls)

        assert content.model_dump(exclude_none=True) == {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "t",
                        "response": {},
                        "parts": [{"file_data": {"file_uri": self.PDF_URL, "mime_type": "application/pdf"}}],
                    }
                }
            ],
        }

    def test_binary_document(self) -> None:
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="t",
                    result=ToolResult(DocumentInput(data=self.PDF, media_type="application/pdf")),
                )
            ]
        )
        [content] = convert_messages([event], SerializerCls)

        assert content.model_dump(exclude_none=True) == {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "t",
                        "response": {},
                        "parts": [{"inline_data": {"data": self.PDF, "mime_type": "application/pdf"}}],
                    }
                }
            ],
        }

    def test_mixed_text_and_image(self) -> None:
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="t",
                    result=ToolResult("caption", ImageInput(url=self.IMG_URL)),
                )
            ]
        )
        [content] = convert_messages([event], SerializerCls)

        assert content.model_dump(exclude_none=True) == {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "t",
                        "response": {"result": "caption"},
                        "parts": [{"file_data": {"file_uri": self.IMG_URL, "mime_type": "image/png"}}],
                    }
                }
            ],
        }

    def test_url_audio_raises(self) -> None:
        event = ToolResultsEvent(
            results=[ToolResultEvent(parent_id="tc_1", name="t", result=ToolResult(AudioInput(url="https://x/a.wav")))]
        )
        with pytest.raises(UnsupportedInputError, match="UrlInput.*gemini"):
            convert_messages([event], SerializerCls)

    def test_url_video_raises(self) -> None:
        event = ToolResultsEvent(
            results=[ToolResultEvent(parent_id="tc_1", name="t", result=ToolResult(VideoInput(url="https://x/v.mp4")))]
        )
        with pytest.raises(UnsupportedInputError, match="UrlInput.*gemini"):
            convert_messages([event], SerializerCls)

    def test_binary_audio_raises(self) -> None:
        event = ToolResultsEvent(
            results=[
                ToolResultEvent(
                    parent_id="tc_1",
                    name="t",
                    result=ToolResult(AudioInput(data=b"\x00", media_type="audio/wav")),
                )
            ]
        )
        with pytest.raises(UnsupportedInputError, match="BinaryInput.*gemini"):
            convert_messages([event], SerializerCls)

    def test_file_id_raises(self) -> None:
        event = ToolResultsEvent(
            results=[ToolResultEvent(parent_id="tc_1", name="t", result=ToolResult(FileIdInput(file_id="file-abc")))]
        )
        with pytest.raises(UnsupportedInputError, match="FileIdInput.*gemini"):
            convert_messages([event], SerializerCls)
