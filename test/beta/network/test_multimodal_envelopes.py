# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for multimodal envelope support in the network layer.

Covers:
- EV_IMAGE / EV_AUDIO / EV_VIDEO constants exist and are distinct
- default_build_image_envelope / default_build_audio_envelope / default_build_video_envelope
- default_extract_turn_input decodes URL / binary / file-ID variants
- default_render_envelope produces descriptive placeholders
- Round-trip: build → extract gives back equivalent Input
"""

import base64

import pytest

from autogen.beta.network.adapters.base import (
    default_build_audio_envelope,
    default_build_image_envelope,
    default_build_video_envelope,
    default_extract_turn_input,
    default_render_envelope,
)
from autogen.beta.network.envelope import EV_AUDIO, EV_IMAGE, EV_TEXT, EV_VIDEO, Envelope

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestEventTypeConstants:
    def test_ev_image_defined(self) -> None:
        assert EV_IMAGE == "ag2.msg.image"

    def test_ev_audio_defined(self) -> None:
        assert EV_AUDIO == "ag2.msg.audio"

    def test_ev_video_defined(self) -> None:
        assert EV_VIDEO == "ag2.msg.video"

    def test_all_distinct(self) -> None:
        assert len({EV_TEXT, EV_IMAGE, EV_AUDIO, EV_VIDEO}) == 4


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


class TestBuildImageEnvelope:
    def test_url_variant(self) -> None:
        env = default_build_image_envelope("ch1", "alice", url="https://example.com/img.png")
        assert env.event_type == EV_IMAGE
        assert env.event_data == {"kind": "url", "url": "https://example.com/img.png"}

    def test_file_id_variant(self) -> None:
        env = default_build_image_envelope("ch1", "alice", file_id="file-abc", filename="photo.jpg")
        assert env.event_data == {"kind": "file_id", "file_id": "file-abc", "filename": "photo.jpg"}

    def test_file_id_no_filename(self) -> None:
        env = default_build_image_envelope("ch1", "alice", file_id="file-xyz")
        assert "filename" not in env.event_data

    def test_binary_variant(self) -> None:
        raw = b"\x89PNG\r\n"
        env = default_build_image_envelope("ch1", "alice", data=raw, media_type="image/png")
        assert env.event_data["kind"] == "binary"
        assert env.event_data["media_type"] == "image/png"
        assert base64.b64decode(env.event_data["data"]) == raw

    def test_binary_missing_media_type_raises(self) -> None:
        with pytest.raises(ValueError, match="media_type"):
            default_build_image_envelope("ch1", "alice", data=b"bytes")

    def test_no_source_raises(self) -> None:
        with pytest.raises(ValueError):
            default_build_image_envelope("ch1", "alice")

    def test_audience_and_causation(self) -> None:
        env = default_build_image_envelope(
            "ch1", "alice", url="https://x.com/a.jpg", audience=["bob"], causation_id="e0"
        )
        assert env.audience == ["bob"]
        assert env.causation_id == "e0"


class TestBuildAudioEnvelope:
    def test_url_variant(self) -> None:
        env = default_build_audio_envelope("ch1", "alice", url="https://example.com/clip.mp3")
        assert env.event_type == EV_AUDIO

    def test_binary_variant(self) -> None:
        raw = b"RIFF"
        env = default_build_audio_envelope("ch1", "alice", data=raw, media_type="audio/wav")
        assert env.event_data["media_type"] == "audio/wav"
        assert base64.b64decode(env.event_data["data"]) == raw


class TestBuildVideoEnvelope:
    def test_url_variant(self) -> None:
        env = default_build_video_envelope("ch1", "alice", url="https://example.com/clip.mp4")
        assert env.event_type == EV_VIDEO

    def test_binary_variant(self) -> None:
        raw = b"\x00\x00\x00\x18ftyp"
        env = default_build_video_envelope("ch1", "alice", data=raw, media_type="video/mp4")
        assert env.event_data["media_type"] == "video/mp4"
        assert base64.b64decode(env.event_data["data"]) == raw


# ---------------------------------------------------------------------------
# default_extract_turn_input — multimodal
# ---------------------------------------------------------------------------


class TestExtractTurnInputMultimodal:
    def _make(self, ev_type: str, event_data: dict) -> Envelope:
        return Envelope(channel_id="ch1", sender_id="alice", audience=None, event_type=ev_type, event_data=event_data)

    def test_ev_text_unchanged(self) -> None:
        env = self._make(EV_TEXT, {"text": "hello"})
        result = default_extract_turn_input(env)
        assert result == "hello"

    def test_image_url_returns_url_input(self) -> None:
        from autogen.beta.events.input_events import BinaryType, UrlInput

        env = self._make(EV_IMAGE, {"kind": "url", "url": "https://x.com/img.png"})
        result = default_extract_turn_input(env)
        assert isinstance(result, UrlInput)
        assert result.url == "https://x.com/img.png"
        assert result.kind == BinaryType.IMAGE

    def test_audio_url_returns_url_input(self) -> None:
        from autogen.beta.events.input_events import BinaryType, UrlInput

        env = self._make(EV_AUDIO, {"kind": "url", "url": "https://x.com/clip.mp3"})
        result = default_extract_turn_input(env)
        assert isinstance(result, UrlInput)
        assert result.kind == BinaryType.AUDIO

    def test_video_url_returns_url_input(self) -> None:
        from autogen.beta.events.input_events import BinaryType, UrlInput

        env = self._make(EV_VIDEO, {"kind": "url", "url": "https://x.com/clip.mp4"})
        result = default_extract_turn_input(env)
        assert isinstance(result, UrlInput)
        assert result.kind == BinaryType.VIDEO

    def test_image_file_id_returns_file_id_input(self) -> None:
        from autogen.beta.events.input_events import FileIdInput

        env = self._make(EV_IMAGE, {"kind": "file_id", "file_id": "file-abc", "filename": "photo.jpg"})
        result = default_extract_turn_input(env)
        assert isinstance(result, FileIdInput)
        assert result.file_id == "file-abc"
        assert result.filename == "photo.jpg"

    def test_image_binary_returns_binary_input(self) -> None:
        from autogen.beta.events.input_events import BinaryInput, BinaryType

        raw = b"\x89PNG\r\n"
        encoded = base64.b64encode(raw).decode()
        env = self._make(EV_IMAGE, {"kind": "binary", "data": encoded, "media_type": "image/png"})
        result = default_extract_turn_input(env)
        assert isinstance(result, BinaryInput)
        assert result.data == raw
        assert result.media_type == "image/png"
        assert result.kind == BinaryType.IMAGE

    def test_empty_url_returns_none(self) -> None:
        env = self._make(EV_IMAGE, {"kind": "url", "url": ""})
        assert default_extract_turn_input(env) is None

    def test_unknown_event_type_returns_none(self) -> None:
        env = self._make("ag2.msg.unknown", {"foo": "bar"})
        assert default_extract_turn_input(env) is None


# ---------------------------------------------------------------------------
# default_extract_turn_input — round-trip via builders
# ---------------------------------------------------------------------------


class TestExtractRoundTrip:
    def test_image_url_round_trip(self) -> None:
        from autogen.beta.events.input_events import UrlInput

        env = default_build_image_envelope("ch1", "alice", url="https://example.com/img.png")
        result = default_extract_turn_input(env)
        assert isinstance(result, UrlInput)
        assert result.url == "https://example.com/img.png"

    def test_image_binary_round_trip(self) -> None:
        from autogen.beta.events.input_events import BinaryInput

        raw = b"\x89PNG\r\n\x1a\n"
        env = default_build_image_envelope("ch1", "alice", data=raw, media_type="image/png")
        result = default_extract_turn_input(env)
        assert isinstance(result, BinaryInput)
        assert result.data == raw

    def test_audio_url_round_trip(self) -> None:
        from autogen.beta.events.input_events import UrlInput

        env = default_build_audio_envelope("ch1", "alice", url="https://example.com/clip.mp3")
        result = default_extract_turn_input(env)
        assert isinstance(result, UrlInput)

    def test_video_binary_round_trip(self) -> None:
        from autogen.beta.events.input_events import BinaryInput

        raw = b"\x00\x00\x00\x18ftyp"
        env = default_build_video_envelope("ch1", "alice", data=raw, media_type="video/mp4")
        result = default_extract_turn_input(env)
        assert isinstance(result, BinaryInput)
        assert result.data == raw


# ---------------------------------------------------------------------------
# default_render_envelope — multimodal
# ---------------------------------------------------------------------------


class TestRenderEnvelopeMultimodal:
    def _make(self, ev_type: str, event_data: dict) -> Envelope:
        return Envelope(channel_id="ch1", sender_id="alice", audience=None, event_type=ev_type, event_data=event_data)

    def test_ev_text_unchanged(self) -> None:
        env = self._make(EV_TEXT, {"text": "hello world"})
        assert default_render_envelope(env) == "hello world"

    def test_image_url_renders_descriptive(self) -> None:
        env = self._make(EV_IMAGE, {"kind": "url", "url": "https://x.com/img.png"})
        rendered = default_render_envelope(env)
        assert rendered is not None
        assert "image" in rendered
        assert "https://x.com/img.png" in rendered

    def test_audio_url_renders_descriptive(self) -> None:
        env = self._make(EV_AUDIO, {"kind": "url", "url": "https://x.com/clip.mp3"})
        rendered = default_render_envelope(env)
        assert rendered is not None
        assert "audio" in rendered

    def test_video_url_renders_descriptive(self) -> None:
        env = self._make(EV_VIDEO, {"kind": "url", "url": "https://x.com/clip.mp4"})
        rendered = default_render_envelope(env)
        assert rendered is not None
        assert "video" in rendered

    def test_binary_renders_media_type(self) -> None:
        env = self._make(EV_IMAGE, {"kind": "binary", "data": "abc=", "media_type": "image/png"})
        rendered = default_render_envelope(env)
        assert rendered is not None
        assert "image/png" in rendered

    def test_file_id_renders_filename(self) -> None:
        env = self._make(EV_IMAGE, {"kind": "file_id", "file_id": "file-abc", "filename": "photo.jpg"})
        rendered = default_render_envelope(env)
        assert rendered is not None
        assert "photo.jpg" in rendered

    def test_unknown_event_type_returns_none(self) -> None:
        env = self._make("ag2.msg.unknown", {})
        assert default_render_envelope(env) is None


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_ev_image_importable_from_network() -> None:
    from autogen.beta.network import EV_IMAGE as _EV_IMAGE

    assert _EV_IMAGE == "ag2.msg.image"


def test_ev_audio_importable_from_network() -> None:
    from autogen.beta.network import EV_AUDIO as _EV_AUDIO

    assert _EV_AUDIO == "ag2.msg.audio"


def test_ev_video_importable_from_network() -> None:
    from autogen.beta.network import EV_VIDEO as _EV_VIDEO

    assert _EV_VIDEO == "ag2.msg.video"


def test_builders_importable_from_network() -> None:
    from autogen.beta.network import (
        default_build_audio_envelope,
        default_build_image_envelope,
        default_build_video_envelope,
    )

    assert callable(default_build_image_envelope)
    assert callable(default_build_audio_envelope)
    assert callable(default_build_video_envelope)
