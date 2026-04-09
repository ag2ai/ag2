# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from autogen.beta.events import AudioInput, AudioUrlInput, BinaryInput, FileIdInput


def test_url_returns_audio_url_input() -> None:
    result = AudioInput("https://example.com/audio.wav")

    assert isinstance(result, AudioUrlInput)
    assert result.url == "https://example.com/audio.wav"


class TestFileId:
    def test_returns_file_id_input(self) -> None:
        result = AudioInput(file_id="file-audio123")

        assert isinstance(result, FileIdInput)
        assert result.file_id == "file-audio123"
        assert result.filename is None

    def test_with_filename(self) -> None:
        result = AudioInput(file_id="file-audio123", filename="recording.wav")

        assert isinstance(result, FileIdInput)
        assert result.filename == "recording.wav"


class TestData:
    def test_returns_binary_input(self) -> None:
        result = AudioInput(data=b"raw", media_type="audio/wav")

        assert isinstance(result, BinaryInput)
        assert result.data == b"raw"
        assert result.media_type == "audio/wav"

    def test_missing_media_type_raises(self) -> None:
        with pytest.raises(ValueError, match="media_type"):
            AudioInput(data=b"raw")


class TestPath:
    def test_infers_wav(self, tmp_path: Path) -> None:
        f = tmp_path / "recording.wav"
        f.write_bytes(b"wav-data")

        result = AudioInput(path=f)

        assert isinstance(result, BinaryInput)
        assert result.data == b"wav-data"
        assert result.media_type == "audio/wav"
        assert result.vendor_metadata == {"filename": "recording.wav"}

    def test_infers_mp3(self, tmp_path: Path) -> None:
        f = tmp_path / "song.mp3"
        f.write_bytes(b"mp3-data")

        result = AudioInput(path=f)

        assert result.media_type == "audio/mpeg"

    def test_unknown_extension_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "audio.xyz"
        f.write_bytes(b"data")

        with pytest.raises(ValueError, match="Cannot infer"):
            AudioInput(path=f)

    def test_unknown_extension_with_explicit_media_type(self, tmp_path: Path) -> None:
        f = tmp_path / "audio.xyz"
        f.write_bytes(b"data")

        result = AudioInput(path=f, media_type="audio/wav")

        assert isinstance(result, BinaryInput)
        assert result.media_type == "audio/wav"

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        f = tmp_path / "clip.ogg"
        f.write_bytes(b"ogg-data")

        result = AudioInput(path=str(f))

        assert result.media_type == "audio/ogg"


def test_no_args_raises() -> None:
    with pytest.raises(ValueError, match="requires one of"):
        AudioInput()
