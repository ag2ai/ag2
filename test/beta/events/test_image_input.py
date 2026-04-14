# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from autogen.beta.events import BinaryInput, FileIdInput, ImageInput, ImageUrlInput


def test_url_returns_image_url_input() -> None:
    result = ImageInput("https://example.com/img.png")

    assert isinstance(result, ImageUrlInput)
    assert result.url == "https://example.com/img.png"


class TestFileId:
    def test_returns_image_file_id_input(self) -> None:
        result = ImageInput(file_id="file-img123")

        assert isinstance(result, FileIdInput)
        assert result.file_id == "file-img123"
        assert result.filename is None

    def test_with_filename(self) -> None:
        result = ImageInput(file_id="file-img123", filename="photo.png")

        assert isinstance(result, FileIdInput)
        assert result.filename == "photo.png"


class TestData:
    def test_returns_binary_input(self) -> None:
        result = ImageInput(data=b"raw", media_type="image/png")

        assert isinstance(result, BinaryInput)
        assert result.data == b"raw"
        assert result.media_type == "image/png"

    def test_missing_media_type_raises(self) -> None:
        with pytest.raises(ValueError, match="media_type"):
            ImageInput(data=b"raw")


class TestPath:
    def test_infers_png(self, tmp_path: Path) -> None:
        f = tmp_path / "photo.png"
        f.write_bytes(b"png-data")

        result = ImageInput(path=f)

        assert isinstance(result, BinaryInput)
        assert result.data == b"png-data"
        assert result.media_type == "image/png"

    def test_infers_jpeg(self, tmp_path: Path) -> None:
        f = tmp_path / "photo.jpg"
        f.write_bytes(b"jpg-data")

        result = ImageInput(path=f)

        assert result.media_type == "image/jpeg"

    def test_unknown_extension_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "photo.bmp"
        f.write_bytes(b"bmp-data")

        with pytest.raises(ValueError, match="Cannot infer"):
            ImageInput(path=f)

    def test_unknown_extension_with_explicit_media_type(self, tmp_path: Path) -> None:
        f = tmp_path / "photo.bmp"
        f.write_bytes(b"bmp-data")

        result = ImageInput(path=f, media_type="image/png")

        assert isinstance(result, BinaryInput)
        assert result.media_type == "image/png"

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        f = tmp_path / "img.webp"
        f.write_bytes(b"webp-data")

        result = ImageInput(path=str(f))

        assert result.media_type == "image/webp"


def test_no_args_raises() -> None:
    with pytest.raises(ValueError, match="requires one of"):
        ImageInput()
