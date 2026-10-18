# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path

import pytest

from ag2.events import AudioInput, BinaryInput, BinaryType, DocumentInput, ImageInput, VideoInput
from ag2.types import VideoMediaType


@pytest.mark.parametrize(
    ("factory", "filename", "media_type", "kind"),
    [
        (ImageInput, "download.jpg", "image/png", BinaryType.IMAGE),
        (DocumentInput, "download.txt", "text/markdown", BinaryType.DOCUMENT),
        (AudioInput, "download.wav", "audio/mpeg", BinaryType.AUDIO),
        (VideoInput, "download.mov", "video/mp4", BinaryType.VIDEO),
    ],
)
@pytest.mark.parametrize("string_path", [True, False])
def test_explicit_media_type_overrides_extension(
    tmp_path: Path,
    factory: Callable[..., BinaryInput],
    filename: str,
    media_type: str,
    kind: BinaryType,
    string_path: bool,
) -> None:
    path = tmp_path / filename
    path.write_bytes(b"content")

    result = factory(path=str(path) if string_path else path, media_type=media_type)

    assert result == BinaryInput(b"content", media_type=media_type, kind=kind, vendor_metadata={"filename": filename})


@pytest.mark.parametrize(
    ("filename", "media_type", "expected"),
    [
        ("clip.mov", None, "video/quicktime"),
        ("clip.mov", "video/quicktime", "video/quicktime"),
        ("clip.unknown", "video/mp4", "video/mp4"),
    ],
)
def test_video_path_inference_and_fallback(
    tmp_path: Path, filename: str, media_type: VideoMediaType | None, expected: str
) -> None:
    path = tmp_path / filename
    path.write_bytes(b"video")

    assert VideoInput(path=path, media_type=media_type) == BinaryInput(
        b"video", media_type=expected, kind=BinaryType.VIDEO, vendor_metadata={"filename": filename}
    )


def test_unknown_video_extension_requires_media_type(tmp_path: Path) -> None:
    path = tmp_path / "clip.unknown"
    path.write_bytes(b"video")

    with pytest.raises(ValueError, match="Cannot infer video media type"):
        VideoInput(path=path)
