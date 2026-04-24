# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls

from autogen.beta.config.dashscope.mappers import convert_messages
from autogen.beta.events import AudioInput, BinaryInput, DocumentInput, FileIdInput, ImageInput, ModelRequest
from autogen.beta.exceptions import UnsupportedInputError


def test_audio_url_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="UrlInput.*dashscope"):
        convert_messages([], [ModelRequest([AudioInput(url="https://example.com/audio.wav")])], SerializerCls)


def test_image_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="UrlInput.*dashscope"):
        convert_messages([], [ModelRequest([ImageInput(url="https://example.com/img.png")])], SerializerCls)


def test_document_url_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="UrlInput.*dashscope"):
        convert_messages([], [ModelRequest([DocumentInput(url="https://example.com/doc.pdf")])], SerializerCls)


def test_file_id_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="FileIdInput.*dashscope"):
        convert_messages([], [ModelRequest([FileIdInput(file_id="file-abc123")])], SerializerCls)


def test_binary_input_raises() -> None:
    with pytest.raises(UnsupportedInputError, match="BinaryInput.*dashscope"):
        convert_messages([], [ModelRequest([BinaryInput(data=b"data", media_type="image/png")])], SerializerCls)
