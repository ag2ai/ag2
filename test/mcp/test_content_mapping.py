# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
from dataclasses import dataclass

import pytest

from ag2 import Agent
from ag2.events import BinaryResult, ModelMessage, ModelResponse
from ag2.mcp import MCPServer
from ag2.mcp.mappers import reply_to_content, to_structured_dict
from ag2.mcp.testing import connect
from ag2.testing import TestConfig

_IMG = b"\x89PNG\r\n\x1a\n"
_AUDIO = b"RIFFxxxxWAVE"
_BLOB = b"%PDF-1.4 binary"


@pytest.mark.asyncio
class TestContentMapping:
    async def test_image_file_maps_to_image_content(self) -> None:
        response = ModelResponse(
            message=ModelMessage("here is the image"),
            files=[BinaryResult(_IMG, metadata={"media_type": "image/png"})],
        )
        server = MCPServer(Agent("painter", config=TestConfig(response)))

        async with connect(server) as session:
            result = await session.call_tool("ask", {"message": "draw"})

        image = next(c for c in result.content if c.type == "image")
        assert image.mimeType == "image/png"
        assert base64.b64decode(image.data) == _IMG
        assert any(c.type == "text" and c.text == "here is the image" for c in result.content)

    async def test_audio_file_maps_to_audio_content(self) -> None:
        response = ModelResponse(
            message=ModelMessage(""),
            files=[BinaryResult(_AUDIO, metadata={"media_type": "audio/wav"})],
        )
        server = MCPServer(Agent("speaker", config=TestConfig(response)))

        async with connect(server) as session:
            result = await session.call_tool("ask", {"message": "speak"})

        audio = next(c for c in result.content if c.type == "audio")
        assert audio.mimeType == "audio/wav"
        assert base64.b64decode(audio.data) == _AUDIO

    async def test_missing_media_type_defaults_to_image(self) -> None:
        # No media-type key in metadata -> _media_type falls back to the default.
        response = ModelResponse(message=ModelMessage(""), files=[BinaryResult(_IMG, metadata={})])
        server = MCPServer(Agent("painter", config=TestConfig(response)))

        async with connect(server) as session:
            result = await session.call_tool("ask", {"message": "draw"})

        image = next(c for c in result.content if c.type == "image")
        assert image.mimeType == "image/png"
        assert base64.b64decode(image.data) == _IMG

    async def test_other_blob_maps_to_embedded_resource(self) -> None:
        response = ModelResponse(
            message=ModelMessage(""),
            files=[BinaryResult(_BLOB, metadata={"media_type": "application/pdf", "filename": "doc.pdf"})],
        )
        server = MCPServer(Agent("writer", config=TestConfig(response)))

        async with connect(server) as session:
            result = await session.call_tool("ask", {"message": "write"})

        resource = next(c for c in result.content if c.type == "resource")
        assert resource.resource.mimeType == "application/pdf"
        assert base64.b64decode(resource.resource.blob) == _BLOB


def test_empty_reply_maps_to_single_empty_text() -> None:
    class _Reply:
        body = None
        files: list[BinaryResult] = []

    blocks = reply_to_content(_Reply())  # type: ignore[arg-type]

    assert len(blocks) == 1
    assert blocks[0].type == "text"
    assert blocks[0].text == ""


def test_to_structured_dict_variants() -> None:
    @dataclass
    class Point:
        x: int
        y: str

    assert to_structured_dict(Point(1, "a")) == {"x": 1, "y": "a"}  # dataclass -> dict
    assert to_structured_dict({"k": 1}) == {"k": 1}  # dict passthrough
    assert to_structured_dict(42) is None  # scalar -> None
    assert to_structured_dict([1, 2]) is None  # list -> None
