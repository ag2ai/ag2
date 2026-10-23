# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator

import httpx
import pytest

pytest.importorskip("elevenlabs")

from elevenlabs.client import AsyncElevenLabs

from ag2.live import ElevenLabsStreamingTTSConfig

pytestmark = [pytest.mark.asyncio, pytest.mark.elevenlabs]


class AudioResponse(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"a" * 1024
        yield b"b" * 1024

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("finish", [False, True])
async def test_closing_tts_stream_releases_http_response(finish: bool) -> None:
    body = AudioResponse()

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "audio/pcm"}, stream=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http_client:
        client = AsyncElevenLabs(api_key="test", httpx_client=http_client)
        audio = ElevenLabsStreamingTTSConfig(client=client).stream("hello")
        try:
            assert await anext(audio) == b"a" * 1024
            if finish:
                assert [chunk async for chunk in audio] == [b"b" * 1024]
        finally:
            await audio.aclose()

        assert body.closed
