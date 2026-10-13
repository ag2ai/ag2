# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json

import pytest

pytest.importorskip("mistralai")
pytest.importorskip("websockets")

import websockets.asyncio.server  # noqa: E402
from mistralai.client import Mistral

from ag2 import Agent, MemoryStream, events, testing
from ag2.context import ConversationContext
from ag2.live import MistralRealtimeTranscriber
from ag2.live.stt import VoiceInput

pytestmark = [pytest.mark.asyncio, pytest.mark.mistral]

VOICE = VoiceInput(b"\x00\x01" * 100, frame_rate=16000, channels=1)


SESSION_CREATED = {
    "type": "session.created",
    "session": {
        "request_id": "req-1",
        "model": "voxtral-mini-transcribe-realtime-2602",
        "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
        "target_streaming_delay_ms": None,
    },
}
DONE = {
    "type": "transcription.done",
    "model": "voxtral-mini-transcribe-realtime-2602",
    "text": "hello there",
    "language": "en",
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


def delta(text: str) -> dict[str, str]:
    return {"type": "transcription.text.delta", "text": text}


class RealtimeServer:
    """A real websocket server the unmodified SDK connects to; records what the client sent."""

    def __init__(self, *replies: dict[str, object]) -> None:
        self.replies = replies
        self.path = ""
        self.received: list[dict[str, object]] = []
        self._server: websockets.asyncio.server.Server | None = None

    async def _handle(self, connection: websockets.asyncio.server.ServerConnection) -> None:
        self.path = connection.request.path
        await connection.send(json.dumps(SESSION_CREATED))
        async for raw in connection:
            message = json.loads(raw)
            self.received.append(message)
            if message["type"] == "input_audio.end":
                for reply in self.replies:
                    await connection.send(json.dumps(reply))
                await connection.close()

    async def __aenter__(self) -> "RealtimeServer":
        self._server = await websockets.asyncio.server.serve(self._handle, "127.0.0.1", 0)
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    def client(self) -> Mistral:
        assert self._server is not None
        port = self._server.sockets[0].getsockname()[1]
        return Mistral(api_key="test-key", server_url=f"http://127.0.0.1:{port}")

    def types(self) -> list[object]:
        return [m["type"] for m in self.received]


async def run_realtime(server: RealtimeServer, voice: VoiceInput = VOICE, **kwargs: object) -> tuple[str, list]:
    stream = MemoryStream()
    seen: list[events.BaseEvent] = []

    async def collect(event: events.BaseEvent) -> None:
        seen.append(event)

    stream.where(events.TranscriptionChunkEvent | events.TranscriptionCompletedEvent).subscribe(collect)
    text = await MistralRealtimeTranscriber(client=server.client(), **kwargs).transcribe(
        voice, ConversationContext(stream=stream)
    )
    await asyncio.sleep(0.01)
    return text, seen


class TestMistralRealtimeTranscriber:
    async def test_emits_each_delta_then_the_completed_event(self) -> None:
        async with RealtimeServer(delta("hello "), delta("there"), DONE) as server:
            text, seen = await run_realtime(server)

        assert text == "hello there"
        assert seen == [
            events.TranscriptionChunkEvent("hello "),
            events.TranscriptionChunkEvent("there"),
            events.TranscriptionCompletedEvent("hello there"),
        ]

    async def test_streams_the_clip_as_pcm_frames_and_closes_the_input(self) -> None:
        # 250 ms of 16 kHz mono: three frames (100 + 100 + 50 ms).
        voice = VoiceInput(b"\x01\x00" * 4000, frame_rate=16000, channels=1)

        async with RealtimeServer(DONE) as server:
            await run_realtime(server, voice, target_streaming_delay_ms=480)

        assert server.path.startswith("/v1/audio/transcriptions/realtime")
        assert "model=voxtral-mini-transcribe-realtime-2602" in server.path
        assert server.types() == [
            "session.update",
            "input_audio.append",
            "input_audio.append",
            "input_audio.append",
            "input_audio.flush",
            "input_audio.end",
        ]
        session = server.received[0]["session"]
        assert session == {
            "audio_format": {"encoding": "pcm_s16le", "sample_rate": 16000},
            "target_streaming_delay_ms": 480,
        }
        audio = b"".join(base64.b64decode(m["audio"]) for m in server.received if m["type"] == "input_audio.append")
        assert audio == voice.content

    async def test_uses_the_clip_sample_rate(self) -> None:
        async with RealtimeServer(DONE) as server:
            await run_realtime(server, VoiceInput(b"\x00\x00" * 10, frame_rate=24000, channels=1))

        assert server.received[0] == {
            "type": "session.update",
            "session": {"audio_format": {"encoding": "pcm_s16le", "sample_rate": 24000}},
        }

    async def test_falls_back_to_the_deltas_when_the_stream_ends_without_done(self) -> None:
        async with RealtimeServer(delta("hello "), delta("there")) as server:
            text, seen = await run_realtime(server)

        assert text == "hello there"
        assert seen[-1] == events.TranscriptionCompletedEvent("hello there")

    async def test_server_error_raises(self) -> None:
        error = {"type": "error", "error": {"message": "bad audio", "code": 400}}
        async with RealtimeServer(error) as server:
            with pytest.raises(RuntimeError, match="bad audio"):
                await run_realtime(server)

    async def test_rejects_stereo_before_connecting(self) -> None:
        async with RealtimeServer(DONE) as server:
            with pytest.raises(ValueError, match="mono"):
                await run_realtime(server, VoiceInput(b"\x00\x00" * 10, frame_rate=16000, channels=2))

        assert server.received == []

    async def test_pipes_the_transcript_into_an_agent(self) -> None:
        tracking = testing.TrackingConfig(testing.TestConfig("Hi!"))
        async with RealtimeServer(delta("hello there"), DONE) as server:
            reply = (
                await MistralRealtimeTranscriber(client=server.client())
                .pipe(Agent("assistant", config=tracking))
                .ask(VOICE)
            )

        assert reply.body == "Hi!"
        assert tracking.mock.call_args.args[0].parts[0].content == "hello there"
