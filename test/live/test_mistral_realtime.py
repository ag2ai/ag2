# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import math
import struct

import pytest

pytest.importorskip("mistralai")
pytest.importorskip("websockets")

import websockets.asyncio.server  # noqa: E402
from mistralai.client import Mistral

from ag2 import Agent, MemoryStream, events, testing
from ag2.context import ConversationContext
from ag2.live import CascadeConfig, MistralRealtimeTranscriber, SilenceTurnDetector
from ag2.live.stt import VoiceInput

pytestmark = [pytest.mark.asyncio, pytest.mark.mistral]

VOICE = VoiceInput(b"\x00\x01" * 100, frame_rate=16000, channels=1)
SAMPLE_RATE = 24000


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

    def __init__(self, *replies: dict[str, object], keep_open: bool = False) -> None:
        self.replies = replies
        self.keep_open = keep_open
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
                if not self.keep_open:
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


class Recorder:
    """Collects the events it is subscribed to and flags the first transcription delta."""

    def __init__(self) -> None:
        self.seen: list[events.BaseEvent] = []
        self.first_chunk = asyncio.Event()

    async def record(self, event: events.BaseEvent) -> None:
        self.seen.append(event)
        if isinstance(event, events.TranscriptionChunkEvent):
            self.first_chunk.set()


async def run_realtime(
    server: RealtimeServer,
    voice: VoiceInput = VOICE,
    recorder: Recorder | None = None,
    **kwargs: object,
) -> tuple[str, list[events.BaseEvent]]:
    stream = MemoryStream()
    recorder = recorder or Recorder()
    stream.where(events.TranscriptionChunkEvent | events.TranscriptionCompletedEvent).subscribe(recorder.record)
    text = await MistralRealtimeTranscriber(client=server.client(), **kwargs).transcribe(
        voice, ConversationContext(stream=stream)
    )
    await asyncio.sleep(0.01)
    return text, recorder.seen


def pcm(seconds: float, amplitude: int) -> bytes:
    n = int(SAMPLE_RATE * seconds)
    return struct.pack(f"<{n}h", *[int(amplitude * math.sin(i * 0.1)) for i in range(n)])


def detector() -> SilenceTurnDetector:
    return SilenceTurnDetector(sample_rate=SAMPLE_RATE, silence=0.3, min_speech=0.1, prefix_padding=0.1)


class FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        self.spoken.append(text)
        return text.encode()


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

    async def test_stream_closed_before_done_raises_without_completing(self) -> None:
        recorder = Recorder()
        async with RealtimeServer(delta("hello "), delta("there")) as server:
            with pytest.raises(RuntimeError, match="closed before the final transcript"):
                await run_realtime(server, recorder=recorder)

        assert recorder.seen == [
            events.TranscriptionChunkEvent("hello "),
            events.TranscriptionChunkEvent("there"),
        ]

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
        assert tracking.mock.call_args.args[0].parts == [events.TextInput("hello there")]


class TestCancellation:
    """The SDK swallows `CancelledError` while it waits for the next websocket event;
    a cancelled transcription must still read as cancelled, never as a finished clip."""

    async def test_cancel_after_the_first_delta_stays_cancelled(self) -> None:
        stream = MemoryStream()
        recorder = Recorder()
        stream.where(events.TranscriptionChunkEvent | events.TranscriptionCompletedEvent).subscribe(recorder.record)

        async with RealtimeServer(delta("unfinished sentence"), keep_open=True) as server:
            task = asyncio.create_task(
                MistralRealtimeTranscriber(client=server.client()).transcribe(VOICE, ConversationContext(stream=stream))
            )
            await asyncio.wait_for(recorder.first_chunk.wait(), timeout=2)

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.01)

        assert task.cancelled()
        assert recorder.seen == [events.TranscriptionChunkEvent("unfinished sentence")]

    async def test_cancelled_transcription_does_not_run_the_agent(self) -> None:
        tracking = testing.TrackingConfig(testing.TestConfig("Hi!"))
        stream = MemoryStream()
        recorder = Recorder()
        stream.where(events.TranscriptionChunkEvent).subscribe(recorder.record)

        async with RealtimeServer(delta("unfinished sentence"), keep_open=True) as server:
            pipeline = MistralRealtimeTranscriber(client=server.client()).pipe(Agent("assistant", config=tracking))
            task = asyncio.create_task(pipeline.ask(VOICE, stream=stream))
            await asyncio.wait_for(recorder.first_chunk.wait(), timeout=2)

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        tracking.mock.assert_not_called()

    async def test_barge_in_drops_the_interrupted_turn(self) -> None:
        context = ConversationContext(stream=MemoryStream())
        recorder = Recorder()
        context.stream.where(
            events.TranscriptionChunkEvent
            | events.TranscriptionCompletedEvent
            | events.AudioInterruptedEvent
            | events.ModelResponse
            | events.SynthesizedAudioEvent
        ).subscribe(recorder.record)

        tts = FakeTTS()
        async with RealtimeServer(delta("unfinished sentence"), keep_open=True) as server:
            config = CascadeConfig(
                stt=MistralRealtimeTranscriber(client=server.client()),
                model=testing.TestConfig("Spoken reply."),
                tts=tts,  # type: ignore[arg-type]
                turn_detector=detector,
                barge_in=True,
            )
            async with config.session(context, serializer=None):  # type: ignore[arg-type]
                for _ in range(2):
                    await context.send(events.RecordedAudioEvent(pcm(0.1, 0)))
                for _ in range(6):
                    await context.send(events.RecordedAudioEvent(pcm(0.1, 6000)))
                for _ in range(4):
                    await context.send(events.RecordedAudioEvent(pcm(0.1, 0)))
                await asyncio.wait_for(recorder.first_chunk.wait(), timeout=2)

                # The user starts talking again while the first clip is still being transcribed.
                await context.send(events.RecordedAudioEvent(pcm(0.1, 6000)))
                await asyncio.sleep(0.1)

        assert recorder.seen == [
            events.TranscriptionChunkEvent("unfinished sentence"),
            events.AudioInterruptedEvent(),
        ]
        assert tts.spoken == []
