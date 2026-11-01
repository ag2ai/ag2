# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import struct
from collections.abc import AsyncIterator, Sequence
from contextlib import ExitStack, aclosing
from typing import Any

import pytest

pytest.importorskip("mistralai")
pytest.importorskip("websockets")

from mistralai.client.models import (
    RealtimeSpeechError,
    RealtimeSpeechErrorDetail,
    SpeechStreamAudioDelta,
    SpeechStreamDone,
    UsageInfo,
)

from ag2 import MemoryStream, events
from ag2.context import ConversationContext
from ag2.live import MistralRealtimeTTSConfig, TTSObserver
from ag2.live.mistral import DEFAULT_MODEL
from ag2.live.protocols import StreamingTTSConfig, TextStreamingTTSConfig

pytestmark = [pytest.mark.asyncio, pytest.mark.mistral]

FLOAT32 = struct.pack("<4f", 0.0, 0.5, -0.5, 1.0)
INT16 = struct.pack("<4h", 0, 16384, -16384, 32767)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def delta(data: bytes) -> SpeechStreamAudioDelta:
    return SpeechStreamAudioDelta(audio_data=b64(data))


DONE = SpeechStreamDone(usage=UsageInfo(prompt_tokens=1, total_tokens=1))


class FakeConnection:
    """Mirrors `RealtimeSpeechConnection`: answers each text piece with the next
    scripted delta, and `end` with the rest of the script.

    Like the real one, its event iterator swallows `CancelledError`.
    """

    def __init__(self, per_text: list[Any], on_end: list[Any]) -> None:
        self._per_text = list(per_text)
        self._on_end = list(on_end)
        self._events: asyncio.Queue[Any] = asyncio.Queue()
        self.sent: list[str] = []
        self.ended = False
        self.closed = False

    async def send_text(self, text: str) -> None:
        self.sent.append(text)
        if self._per_text:
            self._events.put_nowait(self._per_text.pop(0))

    async def end(self) -> None:
        self.ended = True
        for event in self._on_end:
            self._events.put_nowait(event)
        self._events.put_nowait(None)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Any]:
        try:
            while (event := await self._events.get()) is not None:
                yield event
        except asyncio.CancelledError:
            pass


class FakeRealtimeSpeech:
    def __init__(self, per_text: list[Any], on_end: list[Any]) -> None:
        self._per_text = per_text
        self._on_end = on_end
        self.connects: list[dict[str, Any]] = []
        self.connections: list[FakeConnection] = []

    async def connect(self, model: str, **kwargs: Any) -> FakeConnection:
        self.connects.append({"model": model, **kwargs})
        connection = FakeConnection(self._per_text, self._on_end)
        self.connections.append(connection)
        return connection


class FakeAudio:
    def __init__(self, per_text: list[Any], on_end: list[Any]) -> None:
        self.realtime_speech = FakeRealtimeSpeech(per_text, on_end)


class FakeClient:
    """Mirrors the real `Mistral` client's `client.audio.realtime_speech` path."""

    def __init__(self, per_text: Sequence[Any] = (), on_end: Sequence[Any] = ()) -> None:
        self.audio = FakeAudio(list(per_text), list(on_end))


@pytest.fixture
def client() -> FakeClient:
    return FakeClient([delta(FLOAT32[:8])], [delta(FLOAT32[8:]), DONE])


def config(client: FakeClient, **kwargs: Any) -> MistralRealtimeTTSConfig:
    return MistralRealtimeTTSConfig(client=client, **{"voice_id": "v", **kwargs})  # type: ignore[arg-type]


async def pieces(*texts: str) -> AsyncIterator[str]:
    for text in texts:
        yield text


class TestVoiceSelection:
    @pytest.mark.parametrize(
        "voice",
        [
            pytest.param({}, id="neither"),
            pytest.param({"voice_id": "v", "ref_audio": "clip"}, id="both"),
            pytest.param({"voice_id": "", "ref_audio": ""}, id="both-empty"),
        ],
    )
    async def test_requires_exactly_one_of_voice_id_and_ref_audio(
        self, client: FakeClient, voice: dict[str, str]
    ) -> None:
        with pytest.raises(ValueError, match="exactly one of"):
            MistralRealtimeTTSConfig(client=client, **voice)  # type: ignore[arg-type]

    async def test_sends_the_saved_voice_and_pcm(self, client: FakeClient) -> None:
        await config(client).synthesize("hi")

        assert client.audio.realtime_speech.connects == [
            {
                "model": DEFAULT_MODEL,
                "response_format": "pcm",
                "streaming_mode": None,
                "expressiveness": None,
                "voice_id": "v",
            }
        ]

    async def test_sends_the_reference_clip_and_session_options(self, client: FakeClient) -> None:
        tts = MistralRealtimeTTSConfig(
            "custom-model",
            ref_audio="clip",
            streaming_mode="long_form",
            expressiveness=0.7,
            client=client,  # type: ignore[arg-type]
        )

        await tts.synthesize("hi")

        assert client.audio.realtime_speech.connects == [
            {
                "model": "custom-model",
                "response_format": "pcm",
                "streaming_mode": "long_form",
                "expressiveness": 0.7,
                "ref_audio": "clip",
            }
        ]


class TestRealtimeTTSConfig:
    async def test_is_a_text_streaming_config_only(self, client: FakeClient) -> None:
        tts = config(client)

        assert isinstance(tts, TextStreamingTTSConfig)
        assert not isinstance(tts, StreamingTTSConfig)

    async def test_sends_every_piece_and_ends_the_input(self, client: FakeClient) -> None:
        [chunk async for chunk in config(client).stream_text(pieces("Hello", "", " there"))]

        [connection] = client.audio.realtime_speech.connections
        assert connection.sent == ["Hello", " there"]
        assert connection.ended
        assert connection.closed

    async def test_yields_audio_before_the_text_is_complete(self, client: FakeClient) -> None:
        text: asyncio.Queue[str] = asyncio.Queue()

        async def more_text() -> AsyncIterator[str]:
            while True:
                yield await text.get()

        text.put_nowait("Hello")
        async with aclosing(config(client).stream_text(more_text())) as audio:  # type: ignore[type-var]
            assert await anext(audio) == INT16[:4]
            [connection] = client.audio.realtime_speech.connections
            assert not connection.ended

        assert connection.closed

    async def test_carries_a_split_sample_over_to_the_next_chunk(self) -> None:
        client = FakeClient([delta(FLOAT32[:6])], [delta(FLOAT32[6:]), DONE])

        assert [chunk async for chunk in config(client).stream_text(pieces("hi"))] == [INT16[:2], INT16[2:]]

    async def test_rejects_a_stream_that_ends_mid_sample(self) -> None:
        client = FakeClient([], [delta(FLOAT32[:-1]), DONE])

        with pytest.raises(ValueError, match="mid-sample"):
            await config(client).synthesize("hi")

    async def test_server_error_raises(self) -> None:
        error = RealtimeSpeechError(error=RealtimeSpeechErrorDetail(message="unknown voice", code=4001))
        client = FakeClient([], [error])

        with pytest.raises(RuntimeError, match="unknown voice"):
            await config(client).synthesize("hi")

        assert client.audio.realtime_speech.connections[0].closed

    async def test_connection_closing_before_done_raises(self) -> None:
        client = FakeClient([], [delta(FLOAT32)])

        with pytest.raises(RuntimeError, match="closed before the audio finished"):
            await config(client).synthesize("hi")

    async def test_failing_text_source_raises(self, client: FakeClient) -> None:
        async def broken() -> AsyncIterator[str]:
            yield "Hello"
            raise LookupError("model stream broke")

        with pytest.raises(LookupError, match="model stream broke"):
            [chunk async for chunk in config(client).stream_text(broken())]

        assert client.audio.realtime_speech.connections[0].closed

    async def test_cancelling_closes_the_connection(self, client: FakeClient) -> None:
        """The SDK swallows `CancelledError`; the cancellation must still reach the caller."""

        sent = asyncio.Event()

        async def endless() -> AsyncIterator[str]:
            yield "Hello"
            # Asked for the next piece: "Hello" has reached the connection.
            sent.set()
            await asyncio.Event().wait()

        async def consume() -> None:
            async for _ in config(client).stream_text(endless()):
                pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(sent.wait(), timeout=1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.audio.realtime_speech.connections[0].closed

    async def test_synthesize_buffers_the_stream(self, client: FakeClient) -> None:
        assert await config(client).synthesize("hi") == INT16


async def speak(tts: MistralRealtimeTTSConfig, *chunks: str, content: str) -> list[bytes]:
    stream = MemoryStream()
    audio: list[bytes] = []
    context = ConversationContext(stream=stream)

    async def collect(event: events.SynthesizedAudioEvent) -> None:
        audio.append(event.content)

    stream.where(events.SynthesizedAudioEvent).subscribe(collect)
    with ExitStack() as stack:
        TTSObserver(tts).register(stack, context)
        for chunk in chunks:
            await context.send(events.ModelMessageChunk(content=chunk))
        await context.send(events.ModelMessage(content))
        await asyncio.sleep(0.01)

    return audio


class TestViaTTSObserver:
    async def test_one_reply_is_one_connection_fed_every_chunk(self, client: FakeClient) -> None:
        audio = await speak(config(client), "Hi", " there.", content="Hi there.")

        assert audio == [INT16[:4], INT16[4:]]
        [connection] = client.audio.realtime_speech.connections
        assert connection.sent == ["Hi", " there."]
        assert connection.closed

    async def test_speaks_the_completed_reply_without_chunks(self, client: FakeClient) -> None:
        audio = await speak(config(client), content="  Hi there.  ")

        assert audio == [INT16[:4], INT16[4:]]
        assert client.audio.realtime_speech.connections[0].sent == ["Hi there."]

    async def test_server_error_surfaces_at_the_end_of_the_reply(self) -> None:
        error = RealtimeSpeechError(error=RealtimeSpeechErrorDetail(message="unknown voice", code=4001))
        client = FakeClient([], [error])

        with pytest.raises(RuntimeError, match="unknown voice"):
            await speak(config(client), "Hi", content="Hi")

        assert client.audio.realtime_speech.connections[0].closed

    async def test_empty_reply_opens_no_connection(self, client: FakeClient) -> None:
        assert await speak(config(client), content="") == []
        assert client.audio.realtime_speech.connections == []
