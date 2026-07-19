# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator
from contextlib import ExitStack
from typing import Any

import pytest

pytest.importorskip("elevenlabs")

from ag2 import Agent, MemoryStream, events, testing
from ag2.context import ConversationContext
from ag2.live import (
    ElevenLabsStreamingTTSConfig,
    ElevenLabsStreamingTTSObserver,
    ElevenLabsTTSConfig,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.elevenlabs]


class FakeTextToSpeech:
    """Stands in for `AsyncElevenLabs.text_to_speech`.

    Both real endpoints are async *generator* functions (called without
    `await`), so these are too — a coroutine here would let a broken call site
    pass.
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.convert_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        # Set by `stream` between yields so tests can prove chunks are handed
        # over incrementally rather than buffered.
        self.yielded = 0

    async def convert(self, **kwargs: Any) -> AsyncIterator[bytes]:
        self.convert_calls.append(kwargs)
        for chunk in self._chunks:
            yield chunk

    async def stream(self, **kwargs: Any) -> AsyncIterator[bytes]:
        self.stream_calls.append(kwargs)
        for chunk in self._chunks:
            self.yielded += 1
            yield chunk


class FakeClient:
    def __init__(self, chunks: list[bytes]) -> None:
        self.text_to_speech = FakeTextToSpeech(chunks)


@pytest.fixture
def client() -> FakeClient:
    return FakeClient([b"one", b"two", b"three"])


class TestTTSConfig:
    """Synchronous / whole-clip mode."""

    async def test_synthesize_returns_full_clip(self, client: FakeClient) -> None:
        config = ElevenLabsTTSConfig(client=client)

        assert await config.synthesize("hello there") == b"onetwothree"

    async def test_uses_non_streaming_endpoint_with_defaults(self, client: FakeClient) -> None:
        config = ElevenLabsTTSConfig(client=client)

        await config.synthesize("hello there")

        assert client.text_to_speech.convert_calls == [
            {
                "voice_id": "Rachel",
                "text": "hello there",
                "model_id": "eleven_v3",
                "output_format": "pcm_24000",
            }
        ]
        assert client.text_to_speech.stream_calls == []

    async def test_overrides_propagate(self, client: FakeClient) -> None:
        config = ElevenLabsTTSConfig(
            "eleven_multilingual_v2",
            voice_id="Bella",
            output_format="mp3_44100_128",
            client=client,
        )

        await config.synthesize("hello there")

        assert client.text_to_speech.convert_calls == [
            {
                "voice_id": "Bella",
                "text": "hello there",
                "model_id": "eleven_multilingual_v2",
                "output_format": "mp3_44100_128",
            }
        ]

    async def test_empty_response_is_empty_bytes(self) -> None:
        config = ElevenLabsTTSConfig(client=FakeClient([]))

        assert await config.synthesize("hello there") == b""


class TestStreamingTTSConfig:
    async def test_stream_yields_each_chunk(self, client: FakeClient) -> None:
        config = ElevenLabsStreamingTTSConfig(client=client)

        assert [chunk async for chunk in config.stream("hello there")] == [b"one", b"two", b"three"]

    async def test_uses_streaming_endpoint_with_defaults(self, client: FakeClient) -> None:
        config = ElevenLabsStreamingTTSConfig(client=client)

        [chunk async for chunk in config.stream("hello there")]

        assert client.text_to_speech.stream_calls == [
            {
                "voice_id": "Rachel",
                "text": "hello there",
                "model_id": "eleven_flash_v2_5",
                "output_format": "pcm_24000",
            }
        ]
        assert client.text_to_speech.convert_calls == []

    async def test_chunks_arrive_before_generation_finishes(self, client: FakeClient) -> None:
        """The whole point of this mode: no buffering of the full clip."""
        config = ElevenLabsStreamingTTSConfig(client=client)

        seen = 0
        async for _ in config.stream("hello there"):
            seen += 1
            # Consumer has the Nth chunk while the source has produced only N.
            assert client.text_to_speech.yielded == seen
            if seen == 1:
                break

    async def test_empty_chunks_are_dropped(self) -> None:
        config = ElevenLabsStreamingTTSConfig(client=FakeClient([b"one", b"", b"two"]))

        assert [chunk async for chunk in config.stream("hello there")] == [b"one", b"two"]

    async def test_synthesize_buffers_the_stream(self, client: FakeClient) -> None:
        config = ElevenLabsStreamingTTSConfig(client=client)

        assert await config.synthesize("hello there") == b"onetwothree"

    async def test_overrides_propagate(self, client: FakeClient) -> None:
        config = ElevenLabsStreamingTTSConfig(
            "eleven_turbo_v2_5",
            voice_id="Bella",
            output_format="pcm_16000",
            client=client,
        )

        [chunk async for chunk in config.stream("hello there")]

        assert client.text_to_speech.stream_calls == [
            {
                "voice_id": "Bella",
                "text": "hello there",
                "model_id": "eleven_turbo_v2_5",
                "output_format": "pcm_16000",
            }
        ]


async def speak(reply: str, client: FakeClient) -> list[bytes]:
    """One agent turn through the observer, returning the audio it emitted.

    Uses a non-streaming config, so the observer sees only the completed
    `ModelMessage` — no `ModelMessageChunk` at all.
    """
    stream = MemoryStream()
    audio = _collect_audio_on(stream)

    agent = Agent(
        "speaker",
        config=testing.TestConfig(reply),
        observers=[ElevenLabsStreamingTTSObserver(ElevenLabsStreamingTTSConfig(client=client))],
    )
    await agent.ask("say something", stream=stream)
    await asyncio.sleep(0.01)
    return audio


async def speak_streaming(chunks: list[str], client: FakeClient, *, turns: int = 1) -> list[bytes]:
    """Replay a streaming model: chunks in order, then the completed message.

    `TestConfig` cannot emit `ModelMessageChunk`, so the observer is registered
    against a context directly — `register` is the public `Observer` protocol.
    A single observer spans all `turns`, which is what exposes state leaking
    between them.
    """
    stream = MemoryStream()
    audio = _collect_audio_on(stream)

    tts = ElevenLabsStreamingTTSObserver(ElevenLabsStreamingTTSConfig(client=client))
    context = ConversationContext(stream=stream)

    with ExitStack() as stack:
        tts.register(stack, context)
        for _ in range(turns):
            for chunk in chunks:
                await context.send(events.ModelMessageChunk(content=chunk))
            await context.send(events.ModelMessage(content="".join(chunks)))
            await asyncio.sleep(0.01)

    return audio


def _collect_audio_on(stream: MemoryStream) -> list[bytes]:
    audio: list[bytes] = []

    async def collect(event: events.SynthesizedAudioEvent) -> None:
        audio.append(event.content)

    stream.where(events.SynthesizedAudioEvent).subscribe(collect)
    return audio


class TestStreamingTTSObserver:
    async def test_emits_one_event_per_audio_chunk(self, client: FakeClient) -> None:
        assert await speak("Hello, world!", client) == [b"one", b"two", b"three"]

    async def test_non_streaming_config_still_speaks(self, client: FakeClient) -> None:
        """No ModelMessageChunk events at all — the completed message is the fallback."""
        await speak("Hello, world!", client)

        assert client.text_to_speech.stream_calls == [
            {
                "voice_id": "Rachel",
                "text": "Hello, world!",
                "model_id": "eleven_flash_v2_5",
                "output_format": "pcm_24000",
            }
        ]

    async def test_blank_message_is_not_synthesized(self, client: FakeClient) -> None:
        assert await speak("   ", client) == []
        assert client.text_to_speech.stream_calls == []

    async def test_speaks_mid_reply_at_sentence_boundary(self, client: FakeClient) -> None:
        """The TODO this observer was written for: don't wait for the last token."""
        first = "This first sentence is comfortably past the minimum length. "
        rest = "And here is the tail."

        await speak_streaming([first, rest], client)

        # Two requests, not one — the first fired before the reply was complete.
        assert [call["text"] for call in client.text_to_speech.stream_calls] == [first.strip(), rest]

    async def test_short_text_waits_past_sentence_boundary(self, client: FakeClient) -> None:
        """Below min_chars a boundary is not worth a request on its own."""
        await speak_streaming(["Too short. ", "Still short."], client)

        assert [call["text"] for call in client.text_to_speech.stream_calls] == ["Too short. Still short."]

    async def test_buffer_does_not_leak_across_turns(self, client: FakeClient) -> None:
        """A trailing fragment from turn one must not be prepended to turn two."""
        await speak_streaming(["A trailing fragment with no sentence ending"], client, turns=2)

        assert [call["text"] for call in client.text_to_speech.stream_calls] == [
            "A trailing fragment with no sentence ending",
            "A trailing fragment with no sentence ending",
        ]
