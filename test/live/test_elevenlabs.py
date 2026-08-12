# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator
from contextlib import ExitStack
from typing import Any

import pytest
from dirty_equals import IsPartialDict

pytest.importorskip("elevenlabs")

from ag2 import Agent, MemoryStream, events, testing
from ag2.context import ConversationContext
from ag2.live import (
    ElevenLabsStreamingTTSConfig,
    ElevenLabsTTSConfig,
    ElevenLabsTranscriber,
    TTSObserver,
)
from ag2.live.elevenlabs import DEFAULT_VOICE_ID
from ag2.live.stt import VoiceInput

pytestmark = [pytest.mark.asyncio, pytest.mark.elevenlabs]

VOICE = VoiceInput(b"\x00\x01" * 100, frame_rate=16000, channels=1)


class FakeTextToSpeech:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.convert_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
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


class FakeSpeechToText:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.convert_calls: list[dict[str, Any]] = []

    async def convert(self, **kwargs: Any) -> Any:
        recorded = dict(kwargs)
        if file := recorded.get("file"):
            recorded["file"] = file.read()
        self.convert_calls.append(recorded)
        return self._response


class Transcript:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeClient:
    def __init__(self, chunks: list[bytes], transcript: Any = None) -> None:
        self.text_to_speech = FakeTextToSpeech(chunks)
        self.speech_to_text = FakeSpeechToText(transcript or Transcript("hello there"))


@pytest.fixture
def client() -> FakeClient:
    return FakeClient([b"one", b"two", b"three"])


class TestTTSConfig:
    async def test_synthesize_buffers_the_non_streaming_endpoint(self, client: FakeClient) -> None:
        config = ElevenLabsTTSConfig(client=client)

        assert await config.synthesize("hello there") == b"onetwothree"
        assert client.text_to_speech.convert_calls == [
            {
                "voice_id": DEFAULT_VOICE_ID,
                "text": "hello there",
                "model_id": "eleven_v3",
                "output_format": "pcm_24000",
            }
        ]
        assert client.text_to_speech.stream_calls == []

    async def test_stream_yields_audio_before_the_full_response(self, client: FakeClient) -> None:
        config = ElevenLabsStreamingTTSConfig(client=client)

        async for chunk in config.stream("hello there"):
            assert chunk == b"one"
            assert client.text_to_speech.yielded == 1
            break

        assert client.text_to_speech.stream_calls == [
            {
                "voice_id": DEFAULT_VOICE_ID,
                "text": "hello there",
                "model_id": "eleven_flash_v2_5",
                "output_format": "pcm_24000",
            }
        ]
        assert client.text_to_speech.convert_calls == []

    async def test_streaming_config_can_still_buffer_a_complete_clip(self, client: FakeClient) -> None:
        assert await ElevenLabsStreamingTTSConfig(client=client).synthesize("hello there") == b"onetwothree"


class TestStreamingViaTTSObserver:
    async def test_speaks_a_complete_sentence_before_model_completion(self, client: FakeClient) -> None:
        stream = MemoryStream()
        audio: list[bytes] = []
        context = ConversationContext(stream=stream)
        observer = TTSObserver(ElevenLabsStreamingTTSConfig(client=client))

        async def collect(event: events.SynthesizedAudioEvent) -> None:
            audio.append(event.content)

        stream.where(events.SynthesizedAudioEvent).subscribe(collect)
        with ExitStack() as stack:
            observer.register(stack, context)
            await context.send(
                events.ModelMessageChunk(content="This sentence is long enough to cross the streaming threshold. ")
            )
            await asyncio.sleep(0.01)

        assert audio == [b"one", b"two", b"three"]


class TestSTTConfig:
    async def test_uploads_wav_and_emits_the_complete_transcript(self, client: FakeClient) -> None:
        stream = MemoryStream()
        seen: list[events.TranscriptionCompletedEvent] = []

        async def collect(event: events.TranscriptionCompletedEvent) -> None:
            seen.append(event)

        stream.where(events.TranscriptionCompletedEvent).subscribe(collect)
        text = await ElevenLabsTranscriber(client=client).transcribe(VOICE, ConversationContext(stream=stream))
        await asyncio.sleep(0.01)

        assert text == "hello there"
        assert seen == [events.TranscriptionCompletedEvent("hello there")]
        [call] = client.speech_to_text.convert_calls
        assert call == IsPartialDict({"model_id": "scribe_v2", "diarize": False})
        assert call["file"].startswith(b"RIFF")
        assert "language_code" not in call

    async def test_forwards_explicit_scribe_options(self, client: FakeClient) -> None:
        await ElevenLabsTranscriber("scribe_v2", language_code="fr", diarize=True, client=client).transcribe(
            VOICE,
            ConversationContext(stream=MemoryStream()),
        )

        assert client.speech_to_text.convert_calls == [
            IsPartialDict({"model_id": "scribe_v2", "language_code": "fr", "diarize": True})
        ]

    async def test_rejects_responses_without_transcript_text(self) -> None:
        with pytest.raises(TypeError, match="no transcript text"):
            await ElevenLabsTranscriber(client=FakeClient([], transcript=object())).transcribe(
                VOICE,
                ConversationContext(stream=MemoryStream()),
            )

    async def test_pipes_the_transcript_into_an_agent(self, client: FakeClient) -> None:
        tracking = testing.TrackingConfig(testing.TestConfig("Hi!"))
        reply = await ElevenLabsTranscriber(client=client).pipe(Agent("assistant", config=tracking)).ask(VOICE)

        assert reply.body == "Hi!"
        assert tracking.mock.call_args.args[0].parts[0].content == "hello there"
