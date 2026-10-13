# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import struct
from collections.abc import AsyncIterator
from contextlib import ExitStack
from typing import Any

import pytest

pytest.importorskip("mistralai")

from mistralai.client.models import (
    SpeechResponse,
    SpeechStreamAudioDelta,
    SpeechStreamDone,
    SpeechStreamEvents,
    UsageInfoDollarDefs,
)

from ag2 import MemoryStream, events
from ag2.context import ConversationContext
from ag2.live import MistralStreamingTTSConfig, MistralTTSConfig, TTSObserver
from ag2.live.mistral import DEFAULT_MODEL
from ag2.live.protocols import StreamingTTSConfig

pytestmark = [pytest.mark.asyncio, pytest.mark.mistral]

FLOAT32 = struct.pack("<4f", 0.0, 0.5, -0.5, 1.0)
INT16 = struct.pack("<4h", 0, 16384, -16384, 32767)

SENTENCE = "This sentence is long enough to cross the streaming threshold. "


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


class FakeSpeech:
    def __init__(self, deltas: list[bytes]) -> None:
        self._deltas = deltas
        self.calls: list[dict[str, Any]] = []
        self.yielded = 0

    async def complete_async(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not kwargs["stream"]:
            return SpeechResponse(audio_data=b64(b"".join(self._deltas)))
        return self._events()

    async def _events(self) -> AsyncIterator[SpeechStreamEvents]:
        for delta in self._deltas:
            self.yielded += 1
            yield SpeechStreamEvents(
                event="speech.audio.delta",
                data=SpeechStreamAudioDelta(audio_data=b64(delta)),
            )
        yield SpeechStreamEvents(
            event="speech.audio.done",
            data=SpeechStreamDone(usage=UsageInfoDollarDefs(prompt_tokens=1, total_tokens=1)),
        )


class FakeClient:
    def __init__(self, deltas: list[bytes]) -> None:
        self.speech = FakeSpeech(deltas)


@pytest.fixture
def client() -> FakeClient:
    return FakeClient([FLOAT32[:8], FLOAT32[8:]])


async def speak(config: MistralTTSConfig) -> list[bytes]:
    stream = MemoryStream()
    audio: list[bytes] = []
    context = ConversationContext(stream=stream)

    async def collect(event: events.SynthesizedAudioEvent) -> None:
        audio.append(event.content)

    stream.where(events.SynthesizedAudioEvent).subscribe(collect)
    with ExitStack() as stack:
        TTSObserver(config).register(stack, context)
        await context.send(events.ModelMessageChunk(content=SENTENCE))
        await context.send(events.ModelResponse(events.ModelMessage(SENTENCE)))
        await asyncio.sleep(0.01)

    return audio


@pytest.mark.parametrize("config_cls", [MistralTTSConfig, MistralStreamingTTSConfig])
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
        self, config_cls: type[MistralTTSConfig], client: FakeClient, voice: dict[str, str]
    ) -> None:
        with pytest.raises(ValueError, match="exactly one of"):
            config_cls(client=client, **voice)  # type: ignore[arg-type]

    async def test_sends_only_the_saved_voice(self, config_cls: type[MistralTTSConfig], client: FakeClient) -> None:
        config = config_cls(voice_id="v1", ref_audio="", client=client)  # type: ignore[arg-type]

        await config.synthesize("hi")

        assert client.speech.calls == [
            {
                "input": "hi",
                "model": DEFAULT_MODEL,
                "response_format": "pcm",
                "stream": config_cls is MistralStreamingTTSConfig,
                "voice_id": "v1",
            }
        ]

    async def test_sends_only_the_reference_clip(self, config_cls: type[MistralTTSConfig], client: FakeClient) -> None:
        config = config_cls("custom-model", ref_audio="clip", client=client)  # type: ignore[arg-type]

        await config.synthesize("hi")

        assert client.speech.calls == [
            {
                "input": "hi",
                "model": "custom-model",
                "response_format": "pcm",
                "stream": config_cls is MistralStreamingTTSConfig,
                "ref_audio": "clip",
            }
        ]


class TestTTSConfig:
    async def test_returns_int16_pcm_converted_from_float32(self, client: FakeClient) -> None:
        config = MistralTTSConfig(voice_id="v", client=client)  # type: ignore[arg-type]

        assert await config.synthesize("hi") == INT16

    async def test_clips_out_of_range_samples(self) -> None:
        config = MistralTTSConfig(
            voice_id="v",
            client=FakeClient([struct.pack("<2f", 2.0, -2.0)]),  # type: ignore[arg-type]
        )

        assert await config.synthesize("hi") == struct.pack("<2h", 32767, -32767)

    async def test_rejects_a_truncated_final_sample(self) -> None:
        config = MistralTTSConfig(voice_id="v", client=FakeClient([FLOAT32[:-1]]))  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="multiple of 4"):
            await config.synthesize("hi")

    async def test_is_not_a_streaming_config(self, client: FakeClient) -> None:
        assert not isinstance(MistralTTSConfig(voice_id="v", client=client), StreamingTTSConfig)  # type: ignore[arg-type]


class TestStreamingTTSConfig:
    async def test_is_a_streaming_config(self, client: FakeClient) -> None:
        assert isinstance(MistralStreamingTTSConfig(voice_id="v", client=client), StreamingTTSConfig)  # type: ignore[arg-type]

    async def test_yields_audio_before_the_full_response(self, client: FakeClient) -> None:
        config = MistralStreamingTTSConfig(voice_id="v", client=client)  # type: ignore[arg-type]

        async for chunk in config.stream("hi"):
            assert chunk == INT16[:4]
            assert client.speech.yielded == 1
            break

    async def test_yields_int16_chunks_skipping_non_audio_events(self, client: FakeClient) -> None:
        config = MistralStreamingTTSConfig(voice_id="v", client=client)  # type: ignore[arg-type]

        assert [chunk async for chunk in config.stream("hi")] == [INT16[:4], INT16[4:]]

    async def test_carries_a_split_sample_over_to_the_next_chunk(self) -> None:
        config = MistralStreamingTTSConfig(
            voice_id="v",
            client=FakeClient([FLOAT32[:6], FLOAT32[6:]]),  # type: ignore[arg-type]
        )

        chunks = [chunk async for chunk in config.stream("hi")]

        assert chunks == [INT16[:2], INT16[2:]]

    async def test_rejects_a_stream_that_ends_mid_sample(self) -> None:
        config = MistralStreamingTTSConfig(voice_id="v", client=FakeClient([FLOAT32[:-1]]))  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="mid-sample"):
            await config.synthesize("hi")

    async def test_synthesize_buffers_the_stream(self, client: FakeClient) -> None:
        config = MistralStreamingTTSConfig(voice_id="v", client=client)  # type: ignore[arg-type]

        assert await config.synthesize("hi") == INT16
        assert [call["stream"] for call in client.speech.calls] == [True]


class TestViaTTSObserver:
    async def test_streaming_config_speaks_int16_chunks(self, client: FakeClient) -> None:
        config = MistralStreamingTTSConfig(voice_id="v", client=client)  # type: ignore[arg-type]

        assert await speak(config) == [INT16[:4], INT16[4:]]

    async def test_buffered_config_speaks_one_int16_clip(self, client: FakeClient) -> None:
        config = MistralTTSConfig(voice_id="v", client=client)  # type: ignore[arg-type]

        assert await speak(config) == [INT16]
