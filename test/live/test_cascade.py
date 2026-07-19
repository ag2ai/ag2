# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import math
import struct
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from dirty_equals import IsInstance
from typing_extensions import Self

from ag2.annotations import Context
from ag2.config import LLMClient, ModelConfig, ModelProvider
from ag2.context import ConversationContext
from ag2.events import (
    AudioInterruptedEvent,
    AudioPlaybackCompletedEvent,
    AudioPlaybackStartedEvent,
    BaseEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelRequest,
    ModelResponse,
    RecordedAudioEvent,
    SynthesizedAudioEvent,
    ToolCallEvent,
    TranscriptionCompletedEvent,
    Usage,
)
from ag2.live import CascadeConfig, LiveAgent, SilenceTurnDetector, SoundDevicePlayer
from ag2.live.stt import STTConfig, VoiceInput
from ag2.stream import MemoryStream
from ag2.testing import TestConfig

SAMPLE_RATE = 24000


def pcm(seconds: float, amplitude: int) -> bytes:
    n = int(SAMPLE_RATE * seconds)
    return struct.pack(f"<{n}h", *[int(amplitude * math.sin(i * 0.1)) for i in range(n)])


def detector() -> SilenceTurnDetector:
    """Same policy as the default, scaled down so tests need less audio."""
    return SilenceTurnDetector(
        sample_rate=SAMPLE_RATE,
        silence=0.3,
        min_speech=0.1,
        prefix_padding=0.1,
    )


async def speak_one_turn(context: ConversationContext) -> None:
    """Push a full utterance — silence, speech, then the closing silence."""
    for _ in range(2):
        await context.send(RecordedAudioEvent(pcm(0.1, 0)))
    for _ in range(6):
        await context.send(RecordedAudioEvent(pcm(0.1, 6000)))
    for _ in range(4):
        await context.send(RecordedAudioEvent(pcm(0.1, 0)))


class FakeSTT(STTConfig):
    def __init__(self, text: str = "what is the weather") -> None:
        self.text = text
        self.calls: list[VoiceInput] = []

    async def transcribe(self, voice: VoiceInput, context: Context) -> str:
        self.calls.append(voice)
        await context.send(TranscriptionCompletedEvent(self.text))
        return self.text


class FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        self.spoken.append(text)
        return f"audio:{text}".encode()


class FakeStreamingTTS(FakeTTS):
    async def stream(self, text: str) -> AsyncIterator[bytes]:
        self.spoken.append(text)
        for word in text.split():
            yield f"audio:{word}".encode()


class RecordingConfig(ModelConfig):
    """`TrackingConfig`, but keeping every message list rather than the last
    message — these tests assert on what the whole history looked like."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.seen: list[Sequence[BaseEvent]] = []

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.OPENAI

    @property
    def model(self) -> str:
        return "test-model"

    def copy(self) -> Self:
        return self

    def create(self) -> LLMClient:
        inner = self.config.create()

        async def client(messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
            self.seen.append(list(messages))
            return await inner(messages, context=context, **kwargs)

        return client  # type: ignore[return-value]


class ChunkingConfig(ModelConfig):
    """Emits `ModelMessageChunk` the way a streaming provider config does,
    which `TestConfig` (whole-message only) cannot express."""

    def __init__(self, *chunks: str) -> None:
        self.chunks = chunks

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.OPENAI

    @property
    def model(self) -> str:
        return "test-model"

    def copy(self) -> Self:
        return self

    def create(self) -> LLMClient:
        async def client(messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
            for chunk in self.chunks:
                await context.send(ModelMessageChunk(chunk))
            return ModelResponse(ModelMessage("".join(self.chunks)))

        return client  # type: ignore[return-value]


def cascade(model: ModelConfig, stt: FakeSTT, tts: FakeTTS, **kwargs: Any) -> CascadeConfig:
    return CascadeConfig(
        stt=stt,
        model=model,
        tts=tts,  # type: ignore[arg-type]
        turn_detector=detector,
        **kwargs,
    )


@pytest.mark.asyncio
class TestCascadeSession:
    async def test_speaks_a_full_turn(self) -> None:
        """One utterance in, one spoken reply out — over the same events a
        native s2s session uses."""
        stt, tts = FakeSTT(), FakeTTS()
        context = ConversationContext(stream=MemoryStream())

        heard: list[bytes] = []
        context.stream.where(SynthesizedAudioEvent).subscribe(
            lambda e: heard.append(e.content),  # type: ignore[arg-type,return-value]
        )

        config = cascade(TestConfig("It is sunny."), stt, tts)
        async with config.session(context, serializer=None):  # type: ignore[arg-type]
            await speak_one_turn(context)
            await asyncio.sleep(0.05)

        assert len(stt.calls) == 1
        assert tts.spoken == ["It is sunny."]
        assert heard == [b"audio:It is sunny."]

    async def test_history_carries_the_conversation(self) -> None:
        """State lives in the stream's history, so a later turn's model call
        sees the earlier exchange without the session tracking it separately."""
        stt, tts = FakeSTT(), FakeTTS()
        model = RecordingConfig(TestConfig("First.", "Second."))
        context = ConversationContext(stream=MemoryStream())

        async with cascade(model, stt, tts).session(context, serializer=None):  # type: ignore[arg-type]
            await speak_one_turn(context)
            await asyncio.sleep(0.05)
            await speak_one_turn(context)
            await asyncio.sleep(0.05)

        requests = [e for e in model.seen[-1] if isinstance(e, ModelRequest)]
        responses = [e for e in model.seen[-1] if isinstance(e, ModelResponse)]
        assert [r.parts[0].content for r in requests] == ["what is the weather"] * 2  # type: ignore[union-attr]
        assert [r.content for r in responses] == ["First."]

    async def test_raw_audio_never_reaches_the_model(self) -> None:
        """A session logs ten-plus audio events per second; re-sending the
        recording on every turn would dwarf the transcript it accompanies."""
        stt, tts = FakeSTT(), FakeTTS()
        model = RecordingConfig(TestConfig("Understood."))
        context = ConversationContext(stream=MemoryStream())

        async with cascade(model, stt, tts).session(context, serializer=None):  # type: ignore[arg-type]
            await speak_one_turn(context)
            await asyncio.sleep(0.05)

        assert model.seen
        for messages in model.seen:
            assert not any(isinstance(e, (RecordedAudioEvent, SynthesizedAudioEvent)) for e in messages)

    async def test_instructions_reach_the_prompt(self) -> None:
        """`LiveAgent` hands the agent prompt to the session rather than the
        context, so the session installs it — and removes it again on exit."""
        stt, tts = FakeSTT(), FakeTTS()
        context = ConversationContext(stream=MemoryStream())

        config = cascade(TestConfig("Hi."), stt, tts)
        async with config.session(context, instructions=["Be brief."], serializer=None):  # type: ignore[arg-type]
            assert context.prompt == ["Be brief."]

        assert context.prompt == []

    async def test_emits_usage(self) -> None:
        stt, tts = FakeSTT(), FakeTTS()
        context = ConversationContext(stream=MemoryStream())

        response = ModelResponse(
            message=ModelMessage("Sure."),
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            model="test-model",
            provider="openai",
        )
        config = cascade(TestConfig(response), stt, tts)
        async with config.session(context, serializer=None):  # type: ignore[arg-type]
            await speak_one_turn(context)
            await asyncio.sleep(0.05)

        report = await LiveAgent.usage_report(context)
        assert report.total == Usage(prompt_tokens=10, completion_tokens=5)

    async def test_streaming_tts_speaks_per_sentence(self) -> None:
        """With a streaming TTS config, synthesis happens at sentence
        boundaries as the model produces text, not after the whole reply."""
        stt, tts = FakeSTT(), FakeStreamingTTS()
        model = ChunkingConfig(
            "The weather today is sunny and warm. ",
            "You will not need a coat.",
        )
        context = ConversationContext(stream=MemoryStream())

        # min_chars below either sentence's length, so each is sent on its own;
        # at the default of 60 the buffer batches both into one request, which
        # is what the threshold is for.
        config = cascade(model, stt, tts, min_chars=20)
        async with config.session(context, serializer=None):  # type: ignore[arg-type]
            await speak_one_turn(context)
            await asyncio.sleep(0.05)

        assert tts.spoken == [
            "The weather today is sunny and warm.",
            "You will not need a coat.",
        ]

    async def test_silence_produces_no_turn(self) -> None:
        stt, tts = FakeSTT(), FakeTTS()
        context = ConversationContext(stream=MemoryStream())

        async with cascade(TestConfig(), stt, tts).session(context, serializer=None):  # type: ignore[arg-type]
            for _ in range(10):
                await context.send(RecordedAudioEvent(pcm(0.1, 0)))
            await asyncio.sleep(0.05)

        assert stt.calls == []
        assert tts.spoken == []

    async def test_blip_below_min_speech_is_ignored(self) -> None:
        """A cough is not a turn — too short to be worth an STT round-trip."""
        stt, tts = FakeSTT(), FakeTTS()
        context = ConversationContext(stream=MemoryStream())

        async with cascade(TestConfig(), stt, tts).session(context, serializer=None):  # type: ignore[arg-type]
            await context.send(RecordedAudioEvent(pcm(0.05, 6000)))
            for _ in range(4):
                await context.send(RecordedAudioEvent(pcm(0.1, 0)))
            await asyncio.sleep(0.05)

        assert stt.calls == []


@pytest.mark.asyncio
class TestBargeIn:
    async def test_new_speech_interrupts_the_reply(self) -> None:
        """Talking over a reply cancels it and tells the player to drop audio
        it has already queued."""
        stt, tts = FakeSTT(), FakeTTS()
        started = asyncio.Event()

        class SlowConfig(ChunkingConfig):
            def create(self) -> LLMClient:
                async def client(messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
                    started.set()
                    await asyncio.sleep(10)  # outlives the barge-in
                    raise AssertionError("an interrupted turn must not finish")

                return client  # type: ignore[return-value]

        context = ConversationContext(stream=MemoryStream())
        interrupts: list[AudioInterruptedEvent] = []
        context.stream.where(AudioInterruptedEvent).subscribe(
            lambda e: interrupts.append(e),  # type: ignore[arg-type,return-value]
        )

        config = cascade(SlowConfig(), stt, tts, barge_in=True)
        async with config.session(context, serializer=None):  # type: ignore[arg-type]
            await speak_one_turn(context)
            await asyncio.wait_for(started.wait(), timeout=1)

            # A second utterance begins while the model is still thinking.
            await context.send(RecordedAudioEvent(pcm(0.1, 6000)))
            await asyncio.sleep(0.05)

        assert len(interrupts) == 1
        assert tts.spoken == []

    async def test_disabled_barge_in_lets_the_reply_finish(self) -> None:
        stt, tts = FakeSTT(), FakeTTS()
        context = ConversationContext(stream=MemoryStream())
        interrupts: list[AudioInterruptedEvent] = []
        context.stream.where(AudioInterruptedEvent).subscribe(
            lambda e: interrupts.append(e),  # type: ignore[arg-type,return-value]
        )

        config = cascade(TestConfig("Done.", "Again."), stt, tts, barge_in=False)
        async with config.session(context, serializer=None):  # type: ignore[arg-type]
            await speak_one_turn(context)
            await context.send(RecordedAudioEvent(pcm(0.1, 6000)))
            await asyncio.sleep(0.05)

        assert interrupts == []
        assert tts.spoken == ["Done."]


@pytest.mark.asyncio
class TestHalfDuplex:
    """On speakers the microphone hears the reply. A session that acts on that
    audio interrupts its own sentence and then answers its own words."""

    async def test_microphone_is_ignored_while_the_reply_plays(self) -> None:
        stt, tts = FakeSTT(), FakeTTS()
        context = ConversationContext(stream=MemoryStream())

        async with cascade(TestConfig("Done.", "Again."), stt, tts).session(context, serializer=None):  # type: ignore[arg-type]
            await context.send(AudioPlaybackStartedEvent())
            # The reply, coming back in through the mic as a full utterance.
            await speak_one_turn(context)
            await asyncio.sleep(0.05)

        assert stt.calls == []
        assert tts.spoken == []

    async def test_listens_again_once_the_speaker_falls_silent(self) -> None:
        stt, tts = FakeSTT(), FakeTTS()
        context = ConversationContext(stream=MemoryStream())

        async with cascade(TestConfig("Done.", "Again."), stt, tts).session(context, serializer=None):  # type: ignore[arg-type]
            await context.send(AudioPlaybackStartedEvent())
            await context.send(RecordedAudioEvent(pcm(0.1, 6000)))
            await context.send(AudioPlaybackCompletedEvent())

            await speak_one_turn(context)
            await asyncio.sleep(0.05)

        assert tts.spoken == ["Done."]

    async def test_barge_in_keeps_listening_through_playback(self) -> None:
        """Opting in means trusting the mic during playback — the gate is off."""
        stt, tts = FakeSTT(), FakeTTS()
        context = ConversationContext(stream=MemoryStream())

        config = cascade(TestConfig("Done.", "Again."), stt, tts, barge_in=True)
        async with config.session(context, serializer=None):  # type: ignore[arg-type]
            await context.send(AudioPlaybackStartedEvent())
            await speak_one_turn(context)
            await asyncio.sleep(0.05)

        assert stt.calls != []


@pytest.mark.asyncio
class TestLiveAgentIntegration:
    async def test_tool_call_round_trip(self) -> None:
        """Tool calls run through the executor `LiveAgent` registered, and the
        result feeds a second model call whose answer is spoken."""
        stt, tts = FakeSTT(), FakeTTS()
        called: list[str] = []

        def get_weather(city: str) -> str:
            called.append(city)
            return "sunny"

        model = TestConfig(
            ToolCallEvent(name="get_weather", arguments='{"city": "Berlin"}'),
            "It is sunny in Berlin.",
        )

        agent = LiveAgent(
            "assistant",
            config=cascade(model, stt, tts),
            tools=[get_weather],
        )

        async with agent.run() as context:
            await speak_one_turn(context)
            await asyncio.sleep(0.1)

        assert called == ["Berlin"]
        assert tts.spoken == ["It is sunny in Berlin."]

    async def test_drives_a_live_agent_like_any_realtime_config(self) -> None:
        """The whole point: `LiveAgent` needs no knowledge that this config is
        a cascade rather than a native speech-to-speech session."""
        stt, tts = FakeSTT(), FakeTTS()

        agent = LiveAgent(
            "assistant",
            prompt="You are terse.",
            config=cascade(TestConfig("Understood."), stt, tts),
        )

        async with agent.run() as context:
            await speak_one_turn(context)
            await asyncio.sleep(0.05)

        assert tts.spoken == ["Understood."]


class TestPlayerFlush:
    """`SoundDevicePlayer.clear` is what makes barge-in audible: stopping
    synthesis is not enough while queued audio is still on its way out."""

    def test_drops_queued_audio(self) -> None:
        player = SoundDevicePlayer()
        player._audio_queue.put(b"a")
        player._audio_queue.put(b"b")

        player.clear()

        assert player._audio_queue.qsize() == 0

    def test_keeps_the_stop_sentinel(self) -> None:
        """A flush racing `close` must not swallow the worker's stop signal —
        nor leave audio queued behind it."""
        player = SoundDevicePlayer()
        player._audio_queue.put(b"a")
        player._audio_queue.put(None)
        player._audio_queue.put(b"b")

        player.clear()

        assert player._audio_queue.qsize() == 1
        assert player._audio_queue.get() is None


class FakeOutputStream:
    """Stands in for `sd.OutputStream` so playback needs no audio device."""

    def __init__(self) -> None:
        self.writes: list[Any] = []

    def __enter__(self) -> "FakeOutputStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        pass

    def write(self, data: Any) -> None:
        self.writes.append(data)


@pytest.mark.asyncio
class TestPlaybackEvents:
    """The player is the only component that knows when sound is actually in
    the room — a half-duplex session gates its microphone on these."""

    async def test_reports_the_start_and_end_of_a_reply(self) -> None:
        context = ConversationContext(stream=MemoryStream())
        edges: list[BaseEvent] = []
        context.stream.where(AudioPlaybackStartedEvent | AudioPlaybackCompletedEvent).subscribe(
            lambda e: edges.append(e),  # type: ignore[arg-type,return-value]
        )

        async with SoundDevicePlayer(context=context, output_stream=FakeOutputStream()):  # type: ignore[arg-type]
            await context.send(SynthesizedAudioEvent(pcm(0.05, 6000)))
            await asyncio.sleep(0.5)  # outlasts the settle window

        assert edges == [IsInstance(AudioPlaybackStartedEvent), IsInstance(AudioPlaybackCompletedEvent)]

    async def test_a_gap_between_streamed_chunks_does_not_end_playback(self) -> None:
        """Streaming TTS arrives in bursts and the speaker outruns the network;
        an empty queue mid-reply must not read as silence."""
        context = ConversationContext(stream=MemoryStream())
        edges: list[BaseEvent] = []
        context.stream.where(AudioPlaybackStartedEvent | AudioPlaybackCompletedEvent).subscribe(
            lambda e: edges.append(e),  # type: ignore[arg-type,return-value]
        )

        async with SoundDevicePlayer(context=context, output_stream=FakeOutputStream()):  # type: ignore[arg-type]
            for _ in range(3):
                await context.send(SynthesizedAudioEvent(pcm(0.05, 6000)))
                await asyncio.sleep(0.1)  # shorter than the settle window
            await asyncio.sleep(0.05)

        assert edges == [IsInstance(AudioPlaybackStartedEvent)]
