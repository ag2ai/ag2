# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import math
import struct
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from typing_extensions import Self

from ag2.annotations import Context
from ag2.config import LLMClient, ModelConfig, ModelProvider
from ag2.context import ConversationContext, StreamId
from ag2.events import (
    AudioInterruptedEvent,
    AudioPlaybackCompletedEvent,
    AudioPlaybackStartedEvent,
    BaseEvent,
    DataInput,
    ImageInput,
    ModelMessage,
    ModelMessageChunk,
    ModelRequest,
    ModelResponse,
    RecordedAudioEvent,
    SynthesizedAudioEvent,
    TextInput,
    ToolCallEvent,
    TranscriptionCompletedEvent,
    UrlInput,
    Usage,
    VoiceTurn,
)
from ag2.exceptions import UnsupportedInputError
from ag2.history import MemoryStorage
from ag2.live import CascadeConfig, LiveAgent, SilenceTurnDetector
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

    def create_files_client(self) -> Any:
        raise NotImplementedError

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

    def create_files_client(self) -> Any:
        raise NotImplementedError

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

    async def test_pushes_waiting_on_an_interrupted_reply_are_answered_after_the_utterance(self) -> None:
        """Talking over a reply cancels it, not the pushed input queued behind
        it. That input is answered once the user stops talking — here, a cough
        the STT finds no words in."""
        stt, tts = FakeSTT(), FakeTTS()
        model = HeldConfig("Sunny today.", "Rain tomorrow.")
        context = ConversationContext(stream=MemoryStream())

        config = cascade(model, stt, tts, barge_in=True)
        async with config.session(context, serializer=None):  # type: ignore[arg-type]
            await speak_one_turn(context)
            await asyncio.wait_for(model.started.wait(), timeout=1)
            await context.send(ModelRequest([TextInput("and tomorrow?")]))

            stt.text = ""
            await context.send(RecordedAudioEvent(pcm(0.05, 6000)))
            await asyncio.sleep(0.05)
            assert len(model.seen) == 1

            for _ in range(4):
                await context.send(RecordedAudioEvent(pcm(0.1, 0)))
            await asyncio.sleep(0.05)

        assert tts.spoken == ["Rain tomorrow."]
        assert requests(model.seen[-1]) == [["what is the weather"], ["and tomorrow?"]]

    async def test_pushes_waiting_on_an_interrupted_reply_are_answered_after_a_sound_too_short_for_a_turn(
        self,
    ) -> None:
        stt, tts = FakeSTT(), FakeTTS()
        model = HeldConfig("Sunny today.", "Rain tomorrow.")
        context = ConversationContext(stream=MemoryStream())

        config = CascadeConfig(
            stt=stt,
            model=model,
            tts=tts,  # type: ignore[arg-type]
            turn_detector=lambda: SilenceTurnDetector(
                sample_rate=SAMPLE_RATE, silence=0.3, min_speech=0.5, prefix_padding=0.1
            ),
            barge_in=True,
        )
        async with config.session(context, serializer=None):  # type: ignore[arg-type]
            await speak_one_turn(context)
            await asyncio.wait_for(model.started.wait(), timeout=1)
            await context.send(ModelRequest([TextInput("and tomorrow?")]))

            await context.send(RecordedAudioEvent(pcm(0.05, 6000)))
            for _ in range(4):
                await context.send(RecordedAudioEvent(pcm(0.1, 0)))
            await asyncio.sleep(0.05)

        assert stt.calls[1:] == []
        assert tts.spoken == ["Rain tomorrow."]

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


class HeldConfig(ModelConfig):
    """Answers each call with the next scripted text, holding the first call
    until `release` is set — a voice turn still in flight."""

    def __init__(self, *answers: str) -> None:
        self.answers = answers
        self.seen: list[Sequence[BaseEvent]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.OPENAI

    @property
    def model(self) -> str:
        return "test-model"

    def copy(self) -> Self:
        return self

    def create_files_client(self) -> Any:
        raise NotImplementedError

    def create(self) -> LLMClient:
        async def client(messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
            call = len(self.seen)
            self.seen.append(list(messages))
            self.started.set()
            if call == 0:
                await self.release.wait()
            return ModelResponse(ModelMessage(self.answers[call]))

        return client  # type: ignore[return-value]


class HeldSTT(FakeSTT):
    """Transcribes once `release` is set — the user's words still being recognised."""

    def __init__(self, text: str = "what is the weather") -> None:
        super().__init__(text)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def transcribe(self, voice: VoiceInput, context: Context) -> str:
        self.started.set()
        await self.release.wait()
        return await super().transcribe(voice, context)


class HeldStorage(MemoryStorage):
    """Holds the first history read until `release` is set — storage that
    yields to the event loop, as any networked backend does."""

    def __init__(self) -> None:
        super().__init__()
        self.reading = asyncio.Event()
        self.release = asyncio.Event()

    async def get_history(self, stream_id: StreamId) -> Any:
        if not self.reading.is_set():
            self.reading.set()
            await self.release.wait()
        return await super().get_history(stream_id)


def requests(messages: Sequence[BaseEvent]) -> list[list[Any]]:
    return [
        [p.content if isinstance(p, TextInput) else p for p in e.parts] for e in messages if isinstance(e, ModelRequest)
    ]


@pytest.mark.asyncio
class TestPushedInput:
    async def test_pushed_request_is_answered_aloud(self) -> None:
        """Nothing enforces the `ModelRequest` subscription, so this guards it:
        a push into a silent session reaches the model and is spoken."""
        stt, tts = FakeSTT(), FakeTTS()
        context = ConversationContext(stream=MemoryStream())

        async with cascade(TestConfig("Hi there."), stt, tts).session(context, serializer=None):  # type: ignore[arg-type]
            await context.send(ModelRequest([TextInput("hello")]))
            await asyncio.sleep(0.05)

        assert stt.calls == []
        assert tts.spoken == ["Hi there."]

    async def test_pushed_data_reaches_the_model(self) -> None:
        stt, tts = FakeSTT(), FakeTTS()
        model = RecordingConfig(TestConfig("Booked."))
        context = ConversationContext(stream=MemoryStream())

        async with cascade(model, stt, tts).session(context, serializer=None):  # type: ignore[arg-type]
            await context.send(ModelRequest([DataInput({"city": "Berlin"})]))
            await asyncio.sleep(0.05)

        assert requests(model.seen[-1]) == [[DataInput({"city": "Berlin"})]]

    async def test_push_during_voice_turn_is_answered_after_it(self) -> None:
        stt, tts = FakeSTT(), FakeTTS()
        model = HeldConfig("Sunny today.", "Rain tomorrow.")
        context = ConversationContext(stream=MemoryStream())

        async with cascade(model, stt, tts).session(context, serializer=None):  # type: ignore[arg-type]
            await speak_one_turn(context)
            await asyncio.wait_for(model.started.wait(), timeout=1)

            await context.send(ModelRequest([TextInput("and tomorrow?")]))
            await asyncio.sleep(0.05)
            assert len(model.seen) == 1

            model.release.set()
            await asyncio.sleep(0.05)

        assert tts.spoken == ["Sunny today.", "Rain tomorrow."]

    async def test_several_pushes_waiting_on_a_turn_get_one_answer(self) -> None:
        stt, tts = FakeSTT(), FakeTTS()
        model = HeldConfig("Sunny today.", "Rain, then snow.")
        context = ConversationContext(stream=MemoryStream())

        async with cascade(model, stt, tts).session(context, serializer=None):  # type: ignore[arg-type]
            await speak_one_turn(context)
            await asyncio.wait_for(model.started.wait(), timeout=1)

            await context.send(ModelRequest([TextInput("and tomorrow?")]))
            await context.send(ModelRequest([TextInput("and the day after?")]))
            model.release.set()
            await asyncio.sleep(0.05)

        assert len(model.seen) == 2
        assert requests(model.seen[-1]) == [["what is the weather"], ["and tomorrow?"], ["and the day after?"]]
        assert tts.spoken == ["Sunny today.", "Rain, then snow."]

    async def test_transcript_is_published_as_a_voice_turn(self) -> None:
        """Observers and history see the transcript as a `ModelRequest`, and
        can tell it apart from pushed input."""
        stt, tts = FakeSTT(), FakeTTS()
        context = ConversationContext(stream=MemoryStream())
        published: list[ModelRequest] = []
        context.stream.where(ModelRequest).subscribe(
            lambda e: published.append(e),  # type: ignore[arg-type,return-value]
        )

        async with cascade(TestConfig("Sunny."), stt, tts).session(context, serializer=None):  # type: ignore[arg-type]
            await speak_one_turn(context)
            await asyncio.sleep(0.05)

        assert [type(e) for e in published] == [VoiceTurn]

    async def test_voice_turn_on_the_stream_is_not_answered(self) -> None:
        """A `VoiceTurn` is answered by the turn that transcribed it, never as pushed input."""
        stt, tts = FakeSTT(), FakeTTS()
        model = RecordingConfig(TestConfig("Hi there."))
        context = ConversationContext(stream=MemoryStream())

        async with cascade(model, stt, tts).session(context, serializer=None):  # type: ignore[arg-type]
            await context.send(VoiceTurn([TextInput("hello")]))
            await asyncio.sleep(0.05)

        assert model.seen == []
        assert tts.spoken == []

    async def test_push_read_by_a_running_turn_is_not_answered_again(self) -> None:
        """A push stored while a turn is reading history is part of what that
        turn's model call reads, so its answer covers the push."""
        stt, tts = FakeSTT(), FakeTTS()
        model = RecordingConfig(TestConfig("Sunny, and rain tomorrow.", "Rain tomorrow."))
        storage = HeldStorage()
        context = ConversationContext(stream=MemoryStream(storage))

        async with cascade(model, stt, tts).session(context, serializer=None):  # type: ignore[arg-type]
            await speak_one_turn(context)
            await asyncio.wait_for(storage.reading.wait(), timeout=1)

            await context.send(ModelRequest([TextInput("and tomorrow?")]))
            storage.release.set()
            await asyncio.sleep(0.05)

        assert requests(model.seen[0]) == [["what is the weather"], ["and tomorrow?"]]
        assert tts.spoken == ["Sunny, and rain tomorrow."]

    async def test_push_read_by_a_failed_model_call_is_answered_after_it(self) -> None:
        stt, tts = HeldSTT(), FakeTTS()
        model = RecordingConfig(TestConfig(RuntimeError("model down"), "Rain tomorrow."))
        context = ConversationContext(stream=MemoryStream())

        async with cascade(model, stt, tts).session(context, serializer=None):  # type: ignore[arg-type]
            await speak_one_turn(context)
            await asyncio.wait_for(stt.started.wait(), timeout=1)

            await context.send(ModelRequest([TextInput("and tomorrow?")]))
            stt.release.set()
            await asyncio.sleep(0.05)

        assert requests(model.seen[0]) == [["and tomorrow?"], ["what is the weather"]]
        assert tts.spoken == ["Rain tomorrow."]

    async def test_typed_and_spoken_turns_share_one_conversation(self) -> None:
        stt, tts = FakeSTT(), FakeTTS()
        model = RecordingConfig(TestConfig("Noted, Nik.", "Sunny, Nik."))
        context = ConversationContext(stream=MemoryStream())

        async with cascade(model, stt, tts).session(context, serializer=None):  # type: ignore[arg-type]
            await context.send(ModelRequest([TextInput("call me Nik")]))
            await asyncio.sleep(0.05)
            await speak_one_turn(context)
            await asyncio.sleep(0.05)

        assert requests(model.seen[-1]) == [["call me Nik"], ["what is the weather"]]
        assert [e.content for e in model.seen[-1] if isinstance(e, ModelResponse)] == ["Noted, Nik."]
        assert tts.spoken == ["Noted, Nik.", "Sunny, Nik."]


@pytest.mark.asyncio
class TestUnsupportedInput:
    async def test_pushed_media_is_refused_to_the_caller(self) -> None:
        stt, tts = FakeSTT(), FakeTTS()
        context = ConversationContext(stream=MemoryStream())

        async with cascade(TestConfig("Nice cat."), stt, tts).session(context, serializer=None):  # type: ignore[arg-type]
            with pytest.raises(UnsupportedInputError, match=r"UrlInput\(image\)"):
                await context.send(ModelRequest([TextInput("look"), ImageInput("https://example.com/cat.png")]))
            await asyncio.sleep(0.05)

        assert tts.spoken == []

    async def test_refused_media_stays_out_of_later_turns(self) -> None:
        """The refused request is in the stream's history, which the cascade
        rebuilds the conversation from on every turn: its media never reaches
        the model, its text joins the conversation."""
        stt, tts = FakeSTT(), FakeTTS()
        model = RecordingConfig(TestConfig("Sunny."))
        context = ConversationContext(stream=MemoryStream())

        async with cascade(model, stt, tts).session(context, serializer=None):  # type: ignore[arg-type]
            with pytest.raises(UnsupportedInputError):
                await context.send(ModelRequest([TextInput("look"), ImageInput("https://example.com/cat.png")]))
            await speak_one_turn(context)
            await asyncio.sleep(0.05)

        assert requests(model.seen[-1]) == [["look"], ["what is the weather"]]

    async def test_drained_media_is_dropped_once_and_the_rest_answered(self, caplog: pytest.LogCaptureFixture) -> None:
        """A drain has no caller to refuse: the text is answered, the image is
        dropped with one warning, and no later turn trips over it."""
        stt, tts = FakeSTT(), FakeTTS()
        model = RecordingConfig(TestConfig("I can't see images.", "Sunny."))
        agent = LiveAgent("assistant", config=cascade(model, stt, tts))

        with caplog.at_level(logging.WARNING):
            async with agent.run() as context:
                context.enqueue("look at this", ImageInput("https://example.com/cat.png"))
                await asyncio.sleep(0.05)
                await speak_one_turn(context)
                await asyncio.sleep(0.05)

        assert tts.spoken == ["I can't see images.", "Sunny."]
        assert requests(model.seen[-1]) == [["look at this"], ["what is the weather"]]
        assert not any(isinstance(p, UrlInput) for messages in model.seen for p in _parts(messages))
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "UrlInput(image)" in r.getMessage()]
        assert len(warnings) == 1

    async def test_drained_message_with_only_media_is_not_answered(self) -> None:
        stt, tts = FakeSTT(), FakeTTS()
        model = RecordingConfig(TestConfig("Nothing to say."))
        agent = LiveAgent("assistant", config=cascade(model, stt, tts))

        async with agent.run() as context:
            context.enqueue(ImageInput("https://example.com/cat.png"))
            await asyncio.sleep(0.05)

        assert model.seen == []
        assert tts.spoken == []


def _parts(messages: Sequence[BaseEvent]) -> list[Any]:
    return [p for e in messages if isinstance(e, ModelRequest) for p in e.parts]
