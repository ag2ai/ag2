---
title: "Speech-to-Text and Text-to-Speech"
sidebarTitle: "STT & TTS"
---

The STT → Agent → TTS flow turns any existing `Agent` into a voice agent without changing the agent itself. Speech-to-text is added as a **pipeline wrapper**; text-to-speech is added as an **observer** that listens to the model's streamed message chunks.

## Audio I/O primitives

`SoundDeviceRecorder` captures microphone input and `SoundDevicePlayer` plays synthesized speech. Both are thin wrappers around the [`sounddevice`](https://python-sounddevice.readthedocs.io/){.external-link target="_blank"} library and share the same event stream.

```python linenums="1"
from autogen.beta.live import SoundDevicePlayer, SoundDeviceRecorder

recorder = SoundDeviceRecorder()
voice = recorder.record(duration=5)  # blocks for 5s, returns VoiceInput
```

The recorder produces a `VoiceInput` containing 16-bit PCM bytes plus the sample rate and channel count. The player subscribes to `SynthesizedAudioEvent` on its context's stream and plays each chunk on a background thread.

!!! note
    `Recorder.record(duration=...)` is a one-shot, blocking helper for the turn-by-turn flow. For continuous streaming (used by `LiveAgent`), use the recorder as an async context manager — see [LiveAgent](live_agent.md).

## Speech-to-Text

`OpenAITranscriber` implements the `STTConfig` protocol and exposes a `.pipe(agent)` method that wraps an `Agent` in a `VoicePipeline`. Calling `pipeline.ask(voice)` transcribes the audio and forwards the text to the agent's normal `ask()` flow.

```python linenums="1" hl_lines="13 18"
import asyncio

from autogen.beta import Agent, config
from autogen.beta.live import OpenAITranscriber, SoundDeviceRecorder

agent = Agent(
    "assistant",
    config=config.OpenAIConfig("gpt-5", streaming=True),
)

async def main():
    # pipe STT model to agent input
    pipeline = OpenAITranscriber("gpt-4o-mini-transcribe").pipe(agent)
    recorder = SoundDeviceRecorder()

    print("Say something...")
    voice_input = recorder.record(duration=5)
    reply = await pipeline.ask(voice_input)
    print(reply.body)

    print("Say something...")
    voice_input = recorder.record(duration=5)
    # continue the same conversation
    reply = await reply.ask(voice_input)
    print(reply.body)

if __name__ == "__main__":
    asyncio.run(main())
```

`pipeline.ask(...)` returns a `VoiceReply` that exposes the same surface as `AgentReply` (`.body`, `.response`, `.history`, `.ask(...)`) plus `.ask(voice_input)` for the next voice turn. The agent's history is preserved across turns.

!!! tip
    The transcriber emits `TranscriptionChunkEvent` and `TranscriptionCompletedEvent` on the agent's stream as soon as the transcription server starts producing tokens. Subscribe to them to display live captions.

### Translation

If you want the user's speech transcribed **into English** regardless of input language, swap in `OpenAITranslationTranscriber`. It has the same API as `OpenAITranscriber` but uses OpenAI's translation endpoint.

```python linenums="1"
from autogen.beta.live import OpenAITranslationTranscriber

pipeline = OpenAITranslationTranscriber("whisper-1").pipe(agent)
```

## Text-to-Speech

`TTSObserver` is an observer that listens to `ModelMessageChunk` events as the agent streams its response, batches them into sentence-sized chunks, calls a TTS provider, and emits `SynthesizedAudioEvent`s onto the stream. A `SoundDevicePlayer` attached to the same stream then plays them.

```python linenums="1" hl_lines="10-12 16-18"
import asyncio

from autogen.beta import Agent, config
from autogen.beta.live import OpenAITTSConfig, SoundDevicePlayer, TTSObserver

agent = Agent(
    name="assistant",
    prompt="You are a helpful voice assistant.",
    config=config.OpenAIResponsesConfig(model="gpt-5", streaming=True),
    observers=[
        TTSObserver(config=OpenAITTSConfig(model="gpt-4o-mini-tts")),
    ],
)

async def main() -> None:
    async with SoundDevicePlayer() as player:
        # pass the player's stream so synthesized audio reaches the speakers
        await agent.ask("Hello, agent!", stream=player.stream)

if __name__ == "__main__":
    asyncio.run(main())
```

!!! warning
    The agent's `config` must be set up for **streaming output** (e.g., `streaming=True`). `TTSObserver` works at the `ModelMessageChunk` granularity — if the model emits a single non-streaming `ModelMessage`, the observer will still synthesize it, but you lose the sentence-level pipelining that keeps latency low.

### Voice and speed

`OpenAITTSConfig` accepts the standard OpenAI TTS parameters:

```python linenums="1"
from autogen.beta.live import OpenAITTSConfig

config = OpenAITTSConfig(
    model="gpt-4o-mini-tts",
    voice="ballad",  # alloy, ash, ballad, coral, echo, sage, shimmer, verse...
    speed=1.1,
)
```

## Combining STT and TTS

The full round-trip — voice in, voice out — is just both halves wired up at once: pipe the agent through the transcriber, attach a `TTSObserver`, and share a stream with the player.

```python linenums="1" hl_lines="16-18 22 25 28"
import asyncio

from autogen.beta import Agent, config
from autogen.beta.live import (
    OpenAITTSConfig,
    OpenAITranscriber,
    SoundDevicePlayer,
    SoundDeviceRecorder,
    TTSObserver,
)

agent = Agent(
    name="assistant",
    prompt="You are a helpful voice assistant.",
    config=config.OpenAIResponsesConfig(model="gpt-5", streaming=True),
    observers=[
        TTSObserver(config=OpenAITTSConfig(model="tts-1")),
    ],
)

async def main():
    pipeline = OpenAITranscriber("gpt-4o-mini-transcribe").pipe(agent)
    recorder = SoundDeviceRecorder()

    async with SoundDevicePlayer() as player:
        print("Say something...")
        voice_input = recorder.record(duration=3)
        reply = await pipeline.ask(voice_input, stream=player.stream)
        print(reply.body)

        # wait for the audio to finish playing
        player.join()

        print("Say something...")
        voice_input = recorder.record(duration=3)
        reply = await reply.ask(voice_input)
        print(reply.body)

if __name__ == "__main__":
    asyncio.run(main())
```

!!! tip
    `player.join()` blocks the main task until the synthesized audio queue has drained. Use it between turns when you want the assistant to finish speaking before the next recording starts — otherwise the recorder will capture the tail of the assistant's voice.

## What's next

- **[LiveAgent](live_agent.md)** — drop the turn-by-turn round-trip in favor of a streaming, full-duplex realtime session.
- **[Observers](../advanced/observers.md)** — `TTSObserver` is one of many observer patterns; see the harness docs for logging, persistence, and custom observers.
