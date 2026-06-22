---
title: "Voice & Realtime Overview"
sidebarTitle: "Overview"
---

`autogen.beta.live` is the AG2 Beta module for building voice-enabled agents. It covers two complementary patterns: a turn-by-turn **STT → Agent → TTS** pipeline built on top of a regular `Agent`, and a full-duplex **`LiveAgent`** that streams audio to and from a provider's realtime API.

## When to use which

| Pattern | Class | Latency | Use when… |
|---|---|---|---|
| Turn-by-turn voice | `Agent` + `OpenAITranscriber` + `TTSObserver` | ~1–3 s per turn | You already have a text agent and want to add a voice front-end; you need tool execution, middleware, structured output, or any other text-agent feature. |
| Realtime full-duplex | `LiveAgent` + `OpenAIRealTimeConfig` / `GeminiRealTimeConfig` | <500 ms | You want barge-in, interruption, semantic VAD, or a phone-call-like UX. |

!!! note
    Both patterns share the same audio I/O primitives — [`SoundDevicePlayer`](stt_tts.md) and [`SoundDeviceRecorder`](stt_tts.md) — and the same event stream. You can mix observers (TTS, logging, persistence) across both.

## Installation

The audio I/O classes depend on `sounddevice` and `numpy`; the OpenAI and Gemini integrations need their respective SDKs.

```bash
pip install "ag2[openai,gemini] sounddevice[numpy]"
```

!!! warning
    `SoundDevicePlayer` and `SoundDeviceRecorder` require `sounddevice[numpy]` as an **additional** dependency (not optional)

## The two flows at a glance

=== "Turn-by-turn (STT → Agent → TTS)"
    ```python linenums="1"
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
        observers=[TTSObserver(config=OpenAITTSConfig(model="gpt-4o-mini-tts"))],
    )

    async def main() -> None:
        pipeline = OpenAITranscriber("gpt-4o-mini-transcribe").pipe(agent)

        async with SoundDevicePlayer() as player:
            voice = SoundDeviceRecorder().record(duration=3)
            reply = await pipeline.ask(voice, stream=player.stream)
            print(reply.body)

    if __name__ == "__main__":
        asyncio.run(main())
    ```

=== "Realtime (LiveAgent)"
    ```python linenums="1"
    import asyncio

    from autogen.beta.live import (
        LiveAgent,
        SoundDevicePlayer,
        SoundDeviceRecorder,
        openai,
    )

    agent = LiveAgent(
        name="assistant",
        prompt="You are a helpful voice assistant.",
        config=openai.RealTimeConfig(
            "gpt-realtime-2",
            output=openai.AudioOutput(voice="ballad", speed=1.2),
        ),
    )

    async def main() -> None:
        async with (
            agent.run() as context,
            SoundDevicePlayer(context=context),
            SoundDeviceRecorder(context=context),
        ):
            print("Starting...")
            await asyncio.Future()

    if __name__ == "__main__":
        asyncio.run(main())
    ```

## What's next

- **[STT & TTS](stt_tts.md)** — wrap any `Agent` with speech input and output.
- **[LiveAgent](live_agent.md)** — realtime, low-latency voice agents with OpenAI or Gemini.
