# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A cascade (STT + LLM + TTS) driven as if it were a speech-to-speech model.

`CascadeConfig` implements the same `RealtimeConfig` protocol as
S2S implementations of openai and gemini models.

Supports interruptions without any VAD models - so literally any input can interrupt the model speaking.
Supports modular construction of the session - TTS, LLM and STT can be three different providers.
Needs ELEVENLABS_API_KEY and OPENAI_API_KEY.

    python examples/stt_tts.py/cascade_live_agent.py
"""

import asyncio
from datetime import datetime

from ag2 import config
from ag2.events import ModelResponse, TranscriptionCompletedEvent
from ag2.live import (
    CascadeConfig,
    ElevenLabsStreamingTTSConfig,
    LiveAgent,
    OpenAITranscriber,
    SilenceTurnDetector,
    SoundDevicePlayer,
    SoundDeviceRecorder,
)


def current_time() -> str:
    """What time is it right now?"""
    return datetime.now().strftime("%H:%M")


def build_agent() -> LiveAgent:
    cascade = CascadeConfig(
        stt=OpenAITranscriber("gpt-4o-mini-transcribe"),
        model=config.OpenAIConfig("gpt-5-mini", streaming=True),
        tts=ElevenLabsStreamingTTSConfig("eleven_flash_v2_5"),
        turn_detector=lambda: SilenceTurnDetector(),
        # barge_in=True, # enables interrupts, care for speakers.
    )

    return LiveAgent(
        "assistant",
        prompt="You are a voice assistant. Answer in one or two short sentences.",
        config=cascade,
        tools=[current_time],
    )


async def main() -> None:
    agent = build_agent()

    async with (
        agent.run() as context,
        SoundDevicePlayer(context=context),
        SoundDeviceRecorder(context=context),
    ):

        async def on_transcript(event: TranscriptionCompletedEvent) -> None:
            print(f'  you: "{event.content}"')

        async def on_reply(event: ModelResponse) -> None:
            if event.content:
                print(f"  bot: {event.content}")

        context.stream.where(TranscriptionCompletedEvent).subscribe(on_transcript)
        context.stream.where(ModelResponse).subscribe(on_reply)

        print("Listening — say something. Ctrl+C to stop.\n")
        await asyncio.Future()  # run until interrupted

    print(f"\nUsage: {await LiveAgent.usage_report(context)}")


def swap_to_s2s() -> LiveAgent:
    """The same agent on a native speech-to-speech model.

    Only `config` changes — prompt, tools, recorder, and player are untouched.
    That is the property `CascadeConfig` exists to provide.
    """
    from ag2.live import OpenAIRealTimeConfig

    return LiveAgent(
        "assistant",
        prompt="You are a voice assistant. Answer in one or two short sentences.",
        config=OpenAIRealTimeConfig("gpt-realtime"),
        tools=[current_time],
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
