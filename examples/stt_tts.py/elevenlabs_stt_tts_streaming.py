"""ElevenLabs voice round-trip with streaming TTS. No arguments.

The output side delivers audio incrementally instead of waiting for the whole
response:

  * `ElevenLabsStreamingTTSConfig` — audio chunks as they are generated, so
    playback starts at time-to-first-byte rather than after the last sample
  * `ElevenLabsTranscriber` — uploads one completed recording to Scribe and
    returns the transcript for the agent turn

Streaming STT needs a turn or segment policy before it can produce a final
transcript. That policy belongs to a voice transport or a realtime
speech-to-speech integration, not this one-turn STT pipeline.

It speaks first, so if you only want to hear the TTS side, listen to part 1
and press Ctrl+C — nothing touches the microphone until part 2.

Compare `elevenlabs_stt_tts_sync.py`, which uses buffered synthesis. Run
them back to back: part 1 here starts talking noticeably sooner.

Needs ELEVENLABS_API_KEY, and OPENAI_API_KEY for the agent in part 2.

    python examples/stt_tts.py/elevenlabs_stt_tts_streaming.py
"""

import asyncio
import time

from ag2 import Agent, config
from ag2.events import TranscriptionCompletedEvent
from ag2.live import (
    ElevenLabsStreamingTTSConfig,
    ElevenLabsTranscriber,
    SoundDevicePlayer,
    SoundDeviceRecorder,
    TTSObserver,
)

VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel
SAMPLE_RATE = 24000  # pcm_24000, what SoundDevicePlayer and the recorder use

INTRO = (
    "This is the streaming mode. Audio chunks arrive as they are generated, so playback "
    "can begin long before this sentence is finished."
)


async def main() -> None:
    tts = ElevenLabsStreamingTTSConfig("eleven_flash_v2_5", voice_id=VOICE_ID)
    stt = ElevenLabsTranscriber("scribe_v2")

    print("[1/2] speaking — streaming TTS")
    start = time.perf_counter()
    total_bytes = 0

    async with SoundDevicePlayer() as player:
        async for chunk in tts.stream(INTRO):
            if not total_bytes:
                ttfb = time.perf_counter() - start
                print(f"      first audio after {ttfb:.2f}s — playback starts now")
            total_bytes += len(chunk)
            # Straight to the speaker; this is what the mode is for.
            await player.play(chunk)

        print(f"      generated {total_bytes / (SAMPLE_RATE * 2):.1f}s of audio in {time.perf_counter() - start:.2f}s")

    print("\n      TTS done — Ctrl+C here if that is all you wanted to test.\n")

    print("[2/2] say something (recording 5s)...")
    voice = SoundDeviceRecorder(sample_rate=SAMPLE_RATE).record(duration=5)

    async def on_transcript(event: TranscriptionCompletedEvent) -> None:
        print(f'      Captured: "{event.content}"')

    agent = Agent(
        "assistant",
        prompt="You are a helpful voice assistant. Keep answers to a few sentences.",
        config=config.OpenAIConfig("gpt-5-mini", streaming=True),
        observers=[TTSObserver(tts)],
    )

    async with SoundDevicePlayer() as player:
        player.stream.where(TranscriptionCompletedEvent).subscribe(on_transcript)
        start = time.perf_counter()
        reply = await stt.pipe(agent).ask(voice, stream=player.stream)
        print(f"      transcribed + answered in {time.perf_counter() - start:.2f}s")
        print(f"      reply: {reply.body}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
