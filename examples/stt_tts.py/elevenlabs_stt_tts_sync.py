"""ElevenLabs voice round-trip, synchronous mode. No arguments.

Both halves wait for the complete result before doing anything:

  * `ElevenLabsTTSConfig` — one buffered response, required by ``eleven_v3``
  * `ElevenLabsTranscriber` — Scribe's upload endpoint, whole clip at once

It speaks first, so if you only want to hear the TTS side, listen to part 1
and press Ctrl+C — nothing touches the microphone until part 2.

Compare `elevenlabs_stt_tts_streaming.py`, which uses streaming TTS.

Needs ELEVENLABS_API_KEY, and OPENAI_API_KEY for the agent in part 2.

    python examples/stt_tts.py/elevenlabs_stt_tts_sync.py
"""

import asyncio
import time

from ag2 import Agent, config
from ag2.events import TranscriptionCompletedEvent
from ag2.live import (
    ElevenLabsTTSConfig,
    ElevenLabsTranscriber,
    SoundDevicePlayer,
    SoundDeviceRecorder,
    TTSObserver,
)

VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel
SAMPLE_RATE = 24000  # pcm_24000, what SoundDevicePlayer and the recorder use

INTRO = (
    "This is the synchronous[gasp] mode. Nothing comes back until the entire clip has been "
    "generated, so the first byte and the last byte arrive at the same moment."
)


async def main() -> None:
    tts = ElevenLabsTTSConfig("eleven_v3", voice_id=VOICE_ID)
    stt = ElevenLabsTranscriber("scribe_v2")

    print("[1/2] speaking — synchronous TTS")
    start = time.perf_counter()
    pcm = await tts.synthesize(INTRO)
    elapsed = time.perf_counter() - start
    # Nothing was playable until now: no chunks, so no early start.
    print(f"      generated {len(pcm) / (SAMPLE_RATE * 2):.1f}s of audio in {elapsed:.2f}s")

    # A player per phase. Leaving the context drains the queue and joins the
    # worker thread; `player.join()` does NOT wait for playback — it returns as
    # soon as the queue is empty, which is before the audio has been written.
    # Holding one player open across both phases would let the microphone in
    # phase 2 open mid-sentence and record the speaker.
    async with SoundDevicePlayer() as player:
        await player.play(pcm)

    print("\n      TTS done — Ctrl+C here if that is all you wanted to test.\n")

    print("[2/2] say something (recording 5s)...")
    voice = SoundDeviceRecorder(sample_rate=SAMPLE_RATE).record(duration=5)

    # Scribe returns the whole transcript at once, so there is a single
    # completed event and no partials to show.
    async def on_transcript(event: TranscriptionCompletedEvent) -> None:
        print(f'      Captured: "{event.content}"')

    agent = Agent(
        "assistant",
        prompt="You are a helpful voice assistant. Keep answers to a few sentences.",
        config=config.OpenAIConfig("gpt-5-mini", streaming=True),
        observers=[TTSObserver(config=tts)],
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
