"""Manual check for the ElevenLabs TTS backends — hits the real API.

Compares the two modes head to head. Both are ordinary HTTP — the difference is
whether the clip comes back in one piece or in chunks:

  * sync   — `ElevenLabsTTSConfig`, one buffered response (required by eleven_v3)
  * stream — `ElevenLabsStreamingTTSConfig`, chunks as they are generated

Needs ELEVENLABS_API_KEY. Speaks each mode through the default output device and
prints time-to-first-byte, which is the number the streaming path exists to
improve. In stream mode chunks go to the speaker as they land, so the latency
difference is audible and not just a number in the report.

    python examples/stt_tts.py/tts_elevenlabs.py                # both modes, out loud
    python examples/stt_tts.py/tts_elevenlabs.py --mode stream  # one mode only
    python examples/stt_tts.py/tts_elevenlabs.py --save         # also write wavs to cwd
    python examples/stt_tts.py/tts_elevenlabs.py --no-play      # measure only, no audio
"""

import argparse
import asyncio
import time
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from ag2.live import (
    ElevenLabsStreamingTTSConfig,
    ElevenLabsTTSConfig,
    SoundDevicePlayer,
)

# Each mode says which mode it is
TEXT = {
    "sync": (
        "This is the synchronous mode. Nothing comes back until the entire audio has been "
        "generated, so the first byte and the last byte arrive at the same moment."
    ),
    "stream": (
        "This is the streaming mode. Audio chunks arrive as they are generated, so playback "
        "can begin long before this sentence is finished."
    ),
}
SAMPLE_RATE = 24000  # matches pcm_24000, which is what SoundDevicePlayer expects


@asynccontextmanager
async def speaker(enabled: bool) -> AsyncIterator[SoundDevicePlayer | None]:
    """A player, or nothing at all under --no-play.

    One player per mode: leaving the context drains the queue and joins the
    worker thread, so the next mode cannot talk over the tail of this one.
    """
    if not enabled:
        yield None
        return
    async with SoundDevicePlayer() as player:
        yield player


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)  # int16
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm)


def report(mode: str, pcm: bytes, ttfb: float, total: float, chunks: int, save_dir: Path | None) -> None:
    duration = len(pcm) / (SAMPLE_RATE * 2)
    line = f"[{mode:6}] ttfb={ttfb:6.2f}s  total={total:6.2f}s  chunks={chunks:4}  audio={duration:5.2f}s"

    if save_dir is not None:
        path = save_dir / f"elevenlabs_{mode}.wav"
        write_wav(path, pcm)
        line += f"  -> {path}"

    print(line)


async def run_sync(text: str, model: str, voice: str, *, play: bool, save_dir: Path | None) -> None:
    tts = ElevenLabsTTSConfig(model, voice_id=voice)

    start = time.perf_counter()
    pcm = await tts.synthesize(text)
    total = time.perf_counter() - start

    async with speaker(play) as player:
        # Buffered mode surfaces no intermediate chunks — first byte *is* the last byte.
        report("sync", pcm, ttfb=total, total=total, chunks=1, save_dir=save_dir)
        if player is not None:
            await player.play(pcm)


async def run_stream(text: str, model: str, voice: str, *, play: bool, save_dir: Path | None) -> None:
    tts = ElevenLabsStreamingTTSConfig(model, voice_id=voice)

    start = time.perf_counter()
    ttfb = 0.0
    chunks: list[bytes] = []

    async with speaker(play) as player:
        async for chunk in tts.stream(text):
            if not chunks:
                ttfb = time.perf_counter() - start
            chunks.append(chunk)
            # Straight to the speaker — audio starts at ttfb, not after the
            # full clip is collected. This is what the mode is for.
            if player is not None:
                await player.play(chunk)

        # Generation time only; leaving the context then waits on playback.
        total = time.perf_counter() - start
        report("stream", b"".join(chunks), ttfb=ttfb, total=total, chunks=len(chunks), save_dir=save_dir)


def default_model(mode: str) -> str:
    return "eleven_v3" if mode == "sync" else "eleven_flash_v2_5"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sync", "stream", "both"), default="both")
    parser.add_argument("--text", default=None, help="overrides the per-mode default text")
    parser.add_argument("--voice", default="21m00Tcm4TlvDq8ikWAM")  # Rachel
    parser.add_argument("--model", default=None, help="defaults per mode: eleven_v3 / eleven_flash_v2_5")
    parser.add_argument("--no-play", dest="play", action="store_false", help="skip audio, just measure")
    parser.add_argument(
        "--save",
        nargs="?",
        type=Path,
        const=Path("."),
        default=None,
        metavar="DIR",
        help="also write a wav per mode (default: cwd)",
    )
    args = parser.parse_args()

    modes = ("sync", "stream") if args.mode == "both" else (args.mode,)

    for mode in modes:
        runner = run_sync if mode == "sync" else run_stream
        await runner(
            args.text or TEXT[mode],
            args.model or default_model(mode),
            args.voice,
            play=args.play,
            save_dir=args.save,
        )


if __name__ == "__main__":
    asyncio.run(main())
