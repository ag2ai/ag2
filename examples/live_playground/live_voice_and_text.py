import asyncio
import sys
import threading

from ag2 import Context
from ag2.events import ModelResponse
from ag2.live import (
    LiveAgent,
    OpenAIRealTimeConfig,
    SoundDevicePlayer,
    SoundDeviceRecorder,
)

agent = LiveAgent(
    name="assistant",
    prompt="You are a helpful voice assistant. The user may talk to you or type messages.",
    config=OpenAIRealTimeConfig("gpt-realtime-2"),
)


def read_typed_lines(loop: asyncio.AbstractEventLoop, context: Context) -> None:
    # Runs in a daemon thread, so a pending `readline` never blocks shutdown.
    # `enqueue` runs on the event loop that owns the session's stream, so the
    # session is told about the line: the model answers it right away when idle,
    # or at the end of the answer it is currently speaking.
    for line in sys.stdin:
        if text := line.strip():
            loop.call_soon_threadsafe(context.enqueue, text)


async def print_reply(event: ModelResponse) -> None:
    if event.content:
        print(f"  bot: {event.content}")


async def main() -> None:
    async with (
        agent.run() as context,
        SoundDevicePlayer(context=context),
        SoundDeviceRecorder(context=context),
    ):
        context.stream.where(ModelResponse).subscribe(print_reply)

        threading.Thread(
            target=read_typed_lines,
            args=(asyncio.get_running_loop(), context),
            daemon=True,
        ).start()

        print("Starting... talk, or type a line and press Enter. Ctrl+C to stop.")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
