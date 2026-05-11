import asyncio

from autogen.beta.events import ModelMessageChunk
from autogen.beta.live import (
    LiveAgent,
    SoundDeviceRecorder,
    openai,
)

agent = LiveAgent(
    name="assistant",
    prompt="You are a helpful voice assistant.",
    config=openai.RealTimeConfig(
        "gpt-realtime-2",
        # audio output is disabled, just text
        output=openai.TextOutput(),
    ),
)


async def main() -> None:
    async with (
        agent.run() as context,
        SoundDeviceRecorder(context=context),
    ):
        print("Starting...")
        with context.stream.where(ModelMessageChunk).join() as events:
            async for event in events:
                print(event)


if __name__ == "__main__":
    asyncio.run(main())
