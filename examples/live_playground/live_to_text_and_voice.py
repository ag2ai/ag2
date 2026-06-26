import asyncio

from ag2.events import ModelMessageChunk
from ag2.live import (
    LiveAgent,
    OpenAIRealTimeConfig,
    SoundDevicePlayer,
    SoundDeviceRecorder,
)

agent = LiveAgent(
    name="assistant",
    prompt="You are a helpful voice assistant.",
    config=OpenAIRealTimeConfig("gpt-realtime-2"),
)


async def main() -> None:
    async with (
        agent.run() as context,
        SoundDevicePlayer(context=context),
        SoundDeviceRecorder(context=context),
    ):
        print("Starting...")
        # Streaming output audio transcript as ModelMessageChunk
        with context.stream.where(ModelMessageChunk).join() as events:
            async for event in events:
                print(event)


if __name__ == "__main__":
    asyncio.run(main())
