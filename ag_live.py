import asyncio

from autogen.beta.agent import Agent
from autogen.beta.events import ModelMessageChunk, TranscriptionChunkEvent
from autogen.beta.voice import (
    AudioConfig,
    LiveAgent,
    OpenAIRealTimeConfig,
    SoundDevicePlayer,
    SoundDeviceRecorder,
)


async def main() -> None:
    agent = Agent(
        name="assistant",
        prompt="You are a helpful voice assistant.",
    )

    async with (
        LiveAgent(
            name="assistant",
            prompt="You are a helpful voice assistant.",
            config=OpenAIRealTimeConfig(
                "gpt-4o-realtime-preview",
                audio=AudioConfig(voice="ash", speed=1.2),
            ),
        ) as context,
        SoundDevicePlayer(context=context),
        SoundDeviceRecorder(context=context),
    ):
        print("Starting...")
        with context.stream.where(ModelMessageChunk | TranscriptionChunkEvent).join() as events:
            async for event in events:
                print(event)


if __name__ == "__main__":
    asyncio.run(main())
