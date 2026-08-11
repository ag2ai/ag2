import asyncio

from ag2.live import (
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
