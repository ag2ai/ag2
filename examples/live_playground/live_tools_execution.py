import asyncio

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


@agent.tool
async def sum_numbers(a: int, b: int) -> int:
    """You can use this tool to sum two numbers."""
    print(f"Summing {a} and {b}")
    return a + b


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
