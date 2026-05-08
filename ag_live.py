import asyncio

from autogen.beta.events import ModelMessageChunk, TranscriptionChunkEvent
from autogen.beta.tools import ToolResult, tool
from autogen.beta.voice import (
    LiveAgent,
    OpenAIRealTimeConfig,
    SoundDevicePlayer,
    SoundDeviceRecorder,
)


@tool
async def sum_numbers(a: int, b: int) -> int:
    """You can use this tool to sum two numbers."""
    print(f"Summing {a} and {b}")
    return ToolResult(
        {"type": "text", "content": str(a + b)},
        final=True,
    )


async def main() -> None:
    agent = LiveAgent(
        name="assistant",
        prompt="You are a helpful voice assistant.",
        tools=[sum_numbers],
        config=OpenAIRealTimeConfig("gpt-realtime-2"),
    )

    async with (
        agent.run() as context,
        SoundDevicePlayer(context=context),
        SoundDeviceRecorder(context=context),
    ):
        print("Starting...")
        with context.stream.where(ModelMessageChunk | TranscriptionChunkEvent).join() as events:
            async for event in events:
                print(event)


if __name__ == "__main__":
    asyncio.run(main())
