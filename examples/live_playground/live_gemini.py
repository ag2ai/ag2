import asyncio

from ag2.events import ModelMessageChunk, TranscriptionChunkEvent
from ag2.live import (
    LiveAgent,
    SoundDevicePlayer,
    SoundDeviceRecorder,
    gemini,
)
from ag2.tools import ToolResult, tool


@tool
async def sum_numbers(a: int, b: int) -> int:
    """You can use this tool to sum two numbers."""
    print(f"Summing {a} and {b}")
    return ToolResult(
        {"type": "text", "content": str(a + b)},
        final=True,
    )


agent = LiveAgent(
    name="assistant",
    prompt="You are a helpful voice assistant. Always respond in English.",
    tools=[sum_numbers],
    config=gemini.RealTimeConfig(
        "gemini-3.1-flash-live-preview",
        output=gemini.AudioOutput(voice="Puck", language_code="en-US"),
        input=gemini.InputConfig(transcribe=True),
    ),
)


async def main() -> None:
    async with (
        agent.run() as context,
        SoundDevicePlayer(context=context),
        # Gemini Live requires 16 kHz mono PCM input (24 kHz output).
        SoundDeviceRecorder(context=context, sample_rate=16000),
    ):
        print("Starting...")
        with context.stream.where(ModelMessageChunk | TranscriptionChunkEvent).join() as events:
            async for event in events:
                print(event)


if __name__ == "__main__":
    asyncio.run(main())
