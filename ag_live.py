import asyncio

from autogen.beta.events import ModelResponse
from autogen.beta.voice import (
    LiveTranscription,
    OpenAIRealTimeConfig,
    SoundDevicePlayer,
    SoundDeviceRecorder,
)


async def main() -> None:
    async with (
        LiveTranscription(
            OpenAIRealTimeConfig(
                "gpt-4o-realtime-preview",
                session={
                    "modalities": ["audio", "text"],
                    "voice": "ash",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "speed": 1.2,
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": {
                        "type": "semantic_vad",
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
            ),
        ) as context,
        SoundDevicePlayer(context=context),
        SoundDeviceRecorder(context=context),
    ):
        print("Starting...")
        with context.stream.where(ModelResponse).join() as events:
            async for event in events:
                # print(event)
                pass


if __name__ == "__main__":
    asyncio.run(main())
