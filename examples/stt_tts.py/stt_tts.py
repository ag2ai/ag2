import asyncio

from ag2 import Agent, config
from ag2.live import OpenAITTSConfig, OpenAITranscriber, SoundDevicePlayer, SoundDeviceRecorder, TTSObserver

agent = Agent(
    name="assistant",
    prompt="You are a helpful voice assistant.",
    config=config.OpenAIResponsesConfig(model="gpt-5", streaming=True),
    observers=[
        TTSObserver(config=OpenAITTSConfig(model="tts-1")),
    ],
)


async def main():
    pipeline = OpenAITranscriber("gpt-4o-mini-transcribe").pipe(agent)

    recorder = SoundDeviceRecorder()

    async with SoundDevicePlayer() as player:
        print("Say something...")
        voice_input = recorder.record(duration=3)
        reply = await pipeline.ask(voice_input, stream=player.stream)
        print(reply.body)

        # wait for the audio to finish playing
        player.join()

        print("Say something...")
        voice_input = recorder.record(duration=3)
        reply = await reply.ask(voice_input)
        print(reply.body)


if __name__ == "__main__":
    asyncio.run(main())
