import asyncio

from ag2 import Agent, config
from ag2.live import OpenAITranscriber, SoundDeviceRecorder

agent = Agent(
    "test",
    config=config.OpenAIConfig("gpt-5", streaming=True),
)


async def main():
    # pipe STT model to agent input
    pipeline = OpenAITranscriber("gpt-4o-mini-transcribe").pipe(agent)

    recorder = SoundDeviceRecorder()

    print("Say something...")
    voice_input = recorder.record(duration=5)
    # fire the pipeline and get the reply
    reply = await pipeline.ask(voice_input)
    print(reply.body)

    print("Say something...")
    voice_input = recorder.record(duration=5)
    # continue the conversation
    reply = await reply.ask(voice_input)
    print(reply.body)


if __name__ == "__main__":
    asyncio.run(main())
