import asyncio

from autogen.beta import Agent, Context, MemoryStream, config
from autogen.beta.live import OpenAITTSConfig, SoundDevicePlayer, TTSObserver

agent = Agent(
    name="assistant",
    prompt="You are a helpful voice assistant.",
    config=config.OpenAIResponsesConfig(model="gpt-5", streaming=True),
    observers=[
        TTSObserver(config=OpenAITTSConfig(model="gpt-4o-mini-tts")),
    ],
)


async def main() -> None:
    context = Context(stream=MemoryStream())

    async with SoundDevicePlayer(context=context):
        # pass the same with Player's context stream to play the audio
        await agent.ask("Hello, agent!", stream=context.stream)


if __name__ == "__main__":
    asyncio.run(main())
