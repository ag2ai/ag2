import asyncio

from ag2 import Agent, config
from ag2.live import OpenAITTSConfig, SoundDevicePlayer, TTSObserver

agent = Agent(
    name="assistant",
    prompt="You are a helpful voice assistant.",
    config=config.OpenAIResponsesConfig(model="gpt-5", streaming=True),
    observers=[
        TTSObserver(config=OpenAITTSConfig(model="gpt-4o-mini-tts")),
    ],
)


async def main() -> None:
    async with SoundDevicePlayer() as player:
        # pass the same with Player's context stream to play the audio
        await agent.ask("Hello, agent!", stream=player.stream)


if __name__ == "__main__":
    asyncio.run(main())
