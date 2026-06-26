import asyncio

from ag2 import Agent
from ag2.config import AnthropicConfig
from ag2.extensions.docker import DockerEnvironment
from ag2.tools import SandboxCodeTool, SandboxShellTool

env = DockerEnvironment(image="python:3.12-slim")

agent = Agent(
    name="claude",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
    tools=[SandboxShellTool(env), SandboxCodeTool(env)],
)


async def main() -> None:
    async with env:  # tears the container down on exit
        reply = await agent.ask(
            "Write 'hello world' into notes.txt, then run python to print 2 + 2. "
            "Tell me the file contents and the result."
        )
        print(await reply.content())


if __name__ == "__main__":
    asyncio.run(main())
