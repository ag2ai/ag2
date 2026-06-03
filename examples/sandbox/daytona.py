import asyncio

from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.extensions.daytona import DaytonaEnvironment
from autogen.beta.tools import SandboxCodeTool, SandboxShellTool

env = DaytonaEnvironment(image="python:3.12")

agent = Agent(
    name="claude",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
    tools=[SandboxShellTool(env), SandboxCodeTool(env)],
)


async def main() -> None:
    async with env:  # deletes the cloud sandbox on exit
        reply = await agent.ask(
            "Write 'hello world' into notes.txt, then run python to print 2 + 2. "
            "Tell me the file contents and the result."
        )
        print(await reply.content())


if __name__ == "__main__":
    asyncio.run(main())
