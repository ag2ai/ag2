import asyncio
import os
import sys

from autogen import Agent
from autogen.config import AnthropicConfig
from autogen.tools import MCPStdioServerConfig, MCPToolkit


async def main() -> None:
    # Launch the stdio server as a subprocess and expose its tools to the agent.
    agent = Agent(
        name="client",
        config=AnthropicConfig(model="claude-sonnet-4-6"),
        tools=[
            MCPToolkit(
                MCPStdioServerConfig(
                    command=sys.executable,
                    args=["-m", "examples.mcp.server_stdio"],
                    env=dict(os.environ),  # forward ANTHROPIC_API_KEY to the subprocess
                )
            )
        ],
    )

    reply = await agent.ask("Use the ask tool to add 17 and 25. Reply with just the number.")
    print(await reply.content())


if __name__ == "__main__":
    asyncio.run(main())
