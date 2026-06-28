import asyncio

from ag2 import Agent
from ag2.config import AnthropicConfig
from ag2.tools import MCPToolkit


async def main() -> None:
    # The remote MCP server's tools become ordinary local tools for the agent.
    agent = Agent(
        name="client",
        config=AnthropicConfig(model="claude-sonnet-4-6"),
        tools=[MCPToolkit("http://127.0.0.1:8000/mcp")],
    )

    reply = await agent.ask("Use the ask tool to add 17 and 25. Reply with just the number.")
    print(await reply.content())


if __name__ == "__main__":
    asyncio.run(main())
