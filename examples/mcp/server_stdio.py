import asyncio

from ag2 import Agent
from ag2.config import AnthropicConfig
from ag2.mcp import MCPServer
from ag2.tools import tool


@tool(description="Add two integers.")
async def calc_add(a: int, b: int) -> str:
    return str(a + b)


agent = Agent(
    name="claude",
    prompt="You are a concise assistant. Use tools when they help.",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
    tools=[calc_add],
)


async def main() -> None:
    await MCPServer(agent).run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
