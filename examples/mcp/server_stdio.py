import asyncio

from autogen import Agent
from autogen.config import AnthropicConfig
from autogen.mcp import MCPServer
from autogen.tools import tool


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
