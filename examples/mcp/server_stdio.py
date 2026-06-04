"""Serve an AG2 agent as an MCP server over stdio (local clients).

Point Claude Desktop / Cursor / the MCP Inspector at this module to call the
agent through the standard MCP protocol.

Run directly:

    python -m examples.mcp.server_stdio

Or via the MCP Inspector:

    npx @modelcontextprotocol/inspector python -m examples.mcp.server_stdio

Requires ANTHROPIC_API_KEY in the environment.
"""

import asyncio

from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.mcp import MCPServer
from autogen.beta.tools import tool


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
