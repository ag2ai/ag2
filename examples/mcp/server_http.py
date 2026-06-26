import uvicorn

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

app = MCPServer(agent, path="/mcp")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
