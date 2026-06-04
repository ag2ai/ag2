"""Serve an AG2 agent as an MCP server over streamable HTTP (remote / production).

``app`` is a Starlette ASGI app, so you can run it with any ASGI server. Attach
CORS / auth middleware to ``app`` as needed.

Run with uvicorn:

    uvicorn examples.mcp.server_http:app --host 127.0.0.1 --port 8000

Then connect an MCP client to http://127.0.0.1:8000/mcp — e.g.:

    npx @modelcontextprotocol/inspector  # transport: "Streamable HTTP", URL above

Requires ANTHROPIC_API_KEY in the environment.
"""

import uvicorn

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

# Importable by uvicorn as ``examples.mcp.server_http:app``.
app = MCPServer(agent).build_streamable_http(path="/mcp")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
