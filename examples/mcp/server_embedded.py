"""Embed the MCP server inside an existing FastAPI/Starlette app.

``MCPServer.mount_into(app, path=...)`` adds the agent's MCP routes to your app and
wires the streamable-HTTP session-manager lifespan automatically — no manual lifespan
plumbing (without it, MCP requests fail with "Task group is not initialized"). Your own
routes are untouched, and they share one process / port with the agent.

Run:

    uvicorn examples.mcp.server_embedded:app --host 127.0.0.1 --port 8000

    GET  /        -> your API
    POST /mcp     -> the AG2 agent over MCP

Requires ANTHROPIC_API_KEY in the environment.
"""

from fastapi import FastAPI

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

app = FastAPI()


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "my-api", "mcp": "/mcp"}


# Mount the agent at /mcp and compose its lifespan into this app. To require auth,
# pass security=require(...) (see server_http_auth.py) — it stays scoped to /mcp.
MCPServer(agent).mount_into(app, path="/mcp")
