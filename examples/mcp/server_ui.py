import asyncio
import json
from typing import Any

from ag2 import Agent
from ag2.config import AnthropicConfig
from ag2.mcp import MCPServer, mcp_tool
from ag2.mcp_ui import actions, external_url, raw_html


@mcp_tool
async def show_greeting(name: str = "world") -> Any:
    """Return an interactive greeting card with a UI Action button."""
    # A `tool` UI Action: on click the host calls tools/call add_to_cart(...).
    action = actions.tool_call("add_to_cart", {"good_id": "42"})
    payload = json.dumps(action.model_dump(exclude_none=True))
    html = (
        f"<div style='font-family:sans-serif;padding:24px'>"
        f"<h1>Hello, {name} 👋</h1>"
        f"<p>Rendered by an MCP-UI client from an AG2 server.</p>"
        f"<button onclick='window.parent.postMessage({payload}, \"*\")'>"
        f"Add to cart</button></div>"
    )
    return raw_html("ui://ag2/greeting", html)


@mcp_tool
async def show_docs() -> Any:
    """Return the AG2 docs embedded in an iframe."""
    return external_url("ui://ag2/docs", "https://docs.ag2.ai/")


@mcp_tool
async def add_to_cart(good_id: str) -> Any:
    """Target of the greeting card's UI Action; returns an updated card."""
    return raw_html("ui://ag2/cart", f"<b>Added item {good_id} to cart ✓</b>")


agent = Agent(
    name="claude",
    prompt="You are a concise assistant.",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
)

server = MCPServer(agent, tools=[show_greeting, show_docs, add_to_cart])


if __name__ == "__main__":
    asyncio.run(server.run_stdio())
