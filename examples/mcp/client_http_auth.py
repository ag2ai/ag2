"""Call the authenticated streamable-HTTP MCP server (see server_http_auth.py).

Discovers the protected-resource metadata (RFC 9728), then calls the MCP endpoint
with a bearer token.

Run (after starting server_http_auth):

    python -m examples.mcp.client_http_auth

Override with MCP_URL / MCP_TOKEN.
"""

import asyncio
import os
from urllib.parse import urlparse

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = os.environ.get("MCP_URL", "http://127.0.0.1:8000/mcp")
TOKEN = os.environ.get("MCP_TOKEN", "demo-secret-token")


def _well_known_url(mcp_url: str) -> str:
    parsed = urlparse(mcp_url)
    return f"{parsed.scheme}://{parsed.netloc}/.well-known/oauth-protected-resource{parsed.path}"


async def main() -> None:
    # 1) Discover where to authenticate.
    async with httpx.AsyncClient() as http:
        metadata = (await http.get(_well_known_url(URL))).json()
    print("authorization servers:", metadata["authorization_servers"])
    print("scopes supported:", metadata.get("scopes_supported"))

    # 2) Call the MCP endpoint with a bearer token.
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with (
        streamablehttp_client(URL, headers=headers) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
        print("tools:", [t.name for t in tools.tools])
        result = await session.call_tool("ask", {"message": "Add 2 and 3 with calc_add; reply with just the number."})
        print("reply:", [c.text for c in result.content if c.type == "text"])


if __name__ == "__main__":
    asyncio.run(main())
