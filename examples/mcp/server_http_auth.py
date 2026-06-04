"""Serve an AG2 agent as an MCP server over streamable HTTP with OAuth (Resource Server).

Demonstrates RFC 9728 protected-resource discovery + bearer-token enforcement.
The server advertises its authorization server(s) at
``/.well-known/oauth-protected-resource/mcp`` and requires a valid bearer token
on ``/mcp``.

This example uses a trivial static-token verifier so it runs without an external
IdP. In production, replace ``StaticTokenVerifier`` with one that validates tokens
against your authorization server (JWKS signature check or token introspection).

Run:

    uvicorn examples.mcp.server_http_auth:app --host 127.0.0.1 --port 8000

Discover:

    curl http://127.0.0.1:8000/.well-known/oauth-protected-resource/mcp

Then call /mcp with header:  Authorization: Bearer demo-secret-token

Requires ANTHROPIC_API_KEY in the environment.
"""

from collections.abc import Sequence

import uvicorn

from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.mcp import MCPServer
from autogen.beta.mcp.security import AccessToken, oauth2_scheme, require
from autogen.beta.tools import tool


@tool(description="Add two integers.")
async def calc_add(a: int, b: int) -> str:
    return str(a + b)


class StaticTokenVerifier:
    """Demo TokenVerifier — accepts a single hard-coded token. NOT for production."""

    def __init__(self, token: str, *, client_id: str = "demo-client", scopes: Sequence[str] = ("mcp.read",)) -> None:
        self._token = token
        self._client_id = client_id
        self._scopes = list(scopes)

    async def verify_token(self, token: str) -> AccessToken | None:
        if token != self._token:
            return None
        return AccessToken(token=token, client_id=self._client_id, scopes=self._scopes)


agent = Agent(
    name="claude",
    prompt="You are a concise assistant. Use tools when they help.",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
    tools=[calc_add],
)

security = require(
    oauth2_scheme(url="https://auth.example.com"),
    resource_url="http://127.0.0.1:8000/mcp",
    verifier=StaticTokenVerifier("demo-secret-token"),
    required_scopes=["mcp.read"],
    resource_name="AG2 demo agent",
)

app = MCPServer(agent).build_streamable_http(path="/mcp", security=security)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
