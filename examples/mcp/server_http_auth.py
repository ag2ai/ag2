from collections.abc import Sequence

import uvicorn

from ag2 import Agent
from ag2.config import AnthropicConfig
from ag2.mcp import MCPServer
from ag2.mcp.security import AccessToken, oauth2_scheme, require
from ag2.tools import tool


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

app = MCPServer(agent, path="/mcp", security=security)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
