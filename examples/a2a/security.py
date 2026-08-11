from a2a.types import ClientCredentialsOAuthFlow, OAuthFlows

from ag2 import Agent
from ag2.a2a import build_card
from ag2.a2a.security import (
    api_key_scheme,
    bearer_scheme,
    oauth2_scheme,
    require,
)
from ag2.config import AnthropicConfig

agent = Agent(
    name="claude",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
)

bearer = bearer_scheme(name="bearer", description="JWT auth")
api = api_key_scheme(name="api_key", key_name="X-API-Key", location="header")
oauth = oauth2_scheme(
    name="oauth",
    flows=OAuthFlows(
        client_credentials=ClientCredentialsOAuthFlow(token_url="https://idp/token"),
    ),
)

card = build_card(
    agent,
    url="http://127.0.0.1:8000",
    # Each `require(...)` is one OR-alternative. Multiple schemes inside a
    # single require() are AND-ed. `security_schemes` on the card are
    # auto-derived from the schemes referenced here — no double-declaration.
    security=[
        require(bearer),
        require(oauth.with_scopes("read", "write")),
        require(bearer, api),
    ],
)
