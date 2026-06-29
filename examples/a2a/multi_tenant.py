import asyncio

from ag2 import Agent
from ag2.a2a import A2AConfig, build_card
from ag2.config import AnthropicConfig

agent = Agent(
    name="claude",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
)

# Server side: each transport binds to its own tenant scope.
card = build_card(
    agent,
    url="http://127.0.0.1:8000",
    transports=("jsonrpc", "grpc"),
    grpc_url="127.0.0.1:50051",
    tenants={"jsonrpc": "tenant-A", "grpc": "tenant-B"},
)

# Client side: scope every request to a tenant.
config = A2AConfig(card_url="http://127.0.0.1:8000", tenant="tenant-A")
remote = Agent("remote", config=config)


async def main() -> None:
    # Default tenant (tenant-A) for this client.
    reply = await remote.ask("ping")
    print(reply.response.content)

    # One-off override on a single ask:
    reply = await remote.ask("ping", variables={"a2a:tenant": "tenant-Z"})
    print(reply.response.content)


if __name__ == "__main__":
    asyncio.run(main())
