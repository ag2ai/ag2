import asyncio

import uvicorn

from autogen.beta import Agent
from autogen.beta.a2a import A2AServer, build_card
from autogen.beta.config import AnthropicConfig
from autogen.beta.tools import tool


@tool(description="Add two integers.")
async def calc_add(a: int, b: int) -> str:
    return str(a + b)


agent = Agent(
    name="claude",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
    tools=[calc_add],
)


async def main() -> None:
    server = A2AServer(agent)
    card = build_card(agent)
    asgi = server.build_jsonrpc(url="http://127.0.0.1:8000", card=card)

    await uvicorn.Server(uvicorn.Config(asgi, host="127.0.0.1", port=8000)).serve()


if __name__ == "__main__":
    asyncio.run(main())
