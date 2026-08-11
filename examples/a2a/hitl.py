import asyncio

from ag2 import Agent
from ag2.a2a import A2AConfig


async def hitl_hook() -> str:
    return await asyncio.to_thread(input, "server asks input> ")


async def main() -> None:
    remote = Agent(
        "remote",
        config=A2AConfig(card_url="http://127.0.0.1:8000", input_required_timeout=30.0),
        hitl_hook=hitl_hook,
    )
    reply = await remote.ask("start")
    print(reply.response.content)


if __name__ == "__main__":
    asyncio.run(main())
