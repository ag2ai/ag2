import asyncio

from ag2 import Agent
from ag2.a2a import A2AConfig


async def main() -> None:
    remote = Agent(
        "remote",
        config=A2AConfig(card_url="http://127.0.0.1:8000", prefer="jsonrpc"),
    )
    reply = await remote.ask("Add 17 and 25 with calc_add. Just the number.")
    print(reply.response.content)


if __name__ == "__main__":
    asyncio.run(main())
