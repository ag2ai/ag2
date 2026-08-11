import asyncio
from datetime import datetime

from ag2 import Agent
from ag2.a2a import A2AConfig
from ag2.tools import tool


@tool(description="Return the user's local wall-clock time.")
def get_local_time() -> str:
    return datetime.now().isoformat(timespec="seconds")


async def main() -> None:
    remote = Agent(
        "remote",
        config=A2AConfig(card_url="http://127.0.0.1:8000"),
        tools=[get_local_time],
    )
    reply = await remote.ask("What time is it on my machine? Use get_local_time.")
    print(reply.response.content)


if __name__ == "__main__":
    asyncio.run(main())
