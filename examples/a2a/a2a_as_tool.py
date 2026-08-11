import asyncio

from ag2 import Agent
from ag2.a2a import A2AConfig
from ag2.config import AnthropicConfig
from ag2.tools.subagents import persistent_stream

researcher = Agent(
    "researcher",
    config=A2AConfig(card_url="http://research.internal:8000"),
)

writer = Agent(
    "writer",
    prompt="Use the researcher tool to gather facts before writing a draft.",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
    tools=[
        researcher.as_tool(
            description="Delegate research questions to the remote researcher.",
            stream=persistent_stream(),
        ),
    ],
)


async def main() -> None:
    reply = await writer.ask("Write a 3-paragraph brief on the latest A2A spec changes.")
    reply = await reply.ask("Now expand section 2 with the citations the researcher found.")
    print(reply.response.content)


if __name__ == "__main__":
    asyncio.run(main())
