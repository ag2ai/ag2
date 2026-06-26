import asyncio

from ag2 import Agent
from ag2.config import AnthropicConfig
from ag2.tools import MCPServerConfig, MCPToolkit


async def main() -> None:
    agent = Agent(
        name="client",
        config=AnthropicConfig(model="claude-sonnet-4-6"),
        tools=[
            MCPToolkit(
                MCPServerConfig(
                    server_url="http://127.0.0.1:8000/mcp",
                    authorization_token="demo-secret-token",
                )
            )
        ],
    )

    reply = await agent.ask("Use the ask tool to add 2 and 3. Reply with just the number.")
    print(await reply.content())


if __name__ == "__main__":
    asyncio.run(main())
