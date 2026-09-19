import asyncio

from ag2 import Agent
from ag2.config import AnthropicConfig
from ag2.tools.search import DuckDuckSearchTool
from ag2.tools.subagents import SearchMode, recursive_search_tool


async def main() -> None:
    # Wire the recursive search tool to a caller-supplied web search tool. The
    # swarm's nodes share this config and use DuckDuckGo to actually search.
    config = AnthropicConfig(model="claude-sonnet-4-6")
    agent = Agent(
        name="researcher",
        config=config,
        tools=[
            recursive_search_tool(
                config=config,
                search_mode=SearchMode.WIDE,
                tools=[DuckDuckSearchTool(max_results=5)],
                max_depth=3,
                max_children=3,
            )
        ],
    )

    reply = await agent.ask("Compare the leading vector databases for a RAG workload.")
    print(await reply.content())


if __name__ == "__main__":
    asyncio.run(main())
