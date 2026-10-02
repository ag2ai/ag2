"""The calling half of the AG2-to-AG2 round trip.

The served agent's tool asks a question; ``elicitation="ask"`` lets it reach this
side, and ``hitl_hook`` decides what this side's human says back. Both halves need
``ANTHROPIC_API_KEY``.
"""

import asyncio
import os
import sys

from ag2 import Agent
from ag2.config import AnthropicConfig
from ag2.events import HumanInputRequest, HumanMessage
from ag2.tools import MCPAnswerPolicy, MCPStdioServerConfig, MCPToolkit


def hitl_hook(event: HumanInputRequest) -> HumanMessage:
    """Answer the served agent's question. A real deployment would block on a UI."""
    print(f"[the served agent asks] {event.content}")
    return HumanMessage(content="four")


async def main() -> None:
    agent = Agent(
        name="diner",
        config=AnthropicConfig(model="claude-sonnet-5"),
        hitl_hook=hitl_hook,
        tools=[
            MCPToolkit(
                MCPStdioServerConfig(
                    command=sys.executable,
                    args=["-m", "examples.mcp.server_elicitation_stdio"],
                    env=dict(os.environ),  # forward ANTHROPIC_API_KEY to the subprocess
                ),
                answering=MCPAnswerPolicy(elicitation="ask"),
            )
        ],
    )

    reply = await agent.ask("Ask the concierge to book a table for Friday at 8pm.")
    print(await reply.content())


if __name__ == "__main__":
    asyncio.run(main())
