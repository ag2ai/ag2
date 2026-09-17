"""The served half of the AG2-to-AG2 round trip.

A tool inside this agent asks a question, and with ``elicitation_policy="ask"``
the question travels to the human behind whichever MCP client called it. Run
``examples/mcp/client_elicitation.py`` — it launches this module as a subprocess.
"""

import asyncio

from ag2 import Agent, Context
from ag2.config import AnthropicConfig
from ag2.mcp import MCPServer
from ag2.tools import tool


@tool(description="Book a table, confirming the party size with the user.")
async def book_table(when: str, context: Context) -> str:
    size = await context.input(f"How many people for {when}?")
    return f"Booked {when} for {size}."


agent = Agent(
    name="concierge",
    prompt="You are a restaurant concierge. Use book_table to make a booking.",
    config=AnthropicConfig(model="claude-sonnet-5"),
    tools=[book_table],
)


async def main() -> None:
    await MCPServer(agent, elicitation_policy="ask").run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
