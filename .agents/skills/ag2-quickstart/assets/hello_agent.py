"""Hello Agent — minimal AG2 beta example.

Mirrors website/docs/beta/code_examples/01_hello_agent.mdx. The smallest
possible end-to-end: instantiate an Agent with one model config, call ask(),
print the reply, then reuse the same Agent for a second turn.

Run::

    python hello_agent.py
"""

import asyncio

from autogen.beta import Agent
from autogen.beta.config import GeminiConfig


def section(title: str) -> None:
    print(f"\n── {title} ───")


async def main() -> None:
    config = GeminiConfig(model="gemini-3-flash-preview", temperature=0)

    section("Bare Agent — ask and print")

    agent = Agent(
        "greeter",
        prompt="You are a friendly but concise assistant. Reply in one sentence.",
        config=config,
    )

    reply = await agent.ask("Give me a single tip for learning to play chess.")
    print(reply.body)

    section("Reuse the Agent for another ask")

    reply2 = await agent.ask("And a tip for learning poker, in one sentence.")
    print(reply2.body)


if __name__ == "__main__":
    asyncio.run(main())
