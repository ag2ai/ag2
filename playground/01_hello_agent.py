"""01 · Hello Agent

The smallest possible example: a bare ``Agent`` with a single LLM config and
one ``ask()`` call. No tools, no harness primitives, no plugins.

Run::

    .venv-beta/bin/python playground/01_hello_actor.py
"""

import asyncio

from _config import default_config, section

from autogen.beta import Agent


async def main() -> None:
    config = default_config()

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
