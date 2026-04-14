"""01 · Hello Actor

The smallest possible example: a bare ``Actor`` with a single LLM config and
one ``ask()`` call. No tools, no harness primitives, no plugins.

Run::

    .venv-beta/bin/python playground/01_hello_actor.py
"""

from __future__ import annotations

import asyncio

from autogen.beta import Actor

from _config import default_config, section


async def main() -> None:
    config = default_config()

    section("Bare Actor — ask and print")

    agent = Actor(
        "greeter",
        prompt="You are a friendly but concise assistant. Reply in one sentence.",
        config=config,
    )

    reply = await agent.ask("Give me a single tip for learning to play chess.")
    print(reply.body)

    section("Reuse the Actor for another ask")

    reply2 = await agent.ask("And a tip for learning poker, in one sentence.")
    print(reply2.body)


if __name__ == "__main__":
    asyncio.run(main())
