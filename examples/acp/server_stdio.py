"""Serve an AG2 agent to an ACP Client over stdio.

Nothing here is started by you: the Client launches this file as a subprocess and
speaks ACP over the pipe between you. Point an ACP Client at it with a launch
entry of the shape every Client uses:

.. code-block:: json

    {
      "command": "/abs/path/to/.venv/bin/python",
      "args": ["/abs/path/to/examples/acp/server_stdio.py"],
      "env": {"ANTHROPIC_API_KEY": "<your key>"}
    }

Both paths are absolute on purpose. Your virtualenv is not activated for that
process, so a bare ``python`` resolves to an interpreter that does not have
``ag2`` installed.

**stdout is the protocol wire.** A stray ``print()`` anywhere in this process —
yours or a library's — corrupts the JSON-RPC framing and drops the connection.
Write to stderr instead.
"""

import asyncio

from ag2 import Agent
from ag2.acp import ACPAgent
from ag2.config import AnthropicConfig
from ag2.tools import tool


@tool(description="Add two integers.")
async def calc_add(a: int, b: int) -> str:
    return str(a + b)


agent = Agent(
    name="workie",
    prompt="You are a concise assistant. Use tools when they help.",
    config=AnthropicConfig(model="claude-sonnet-5", streaming=True),
    tools=[calc_add],
)


async def main() -> None:
    await ACPAgent(agent).run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
