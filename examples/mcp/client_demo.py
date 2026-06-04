"""Self-contained demo: launch the stdio MCP server and call its agent.

Spawns ``examples.mcp.server_stdio`` as a subprocess, connects over stdio,
lists the exposed tool, and calls it — streaming progress updates as the agent
produces its reply. No second terminal and no external MCP client needed.

Run:

    python -m examples.mcp.client_demo

Requires ANTHROPIC_API_KEY in the environment.
"""

import asyncio
import os
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def on_progress(progress: float, total: float | None, message: str | None) -> None:
    print(f"  …progress[{progress:g}]: {message!r}")


async def main() -> None:
    # stdio servers run in a minimal environment by default, so forward ours
    # explicitly — otherwise the spawned server can't see ANTHROPIC_API_KEY.
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "examples.mcp.server_stdio"],
        env=dict(os.environ),
    )

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        tools = await session.list_tools()
        print("exposed tools:", [t.name for t in tools.tools])

        result = await session.call_tool(
            "ask",
            {"message": "Add 17 and 25 with calc_add, then reply with just the number."},
            progress_callback=on_progress,
        )

        print("isError:", result.isError)
        for block in result.content:
            if block.type == "text":
                print("reply:", block.text)


if __name__ == "__main__":
    asyncio.run(main())
