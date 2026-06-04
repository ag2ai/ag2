"""Interactive multi-turn MCP client over stdio.

Spawns the stdio server (``server_stdio.py``) and opens a chat loop: type a
message, get a reply, repeat. The served agent treats each tool call as a fresh
conversation (the server is stateless), so this client keeps continuity by
threading the running transcript back through the tool's ``context`` argument.

Run:

    python -m examples.mcp.client_interactive

Commands:
    /reset   clear the conversation context
    /tools   list the server's tools
    /quit    exit (Ctrl-D / Ctrl-C also exit)

Requires ANTHROPIC_API_KEY in the environment.
"""

import asyncio
import os
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def ainput(prompt: str) -> str:
    """Read a line from the terminal without blocking the event loop."""
    return await asyncio.to_thread(input, prompt)


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "examples.mcp.server_stdio"],
        env=dict(os.environ),  # forward ANTHROPIC_API_KEY to the server subprocess
    )

    transcript: list[str] = []

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        print(f"connected — tools: {[t.name for t in tools.tools]}")
        print("chat with the agent (/reset, /tools, /quit):")

        while True:
            try:
                line = (await ainput("you> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line:
                continue
            if line in ("/quit", "/exit"):
                break
            if line == "/reset":
                transcript.clear()
                print("(context cleared)")
                continue
            if line == "/tools":
                listed = await session.list_tools()
                print([t.name for t in listed.tools])
                continue

            arguments: dict[str, str] = {"message": line}
            if transcript:
                arguments["context"] = "\n".join(transcript)

            result = await session.call_tool("ask", arguments)
            reply = "".join(c.text for c in result.content if c.type == "text")

            if result.isError:
                print(f"bot> (error) {reply}")
                continue

            print(f"bot> {reply}")
            transcript.append(f"User: {line}")
            transcript.append(f"Assistant: {reply}")


if __name__ == "__main__":
    asyncio.run(main())
