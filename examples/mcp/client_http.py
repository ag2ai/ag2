"""Call a running streamable-HTTP MCP server (see ``server_http.py``).

Start the server first:

    uvicorn examples.mcp.server_http:app --host 127.0.0.1 --port 8000

Then call it (optionally pass your own message):

    python -m examples.mcp.client_http
    python -m examples.mcp.client_http "Add 2 and 3 with calc_add; reply with just the number."

Override the endpoint with MCP_URL, e.g. MCP_URL=http://127.0.0.1:8000/mcp.
"""

import asyncio
import os
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

DEFAULT_URL = "http://127.0.0.1:8000/mcp"
DEFAULT_MESSAGE = "Add 17 and 25 with calc_add, then reply with just the number."


async def on_progress(progress: float, total: float | None, message: str | None) -> None:
    print(f"  …progress[{progress:g}]: {message!r}")


async def main() -> None:
    url = os.environ.get("MCP_URL", DEFAULT_URL)
    message = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MESSAGE

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("exposed tools:", [t.name for t in tools.tools])

            result = await session.call_tool("ask", {"message": message}, progress_callback=on_progress)

            print("isError:", result.isError)
            if result.structuredContent is not None:
                print("structuredContent:", result.structuredContent)
            for block in result.content:
                if block.type == "text":
                    print("reply:", block.text)


if __name__ == "__main__":
    asyncio.run(main())
