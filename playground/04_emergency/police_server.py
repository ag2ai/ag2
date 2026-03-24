#!/usr/bin/env python3
"""Police server — Police agent on port 8902.

Usage:
    python playground/04_emergency/police_server.py
    python playground/04_emergency/police_server.py --model gemini-3-flash-preview
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _shared import BOLD, CYAN, PORTS, RESET, DIM, make_police, subscribe_hub_logging

from autogen.beta.network import Hub


async def main() -> None:
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "gemini-3-flash-preview"
    port = PORTS["police"]

    hub = Hub(max_delegation_depth=4)
    await hub.register(make_police(model), capabilities=["traffic", "security", "investigation"],
                       description="Police commander - traffic control and scene security")

    subscribe_hub_logging(hub, label="POLICE")

    print()
    print(f"  {CYAN}{BOLD}{'=' * 52}{RESET}")
    print(f"  {CYAN}{BOLD}  POLICE SERVER{RESET}")
    print(f"  {CYAN}{BOLD}{'=' * 52}{RESET}")
    print(f"  {DIM}Port: {port}  |  Model: {model}{RESET}")
    print(f"  {DIM}Agents: police{RESET}")
    print()

    async with hub.serve(host="0.0.0.0", port=port):
        print(f"  {CYAN}Serving on http://0.0.0.0:{port} — Ctrl+C to stop{RESET}")
        print()
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
