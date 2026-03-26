#!/usr/bin/env python3
"""Fire department server — Fire Chief agent on port 8903.

Start this server to add fire response capability to the dispatch network.
Connect it to dispatch: python playground/04_emergency/call.py --connect http://localhost:8903

Usage:
    python playground/04_emergency/fire_server.py
    python playground/04_emergency/fire_server.py --model gemini-3-flash-preview
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _shared import BOLD, DIM, PORTS, RED, RESET, make_fire_chief, subscribe_hub_logging

from autogen.beta.network import Hub


async def main() -> None:
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "gemini-3-flash-preview"
    port = PORTS["fire"]

    hub = Hub(max_delegation_depth=4)
    await hub.register(
        make_fire_chief(model),
        capabilities=["fire", "hazmat", "rescue"],
        description="Fire chief - fire suppression, hazmat, and rescue operations",
    )

    subscribe_hub_logging(hub, label="FIRE")

    print()
    print(f"  {RED}{BOLD}{'=' * 52}{RESET}")
    print(f"  {RED}{BOLD}  FIRE DEPARTMENT SERVER{RESET}")
    print(f"  {RED}{BOLD}{'=' * 52}{RESET}")
    print(f"  {DIM}Port: {port}  |  Model: {model}{RESET}")
    print(f"  {DIM}Agents: fire{RESET}")
    print()

    async with hub.serve(host="0.0.0.0", port=port):
        print(f"  {RED}Serving on http://0.0.0.0:{port} — Ctrl+C to stop{RESET}")
        print(f"  {DIM}Connect to dispatch:{RESET}")
        print(f"  {DIM}  python playground/04_emergency/call.py --connect http://localhost:{port}{RESET}")
        print()
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
