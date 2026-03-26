#!/usr/bin/env python3
"""Operations server — Security + Maintenance agents on port 8912.

Usage:
    python playground/05_smart_building/operations_server.py
    python playground/05_smart_building/operations_server.py --model gemini-3-flash-preview
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _shared import BOLD, DIM, PORTS, RESET, YELLOW, make_maintenance, make_security, subscribe_hub_logging

from autogen.beta.network import Hub


async def main() -> None:
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "gemini-3-flash-preview"
    port = PORTS["operations"]

    hub = Hub(max_delegation_depth=4)
    await hub.register(
        make_security(model),
        capabilities=["access-control", "surveillance", "alarms"],
        description="Security manager — cameras, door locks, alarms, access logs",
    )
    await hub.register(
        make_maintenance(model),
        capabilities=["repairs", "inspections", "equipment"],
        description="Maintenance manager — work orders, equipment checks, parts",
    )

    subscribe_hub_logging(hub, label="OPS")

    print()
    print(f"  {YELLOW}{BOLD}{'=' * 52}{RESET}")
    print(f"  {YELLOW}{BOLD}  OPERATIONS SERVER{RESET}  {DIM}Security + Maintenance{RESET}")
    print(f"  {YELLOW}{BOLD}{'=' * 52}{RESET}")
    print(f"  {DIM}Port: {port}  |  Model: {model}{RESET}")
    print(f"  {DIM}Agents: security, maintenance{RESET}")
    print()

    async with hub.serve(host="0.0.0.0", port=port):
        print(f"  {YELLOW}Serving on http://0.0.0.0:{port} — Ctrl+C to stop{RESET}")
        print()
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
