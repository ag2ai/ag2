#!/usr/bin/env python3
"""Medical server — EMS + Hospital agents on port 8901.

Usage:
    python playground/04_emergency/medical_server.py
    python playground/04_emergency/medical_server.py --model gemini-3-flash-preview
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _shared import BOLD, DIM, GREEN, PORTS, RESET, make_ems, make_hospital, subscribe_hub_logging

from autogen.beta.network import Hub


async def main() -> None:
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "gemini-3-flash-preview"
    port = PORTS["medical"]

    hub = Hub(max_delegation_depth=4)
    await hub.register(make_ems(model), capabilities=["medical", "ambulance", "patient-care"],
                       description="EMS coordinator - ambulance dispatch and patient care")
    await hub.register(make_hospital(model), capabilities=["emergency-room", "trauma", "specialists"],
                       description="ER coordinator - hospital readiness and specialist assignment")

    subscribe_hub_logging(hub, label="MEDICAL")

    print()
    print(f"  {GREEN}{BOLD}{'=' * 52}{RESET}")
    print(f"  {GREEN}{BOLD}  MEDICAL SERVER{RESET}  {DIM}EMS + Hospital{RESET}")
    print(f"  {GREEN}{BOLD}{'=' * 52}{RESET}")
    print(f"  {DIM}Port: {port}  |  Model: {model}{RESET}")
    print(f"  {DIM}Agents: ems, hospital{RESET}")
    print()

    async with hub.serve(host="0.0.0.0", port=port):
        print(f"  {GREEN}Serving on http://0.0.0.0:{port} — Ctrl+C to stop{RESET}")
        print()
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
