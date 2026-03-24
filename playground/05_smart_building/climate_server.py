#!/usr/bin/env python3
"""Climate server — HVAC + Energy agents on port 8911.

Usage:
    python playground/05_smart_building/climate_server.py
    python playground/05_smart_building/climate_server.py --model gemini-3-flash-preview
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _shared import BOLD, CYAN, DIM, PORTS, RESET, make_energy, make_hvac, subscribe_hub_logging

from autogen.beta.network import Hub


async def main() -> None:
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "gemini-3-flash-preview"
    port = PORTS["climate"]

    hub = Hub(max_delegation_depth=4)
    await hub.register(make_hvac(model), capabilities=["climate", "temperature", "ventilation"],
                       description="HVAC controller — climate control, temperature, air quality")
    await hub.register(make_energy(model), capabilities=["power", "lighting", "solar"],
                       description="Energy manager — power meters, lighting, solar, power modes")

    subscribe_hub_logging(hub, label="CLIMATE")

    print()
    print(f"  {CYAN}{BOLD}{'=' * 52}{RESET}")
    print(f"  {CYAN}{BOLD}  CLIMATE SERVER{RESET}  {DIM}HVAC + Energy{RESET}")
    print(f"  {CYAN}{BOLD}{'=' * 52}{RESET}")
    print(f"  {DIM}Port: {port}  |  Model: {model}{RESET}")
    print(f"  {DIM}Agents: hvac, energy{RESET}")
    print()

    async with hub.serve(host="0.0.0.0", port=port):
        print(f"  {CYAN}Serving on http://0.0.0.0:{port} — Ctrl+C to stop{RESET}")
        print()
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
