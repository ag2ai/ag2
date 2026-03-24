#!/usr/bin/env python3
"""Dispatch client — connects to medical + police servers and sends an emergency.

Start the servers first, then run this:
    python playground/04_emergency/dispatch.py
    python playground/04_emergency/dispatch.py --scenario 2
    python playground/04_emergency/dispatch.py --scenario 3
    python playground/04_emergency/dispatch.py "Chemical spill at warehouse..."
    python playground/04_emergency/dispatch.py --model gemini-3-flash-preview
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _shared import BOLD, DIM, GREEN, MAGENTA, PORTS, RED, RESET, YELLOW, make_dispatch, subscribe_hub_logging

from autogen.beta.network import Hub

SCENARIOS = {
    1: (
        "Single Critical Injury",
        "EMERGENCY CALL: Maria Torres calling from Highway 101 near exit 42. "
        "Terrible car accident - a sedan hit the median barrier and flipped. "
        "Driver is trapped, severely injured, visible blood, barely moving. "
        "Southbound lanes completely blocked. Multiple cars stopped. "
        "Please send help immediately!",
    ),
    2: (
        "Multi-Casualty Pileup",
        "EMERGENCY CALL: Multi-vehicle pileup on Interstate 280 at Woodside Road. "
        "A truck jackknifed, at least 4 cars involved. 3 people injured - one "
        "unconscious on the road, one walking but bleeding heavily from the head, "
        "a child in a jammed car. Both lanes blocked, debris everywhere, "
        "one car leaking fluid. Bystanders trying to help.",
    ),
    3: (
        "Industrial Spinal Injury",
        "EMERGENCY CALL: Industrial accident at Patterson Steel Works, Oak Avenue. "
        "Worker fell 30 feet from scaffold onto concrete. Conscious but extreme pain, "
        "can't move legs, visible lower back deformity, difficulty breathing. "
        "Coworkers keeping him still. Send ambulance to Gate B.",
    ),
}


async def main() -> None:
    # Parse args
    model = "gemini-3.1-pro-preview"
    scenario_num = 1
    custom = None

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--model" and i + 1 < len(argv):
            model = argv[i + 1]; i += 2
        elif argv[i] == "--scenario" and i + 1 < len(argv):
            scenario_num = int(argv[i + 1]); i += 2
        elif not argv[i].startswith("-"):
            custom = argv[i]; i += 1
        else:
            i += 1

    if custom:
        title, message = "Custom", custom
    else:
        title, message = SCENARIOS[scenario_num]

    # Create local dispatch hub
    hub = Hub(max_delegation_depth=4)
    await hub.register(make_dispatch(model), capabilities=["dispatch", "triage", "coordination"],
                       description="911 dispatcher - coordinates emergency response")

    # Connect to remote servers
    print()
    print(f"  {DIM}Connecting to remote servers...{RESET}")
    try:
        med = await hub.connect(f"http://localhost:{PORTS['medical']}")
        print(f"  {GREEN}Medical server ({PORTS['medical']}): {', '.join(med)}{RESET}")
    except Exception as e:
        print(f"  {RED}Failed to connect to medical server: {e}{RESET}")
        print(f"  {DIM}Start it first: python playground/04_emergency/medical_server.py{RESET}")
        await hub.close()
        return

    try:
        pol = await hub.connect(f"http://localhost:{PORTS['police']}")
        print(f"  {GREEN}Police server  ({PORTS['police']}): {', '.join(pol)}{RESET}")
    except Exception as e:
        print(f"  {RED}Failed to connect to police server: {e}{RESET}")
        print(f"  {DIM}Start it first: python playground/04_emergency/police_server.py{RESET}")
        await hub.close()
        return

    subscribe_hub_logging(hub, label="DISPATCH")

    # Header
    print()
    print(f"  {YELLOW}{BOLD}{'=' * 60}{RESET}")
    print(f"  {YELLOW}{BOLD}  911 DISPATCH{RESET}  {DIM}{title}{RESET}")
    print(f"  {YELLOW}{BOLD}{'=' * 60}{RESET}")
    print()
    print(f"  {DIM}Delegation flow:{RESET}")
    print(f"    dispatch {MAGENTA}--HTTP-->{RESET} ems (medical:{PORTS['medical']})")
    print(f"                         ems {MAGENTA}--local-->{RESET} hospital")
    print(f"    dispatch {MAGENTA}--HTTP-->{RESET} police (police:{PORTS['police']})")
    print()

    # Print emergency call
    print(f"  {BOLD}Emergency call:{RESET}")
    words = message.split()
    line = "    "
    for w in words:
        if len(line) + len(w) + 1 > 76:
            print(line)
            line = "    " + w
        else:
            line += " " + w if len(line) > 4 else w
    if len(line) > 4:
        print(line)
    print()
    print(f"  {DIM}{'─' * 60}{RESET}")
    print()

    # Run
    t0 = time.monotonic()
    reply = await hub.ask("dispatch", message)
    elapsed = time.monotonic() - t0

    # Result
    print()
    print(f"  {GREEN}{BOLD}{'=' * 60}{RESET}")
    print(f"  {GREEN}{BOLD}  DISPATCH RESPONSE{RESET}  {DIM}({elapsed:.1f}s){RESET}")
    print(f"  {GREEN}{BOLD}{'=' * 60}{RESET}")
    print()
    for rline in (reply.content or "").split("\n"):
        print(f"  {rline}")
    print()

    await hub.close()


if __name__ == "__main__":
    asyncio.run(main())
