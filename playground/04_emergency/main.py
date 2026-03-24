#!/usr/bin/env python3
"""Emergency Dispatch Network — in-process mode (single terminal).

All four actors run in one process with Hub providing discovery,
routing, and delegation automatically via LocalChannel.

Usage:
    python playground/04_emergency/main.py
    python playground/04_emergency/main.py --scenario 2
    python playground/04_emergency/main.py --scenario 3
    python playground/04_emergency/main.py --model gemini-3-flash-preview
    python playground/04_emergency/main.py "Chemical spill at warehouse..."

For the distributed multi-terminal version, see:
    medical_server.py, police_server.py, dispatch.py
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _shared import BOLD, DIM, GREEN, MAGENTA, RED, RESET, YELLOW, make_dispatch, make_ems, make_hospital, make_police, subscribe_hub_logging

from autogen.beta.network import Hub, TelemetryPlugin

SCENARIOS = {
    1: ("Single Critical Injury",
        "EMERGENCY CALL: Maria Torres calling from Highway 101 near exit 42. "
        "Terrible car accident - a sedan hit the median barrier and flipped. "
        "Driver is trapped, severely injured, visible blood, barely moving. "
        "Southbound lanes completely blocked. Multiple cars stopped. "
        "Please send help immediately!"),
    2: ("Multi-Casualty Pileup",
        "EMERGENCY CALL: Multi-vehicle pileup on Interstate 280 at Woodside Road. "
        "A truck jackknifed, at least 4 cars involved. 3 people injured - one "
        "unconscious on the road, one walking but bleeding heavily from the head, "
        "a child in a jammed car. Both lanes blocked, debris everywhere, "
        "one car leaking fluid. Bystanders trying to help."),
    3: ("Industrial Spinal Injury",
        "EMERGENCY CALL: Industrial accident at Patterson Steel Works, Oak Avenue. "
        "Worker fell 30 feet from scaffold onto concrete. Conscious but extreme pain, "
        "can't move legs, visible lower back deformity, difficulty breathing. "
        "Coworkers keeping him still. Send ambulance to Gate B."),
}


async def main() -> None:
    model = "gemini-3.1-pro-preview"
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]

    if "--scenario" in sys.argv:
        num = int(sys.argv[sys.argv.index("--scenario") + 1])
        title, message = SCENARIOS[num]
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        title, message = "Custom", sys.argv[1]
    else:
        title, message = SCENARIOS[1]

    hub = Hub(max_delegation_depth=4, plugins=[TelemetryPlugin()])
    await hub.register(make_dispatch(model), capabilities=["dispatch", "triage", "coordination"],
                       description="911 dispatcher - coordinates emergency response")
    await hub.register(make_ems(model), capabilities=["medical", "ambulance", "patient-care"],
                       description="EMS coordinator - ambulance dispatch and patient care")
    await hub.register(make_police(model), capabilities=["traffic", "security", "investigation"],
                       description="Police commander - traffic control and scene security")
    await hub.register(make_hospital(model), capabilities=["emergency-room", "trauma", "specialists"],
                       description="ER coordinator - hospital readiness and specialist assignment")

    subscribe_hub_logging(hub)

    print()
    print(f"  {BOLD}{'=' * 60}{RESET}")
    print(f"  {BOLD}EMERGENCY DISPATCH NETWORK{RESET}  {DIM}(in-process){RESET}")
    print(f"  {BOLD}{'=' * 60}{RESET}")
    print()
    print(f"  {YELLOW}Scenario:{RESET}  {title}")
    print(f"  {YELLOW}Model:{RESET}     {model}")
    print(f"  {YELLOW}Actors:{RESET}    dispatch, ems, police, hospital")
    print()

    t0 = time.monotonic()
    reply = await hub.ask("dispatch", message)
    elapsed = time.monotonic() - t0

    print()
    print(f"  {GREEN}{BOLD}{'=' * 60}{RESET}")
    print(f"  {GREEN}{BOLD}DISPATCH RESPONSE{RESET}  {DIM}({elapsed:.1f}s){RESET}")
    print(f"  {GREEN}{BOLD}{'=' * 60}{RESET}")
    print()
    for rline in (reply.body or "").split("\n"):
        print(f"  {rline}")
    print()

    await hub.close()


if __name__ == "__main__":
    asyncio.run(main())
