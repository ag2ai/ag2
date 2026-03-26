#!/usr/bin/env python3
"""Smart Building Manager — in-process mode (single terminal).

All four actors run in one process with Network (Hub + Scheduler)
providing discovery, routing, scheduling, and delegation automatically.

Usage:
    python playground/05_smart_building/main.py
    python playground/05_smart_building/main.py --scenario 2
    python playground/05_smart_building/main.py --scenario 3
    python playground/05_smart_building/main.py --duration 60
    python playground/05_smart_building/main.py --model gemini-3-flash-preview

For the distributed multi-terminal version, see:
    climate_server.py, operations_server.py, controller.py
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _shared import (
    BOLD,
    CYAN,
    DIM,
    GREEN,
    RED,
    RESET,
    YELLOW,
    make_energy,
    make_hvac,
    make_maintenance,
    make_security,
    subscribe_hub_logging,
)

from autogen.beta.network import IntervalWatch, Network, TelemetryPlugin

SCENARIOS = {
    1: ("Autonomous Building", None),
    2: (
        "Security Breach Response",
        "ALERT: Unauthorized access detected at server room door SR-01 at 22:47. "
        "Camera shows unrecognized individual. Building is in after-hours mode. "
        "Investigate immediately: check all cameras, lock down the server room, "
        "trigger intrusion alarm, and coordinate with other building systems — "
        "energy should switch to emergency lighting, maintenance should verify "
        "server room equipment integrity.",
    ),
    3: (
        "Combined: Scheduler + Emergency",
        "CRITICAL: HVAC failure on floor 3 — temperature readings at 95\u00b0F and rising. "
        "Fire suppression system check needed immediately. Create emergency work orders, "
        "coordinate with HVAC to confirm the failure and shut down the faulty unit, "
        "and ask security to verify no fire hazard via camera sweep of floor 3.",
    ),
}


async def main() -> None:
    model = "gemini-3-flash-preview"
    scenario_num = 1
    duration = 40
    custom = None

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--model" and i + 1 < len(argv):
            model = argv[i + 1]
            i += 2
        elif argv[i] == "--scenario" and i + 1 < len(argv):
            scenario_num = int(argv[i + 1])
            i += 2
        elif argv[i] == "--duration" and i + 1 < len(argv):
            duration = int(argv[i + 1])
            i += 2
        elif not argv[i].startswith("-"):
            custom = argv[i]
            i += 1
        else:
            i += 1

    title, interactive_message = SCENARIOS.get(scenario_num, SCENARIOS[1])
    if custom:
        title, interactive_message = "Custom Task", custom

    # Create actors
    hvac = make_hvac(model)
    security = make_security(model)
    energy = make_energy(model)
    maintenance = make_maintenance(model)

    # Build network
    telemetry = TelemetryPlugin()
    network = Network(plugins=[telemetry], max_delegation_depth=4)
    await network.register(hvac, capabilities=["climate", "temperature", "ventilation"],
                           description="HVAC controller — climate control, temperature, air quality")
    await network.register(security, capabilities=["access-control", "surveillance", "alarms"],
                           description="Security manager — cameras, door locks, alarms, access logs")
    await network.register(energy, capabilities=["power", "lighting", "solar"],
                           description="Energy manager — power meters, lighting, solar, power modes")
    await network.register(maintenance, capabilities=["repairs", "inspections", "equipment"],
                           description="Maintenance manager — work orders, equipment checks, parts")

    # Schedule watches
    network.schedule(IntervalWatch(15), target="hvac",
                     task="Check temperatures in all zones. Adjust any zones outside comfort range 68-74\u00b0F. Report status.")
    network.schedule(IntervalWatch(20), target="security",
                     task="Run camera sweep of all zones. Log any motion detected. Report security status.")
    network.schedule(IntervalWatch(30), target="energy",
                     task="Building closing: switch to eco mode. Reduce lighting to 20% in unoccupied zones. Report energy savings.")
    network.schedule(IntervalWatch(45), target="maintenance",
                     task="Check status of all critical equipment (HVAC units, elevators, fire suppression). Create work orders for anything needing attention.")

    subscribe_hub_logging(network.hub)

    # Header
    print()
    print(f"  {BOLD}{'=' * 64}{RESET}")
    print(f"  {BOLD}SMART BUILDING MANAGER{RESET}  {DIM}(in-process){RESET}")
    print(f"  {BOLD}{'=' * 64}{RESET}")
    print()
    print(f"  {CYAN}Scenario:{RESET}   {title}")
    print(f"  {CYAN}Model:{RESET}      {model}")
    print(f"  {CYAN}Duration:{RESET}   {duration}s")
    print(f"  {CYAN}Actors:{RESET}     hvac, security, energy, maintenance")
    print()
    print(f"  {BOLD}Scheduled Watches:{RESET}")
    print(f"    {YELLOW}HVAC{RESET}         every 15s")
    print(f"    {YELLOW}Security{RESET}     every 20s")
    print(f"    {YELLOW}Energy{RESET}       every 30s")
    print(f"    {YELLOW}Maintenance{RESET}  every 45s")
    print()

    if interactive_message:
        print(f"  {BOLD}Interactive task:{RESET}")
        words = interactive_message.split()
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

    print(f"  {DIM}{'─' * 64}{RESET}")
    print()

    # Run
    t0 = time.monotonic()

    if scenario_num == 1 and not custom:
        print(f"  {BOLD}Starting autonomous building operations...{RESET}")
        print(f"  {DIM}(Scheduler will run for {duration}s){RESET}")
        print()
        async with network:
            await asyncio.sleep(duration)

    elif scenario_num == 2:
        print(f"  {RED}{BOLD}Dispatching security breach response...{RESET}")
        print()
        reply = await network.hub.ask(security, interactive_message or "")
        elapsed = time.monotonic() - t0
        print()
        print(f"  {GREEN}{BOLD}SECURITY RESPONSE{RESET}  {DIM}({elapsed:.1f}s){RESET}")
        print()
        for rline in (reply.body or "").split("\n"):
            print(f"  {rline}")
        print()
        await network.hub.close()
        _print_telemetry(telemetry, time.monotonic() - t0)
        return

    elif scenario_num == 3:
        print(f"  {RED}{BOLD}Starting scheduler + dispatching emergency...{RESET}")
        print()
        async with network:
            reply = await network.ask(maintenance, interactive_message or "")
            elapsed_task = time.monotonic() - t0
            print()
            print(f"  {GREEN}{BOLD}EMERGENCY RESPONSE{RESET}  {DIM}({elapsed_task:.1f}s){RESET}")
            print()
            for rline in (reply.body or "").split("\n"):
                print(f"  {rline}")
            print()
            remaining = max(0, duration - (time.monotonic() - t0))
            if remaining > 0:
                print(f"  {DIM}(Scheduler continues for {remaining:.0f}s more...){RESET}")
                await asyncio.sleep(remaining)

    else:
        async with network:
            await asyncio.sleep(duration)

    elapsed = time.monotonic() - t0
    print()
    print(f"  {GREEN}{BOLD}BUILDING OPERATIONS COMPLETE{RESET}  {DIM}({elapsed:.1f}s){RESET}")
    print()
    _print_telemetry(telemetry, elapsed)


def _print_telemetry(telemetry: TelemetryPlugin, elapsed: float) -> None:
    m = telemetry.metrics
    print(f"  {BOLD}Telemetry:{RESET}")
    print(f"    Total delegations:  {m.total_delegations}")
    print(f"    Total completions:  {m.total_completions}")
    if m.by_target:
        print("    By target:")
        for agent, count in sorted(m.by_target.items()):
            print(f"      {agent}: {count}")
    print(f"    Runtime: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    asyncio.run(main())
