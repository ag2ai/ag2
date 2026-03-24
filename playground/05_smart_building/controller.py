#!/usr/bin/env python3
"""Controller — connects to climate + operations servers, runs scheduler + tasks.

Start the servers first, then run this:
    python playground/05_smart_building/controller.py
    python playground/05_smart_building/controller.py --scenario 2
    python playground/05_smart_building/controller.py --scenario 3
    python playground/05_smart_building/controller.py --duration 60
    python playground/05_smart_building/controller.py "Check all HVAC zones..."
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _shared import BOLD, CYAN, DIM, GREEN, MAGENTA, PORTS, RED, RESET, YELLOW, subscribe_hub_logging

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
    # Parse args
    model = "gemini-3-flash-preview"
    scenario_num = 1
    duration = 40
    custom = None

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--model" and i + 1 < len(argv):
            model = argv[i + 1]; i += 2
        elif argv[i] == "--scenario" and i + 1 < len(argv):
            scenario_num = int(argv[i + 1]); i += 2
        elif argv[i] == "--duration" and i + 1 < len(argv):
            duration = int(argv[i + 1]); i += 2
        elif not argv[i].startswith("-"):
            custom = argv[i]; i += 1
        else:
            i += 1

    title, interactive_message = SCENARIOS.get(scenario_num, SCENARIOS[1])
    if custom:
        title, interactive_message = "Custom Task", custom

    # Build network + connect to remote servers
    telemetry = TelemetryPlugin()
    network = Network(plugins=[telemetry], max_delegation_depth=4)

    print()
    print(f"  {DIM}Connecting to remote servers...{RESET}")
    try:
        climate = await network.connect(f"http://localhost:{PORTS['climate']}")
        print(f"  {GREEN}Climate server  ({PORTS['climate']}): {', '.join(climate)}{RESET}")
    except Exception as e:
        print(f"  {RED}Failed to connect to climate server: {e}{RESET}")
        print(f"  {DIM}Start it first: python playground/05_smart_building/climate_server.py{RESET}")
        return

    try:
        ops = await network.connect(f"http://localhost:{PORTS['operations']}")
        print(f"  {GREEN}Operations server ({PORTS['operations']}): {', '.join(ops)}{RESET}")
    except Exception as e:
        print(f"  {RED}Failed to connect to operations server: {e}{RESET}")
        print(f"  {DIM}Start it first: python playground/05_smart_building/operations_server.py{RESET}")
        return

    # Schedule watches
    network.schedule(IntervalWatch(15), target="hvac",
                     task="Check temperatures in all zones. Adjust any zones outside comfort range 68-74\u00b0F. Report status.")
    network.schedule(IntervalWatch(20), target="security",
                     task="Run camera sweep of all zones. Log any motion detected. Report security status.")
    network.schedule(IntervalWatch(30), target="energy",
                     task="Building closing: switch to eco mode. Reduce lighting to 20% in unoccupied zones. Report energy savings.")
    network.schedule(IntervalWatch(45), target="maintenance",
                     task="Check status of all critical equipment (HVAC units, elevators, fire suppression). Create work orders for anything needing attention.")

    subscribe_hub_logging(network.hub, label="CTRL")

    # Header
    print()
    print(f"  {BOLD}{'=' * 64}{RESET}")
    print(f"  {BOLD}SMART BUILDING CONTROLLER{RESET}  {DIM}(distributed){RESET}")
    print(f"  {BOLD}{'=' * 64}{RESET}")
    print()
    print(f"  {CYAN}Scenario:{RESET}   {title}")
    print(f"  {CYAN}Model:{RESET}      {model}")
    print(f"  {CYAN}Duration:{RESET}   {duration}s")
    print(f"  {CYAN}Remote:{RESET}     hvac + energy (:{PORTS['climate']}), security + maintenance (:{PORTS['operations']})")
    print()
    print(f"  {BOLD}Scheduled Watches:{RESET}")
    print(f"    {YELLOW}HVAC{RESET}         every 15s  temperature checks")
    print(f"    {YELLOW}Security{RESET}     every 20s  camera sweeps")
    print(f"    {YELLOW}Energy{RESET}       every 30s  closing-time optimization")
    print(f"    {YELLOW}Maintenance{RESET}  every 45s  equipment inspections")
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

    # Run scenario
    t0 = time.monotonic()

    if scenario_num == 1 and not custom:
        print(f"  {BOLD}Starting distributed autonomous operations...{RESET}")
        print(f"  {DIM}(Scheduler fires tasks to remote agents for {duration}s){RESET}")
        print()
        async with network:
            await asyncio.sleep(duration)

    elif scenario_num == 2:
        print(f"  {RED}{BOLD}Dispatching security breach to remote server...{RESET}")
        print()
        reply = await network.hub.ask("security", interactive_message or "")
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
            reply = await network.ask("maintenance", interactive_message or "")
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
        # Custom task with scheduler
        async with network:
            reply = await network.ask("hvac", custom or "")
            print()
            print(f"  {GREEN}{BOLD}RESPONSE:{RESET}")
            print()
            for rline in (reply.body or "").split("\n"):
                print(f"  {rline}")
            print()
            remaining = max(0, duration - (time.monotonic() - t0))
            if remaining > 0:
                await asyncio.sleep(remaining)

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
        print(f"    By target:")
        for agent, count in sorted(m.by_target.items()):
            print(f"      {agent}: {count}")
    print(f"    Runtime: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    asyncio.run(main())
