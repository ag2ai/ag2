#!/usr/bin/env python3
"""Emergency caller — send incidents to the dispatch center.

Usage:
    python playground/04_emergency/call.py                              # scenario 1
    python playground/04_emergency/call.py --scenario 2                 # specific scenario
    python playground/04_emergency/call.py "Chemical spill at..."       # custom emergency
    python playground/04_emergency/call.py --batch                      # 3 incidents at once
    python playground/04_emergency/call.py --connect http://host:8903   # connect new service
    python playground/04_emergency/call.py --status                     # show dispatch status
    python playground/04_emergency/call.py --list                       # list scenarios
"""

import asyncio
import sys

import aiohttp

DISPATCH_URL = "http://localhost:8900"

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"

SCENARIOS = {
    1: (
        "Critical Highway Accident",
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
    4: (
        "Minor Fender Bender",
        "EMERGENCY CALL: Low-speed fender bender at the intersection of Main St "
        "and 2nd Ave. Two cars, bumper damage only. No injuries, both drivers are "
        "out and exchanging insurance info. Minor traffic backup in the right lane. "
        "No airbags deployed. No fluids leaking.",
    ),
    5: (
        "Warehouse Fire",
        "EMERGENCY CALL: Large fire at Henderson Warehouse Complex on Industrial Blvd. "
        "Thick black smoke visible from miles away. Loading dock fully engulfed. "
        "Two workers unaccounted for, may still be inside. "
        "Nearby chemical storage building at risk of catching fire. "
        "Explosions heard. Multiple 911 calls from surrounding businesses.",
    ),
}


async def send_emergency(message: str, title: str = "Custom") -> dict | None:
    """Send a single emergency to dispatch. Returns response data."""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        try:
            async with session.post(
                f"{DISPATCH_URL}/emergency",
                json={"message": message},
            ) as resp:
                return await resp.json()
        except aiohttp.ClientError as e:
            print(f"  {RED}Cannot reach dispatch server: {e}{RESET}")
            print(f"  {DIM}Start it: python playground/04_emergency/dispatch_server.py{RESET}")
            return None


async def cmd_emergency(title: str, message: str) -> None:
    """Send one emergency and print the full result."""
    print()
    print(f"  {YELLOW}{BOLD}Calling 911...{RESET}  {DIM}{title}{RESET}")
    print()

    data = await send_emergency(message, title)
    if data is None:
        return

    if data.get("status") == "ok":
        iid = data.get("incident_id", "?")
        elapsed = data.get("elapsed", "?")
        print(f"  {GREEN}{BOLD}{'=' * 60}{RESET}")
        print(f"  {GREEN}{BOLD}INCIDENT {iid} RESOLVED{RESET}  {DIM}({elapsed}s){RESET}")
        print(f"  {GREEN}{BOLD}{'=' * 60}{RESET}")
        print()
        for line in (data.get("result") or "").split("\n"):
            print(f"  {line}")
        print()
    else:
        print(f"  {RED}Error: {data.get('reason', 'unknown')}{RESET}")


async def cmd_batch() -> None:
    """Send multiple emergencies simultaneously."""
    batch = [
        SCENARIOS[1],  # Critical — highway accident
        SCENARIOS[4],  # Minor — fender bender
        SCENARIOS[2],  # Critical — multi-casualty pileup
    ]

    print()
    print(f"  {YELLOW}{BOLD}Sending {len(batch)} emergencies simultaneously...{RESET}")
    print()
    for i, (title, _) in enumerate(batch, 1):
        print(f"    {i}. {title}")
    print()

    async def _send(num: int, title: str, message: str) -> None:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
            try:
                async with session.post(
                    f"{DISPATCH_URL}/emergency",
                    json={"message": message},
                ) as resp:
                    data = await resp.json()
                    if data.get("status") == "ok":
                        iid = data.get("incident_id", "?")
                        elapsed = data.get("elapsed", "?")
                        print(f"  {GREEN}[{num}] {title}: {iid} resolved ({elapsed}s){RESET}")
                    else:
                        print(f"  {RED}[{num}] {title}: Error — {data.get('reason')}{RESET}")
            except Exception as e:
                print(f"  {RED}[{num}] {title}: Failed — {e}{RESET}")

    await asyncio.gather(*[_send(i, title, msg) for i, (title, msg) in enumerate(batch, 1)])

    print()
    print(f"  {GREEN}{BOLD}All incidents processed.{RESET}")
    print(f"  {DIM}Check the dispatch server terminal for full delegation logs.{RESET}")
    print()


async def cmd_connect(endpoint: str) -> None:
    """Connect a new service to dispatch."""
    print(f"  {YELLOW}Connecting dispatch to {endpoint}...{RESET}")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{DISPATCH_URL}/connect",
                json={"endpoint": endpoint},
            ) as resp:
                data = await resp.json()
                if data.get("status") == "ok":
                    agents = data.get("agents", [])
                    print(f"  {GREEN}Connected! New agents: {', '.join(agents)}{RESET}")
                else:
                    print(f"  {RED}Error: {data.get('reason', 'unknown')}{RESET}")
        except aiohttp.ClientError as e:
            print(f"  {RED}Cannot reach dispatch: {e}{RESET}")


async def cmd_status() -> None:
    """Show dispatch center status."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{DISPATCH_URL}/status") as resp:
                data = await resp.json()
        except aiohttp.ClientError as e:
            print(f"  {RED}Cannot reach dispatch: {e}{RESET}")
            return

    print()
    print(f"  {BOLD}Dispatch Center Status{RESET}")
    print()
    print(f"  {BOLD}Agents:{RESET}")
    for a in data.get("agents", []):
        caps = ", ".join(a.get("capabilities", []))
        print(f"    {a['name']} [{caps}]")
    print()
    conns = data.get("connections", {})
    if conns:
        print(f"  {BOLD}Connections:{RESET}")
        for label, ep in conns.items():
            print(f"    {label}: {ep}")
        print()
    count = data.get("incident_count", 0)
    print(f"  {DIM}Incidents handled: {count}{RESET}")
    print()


def cmd_list() -> None:
    """List available scenarios."""
    print()
    print(f"  {BOLD}Available Scenarios{RESET}")
    print()
    for num, (title, desc) in SCENARIOS.items():
        print(f"    {BOLD}{num}.{RESET} {title}")
        print(f"       {DIM}{desc[:80]}...{RESET}")
        print()


async def main() -> None:
    argv = sys.argv[1:]

    if "--list" in argv or "-l" in argv:
        cmd_list()
        return

    if "--status" in argv:
        await cmd_status()
        return

    if "--connect" in argv:
        idx = argv.index("--connect")
        if idx + 1 < len(argv):
            await cmd_connect(argv[idx + 1])
        else:
            print(f"  {RED}Usage: --connect <endpoint>{RESET}")
        return

    if "--batch" in argv:
        await cmd_batch()
        return

    # Parse scenario or custom message
    scenario_num = 1
    custom = None
    i = 0
    while i < len(argv):
        if argv[i] == "--scenario" and i + 1 < len(argv):
            scenario_num = int(argv[i + 1])
            i += 2
        elif not argv[i].startswith("-"):
            custom = argv[i]
            i += 1
        else:
            i += 1

    if custom:
        title, message = "Custom", custom
    else:
        title, message = SCENARIOS[scenario_num]

    await cmd_emergency(title, message)


if __name__ == "__main__":
    asyncio.run(main())
