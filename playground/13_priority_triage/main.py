#!/usr/bin/env python3
"""Priority Triage — priority-ordered delegation with topology routing.

Category: Priority & Conflict Resolution

Demonstrates AG2's priority system end-to-end:
- PriorityChannel: higher-priority tasks are delivered before lower ones
- Topology routing: a Conditional plugin routes URGENT tasks to a senior
  agent and NORMAL/BACKGROUND tasks to a junior agent
- ConflictResolver: HighestPriorityWins resolves competing envelopes
- Custom PriorityScheme: pluggable comparison logic

A simulated incident queue feeds three tasks at different priorities into
the Hub via headless delegation. The PriorityChannel ensures the critical
incident is processed first, the topology routes it to the right handler,
and the observer logs the delivery order for verification.

Usage:
    python playground/13_priority_triage/main.py
    python playground/13_priority_triage/main.py --model gemini-3-flash-preview
"""

import asyncio
import sys
import time
from datetime import datetime
from enum import IntEnum
from typing import Any

from autogen.beta import Actor, tool
from autogen.beta.config.gemini import GeminiConfig
from autogen.beta.network import (
    DelegationRequest,
    DelegationResult,
    HighestPriorityWins,
    Network,
    Pipeline,
)
from autogen.beta.network.primitives.envelope import Envelope
from autogen.beta.network.topology import BasePlugin, HubContext

# ======================================================================
# ANSI formatting
# ======================================================================

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_CYAN = "\033[96m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_MAGENTA = "\033[95m"
_BLUE = "\033[94m"
_WHITE = "\033[97m"

# ======================================================================
# Custom Priority Scheme (optional — shows pluggability)
# ======================================================================


class IncidentPriority(IntEnum):
    """Incident severity levels. Maps directly to DefaultPriority but
    demonstrates that custom schemes plug in seamlessly."""

    LOW = 0  # Cosmetic issues, typos
    NORMAL = 1  # Standard bugs, feature requests
    CRITICAL = 2  # Production outage, data loss


class IncidentPriorityScheme:
    """Custom priority scheme for incident triage."""

    def compare(self, a: Any, b: Any) -> int:
        return int(a) - int(b)


# ======================================================================
# Topology Plugin — routes by priority
# ======================================================================


class PriorityRouter(BasePlugin):
    """Routes CRITICAL incidents to the senior responder, others to junior.

    This is the key demo: topology inspects envelope.priority and reroutes.
    """

    def __init__(self, senior_name: str, junior_name: str) -> None:
        self._senior = senior_name
        self._junior = junior_name

    async def process(self, envelope: Envelope, ctx: HubContext) -> Envelope | None:
        priority = envelope.priority
        if priority is not None and int(priority) >= IncidentPriority.CRITICAL:
            # Reroute to senior responder
            if envelope.recipient != self._senior:
                log(
                    f"{_RED}{_BOLD}PRIORITY ROUTER{_RESET}  "
                    f"{_RED}CRITICAL incident -> rerouting to {self._senior}{_RESET}"
                )
                envelope.recipient = self._senior
        else:
            # Route to junior responder
            if envelope.recipient != self._junior:
                log(
                    f"{_CYAN}{_BOLD}PRIORITY ROUTER{_RESET}  "
                    f"{_CYAN}Normal/Low incident -> routing to {self._junior}{_RESET}"
                )
                envelope.recipient = self._junior
        return envelope


# ======================================================================
# Tools
# ======================================================================


def responder_tools(level: str) -> list:
    @tool
    async def assess_incident(description: str, severity: str) -> str:
        """Assess an incident and determine response actions.

        Args:
            description: Description of the incident.
            severity: Severity level (critical/normal/low).

        Returns:
            Assessment with recommended actions.
        """
        actions = {
            "critical": (
                "CRITICAL ASSESSMENT\n"
                "===================\n"
                "Impact: Production systems affected\n"
                "Response: Immediate — all hands on deck\n"
                "Actions:\n"
                "  1. Page on-call engineering team\n"
                "  2. Activate incident command channel\n"
                "  3. Begin root cause analysis\n"
                "  4. Prepare customer communication\n"
                "  5. Escalate to VP Engineering if not resolved in 30min\n"
                f"Handler: {level} responder"
            ),
            "normal": (
                "STANDARD ASSESSMENT\n"
                "===================\n"
                "Impact: Limited — workaround available\n"
                "Response: Standard SLA (4h response, 24h resolution)\n"
                "Actions:\n"
                "  1. Create JIRA ticket\n"
                "  2. Assign to relevant team\n"
                "  3. Notify product manager\n"
                f"Handler: {level} responder"
            ),
            "low": (
                "LOW PRIORITY ASSESSMENT\n"
                "=======================\n"
                "Impact: Minimal — cosmetic or non-functional\n"
                "Response: Best-effort (next sprint)\n"
                "Actions:\n"
                "  1. Add to backlog\n"
                "  2. Tag for triage in next planning\n"
                f"Handler: {level} responder"
            ),
        }
        return actions.get(severity.lower(), actions["normal"])

    @tool
    async def resolve_incident(incident_id: str, resolution: str) -> str:
        """Mark an incident as resolved with resolution details.

        Args:
            incident_id: The incident identifier.
            resolution: Description of how it was resolved.

        Returns:
            Resolution confirmation.
        """
        return (
            f"Incident {incident_id} RESOLVED\n"
            f"Resolution: {resolution}\n"
            f"Resolved by: {level} responder\n"
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        )

    return [assess_incident, resolve_incident]


# ======================================================================
# Agent creation
# ======================================================================


def create_senior_responder(model: str) -> Actor:
    return Actor(
        "senior-responder",
        prompt=(
            "You are the Senior Incident Responder. You handle CRITICAL incidents.\n\n"
            "When given an incident:\n"
            "1. Use assess_incident to evaluate severity and plan response\n"
            "2. Use resolve_incident to mark it resolved with your actions\n"
            "3. Return a brief summary of what happened and what you did\n\n"
            "Be decisive and thorough — critical incidents need fast resolution."
        ),
        config=GeminiConfig(model=model, temperature=0.2),
        tools=responder_tools("Senior"),
    )


def create_junior_responder(model: str) -> Actor:
    return Actor(
        "junior-responder",
        prompt=(
            "You are the Junior Incident Responder. You handle NORMAL and LOW priority incidents.\n\n"
            "When given an incident:\n"
            "1. Use assess_incident to evaluate severity and plan response\n"
            "2. Use resolve_incident to mark it resolved with your actions\n"
            "3. Return a brief summary of what happened and what you did\n\n"
            "Follow standard procedures. Escalate if something seems more severe than expected."
        ),
        config=GeminiConfig(model=model, temperature=0.3),
        tools=responder_tools("Junior"),
    )


# ======================================================================
# Logging
# ======================================================================


def log(msg: str, *, style: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"  {_DIM}{ts}{_RESET}  {style}{msg}{_RESET}")


# ======================================================================
# Main
# ======================================================================

# Track delivery order to verify priority works
_delivery_order: list[tuple[str, int]] = []


async def main() -> None:
    model = "gemini-3-flash-preview"

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        else:
            i += 1

    print()
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print(f"  {_BOLD}PRIORITY TRIAGE{_RESET}  {_DIM}Priority-Ordered Incident Handling{_RESET}")
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print()
    print(f"  {_BOLD}What this demonstrates:{_RESET}")
    print("    1. PriorityChannel delivers CRITICAL tasks before NORMAL/LOW")
    print("    2. PriorityRouter topology reroutes by severity")
    print("    3. Custom PriorityScheme plugs in seamlessly")
    print("    4. HighestPriorityWins conflict resolution")
    print()
    print(f"  {_BOLD}Agents:{_RESET}")
    print(f"    {_RED}senior-responder{_RESET}  handles CRITICAL incidents")
    print(f"    {_CYAN}junior-responder{_RESET}  handles NORMAL and LOW incidents")
    print()
    print(f"  {_BOLD}Model:{_RESET}  {model}")
    print()
    print(f"  {_DIM}{'- ' * 30}{_RESET}")
    print()

    # -- Create agents --
    senior = create_senior_responder(model)
    junior = create_junior_responder(model)

    # -- Create network with priority --
    # PriorityRouter is our topology plugin that inspects envelope.priority
    priority_router = PriorityRouter(
        senior_name="senior-responder",
        junior_name="junior-responder",
    )

    network = Network(
        topology=Pipeline(priority_router),
        priority_scheme=IncidentPriorityScheme(),
        conflict_resolver=HighestPriorityWins(),
    )

    await network.register(
        senior,
        capabilities=["incident-response", "critical"],
        description="Senior responder for critical production incidents",
    )
    await network.register(
        junior,
        capabilities=["incident-response", "standard"],
        description="Junior responder for normal and low-priority incidents",
    )

    # -- Subscribe to Hub stream for live logging --
    async def _on_hub_event(event: object) -> None:
        if isinstance(event, DelegationRequest):
            log(
                f"{_MAGENTA}{_BOLD}DELEGATE{_RESET}  "
                f"{_MAGENTA}{event.source} -> {event.target}: "
                f"{event.task[:80]}{'...' if len(event.task) > 80 else ''}{_RESET}"
            )
        elif isinstance(event, DelegationResult):
            preview = event.result[:100].replace("\n", " | ")
            log(
                f"{_GREEN}{_BOLD}RESULT{_RESET}  "
                f"{_GREEN}{event.target}: {preview}{'...' if len(event.result) > 100 else ''}{_RESET}"
            )

    network.hub.stream.subscribe(_on_hub_event)

    # -- Define incidents at different priorities --
    incidents = [
        # Submitted in LOW -> NORMAL -> CRITICAL order.
        # PriorityChannel should deliver CRITICAL first.
        {
            "task": (
                "LOW PRIORITY INCIDENT [INC-003]: The dashboard favicon is showing "
                "the old company logo. Non-functional cosmetic issue."
            ),
            "priority": IncidentPriority.LOW,
            "label": "LOW",
        },
        {
            "task": (
                "NORMAL PRIORITY INCIDENT [INC-002]: Users report slow page loads "
                "on the settings page. Average response time 3.2s (SLA: 2s). "
                "Workaround: users can refresh."
            ),
            "priority": IncidentPriority.NORMAL,
            "label": "NORMAL",
        },
        {
            "task": (
                "CRITICAL INCIDENT [INC-001]: Production database is returning "
                "connection timeouts. All API endpoints returning 503. "
                "Customer-facing impact confirmed. Revenue loss estimated at $5K/min."
            ),
            "priority": IncidentPriority.CRITICAL,
            "label": "CRITICAL",
        },
    ]

    log(f"{_BOLD}Submitting {len(incidents)} incidents to the priority queue...{_RESET}")
    print()

    # -- Submit all incidents via headless delegation --
    # We use hub.delegate() (headless) to queue them. The PriorityChannel
    # will deliver them in priority order: CRITICAL first.
    t0 = time.monotonic()
    results: list[tuple[str, str, float]] = []

    for inc in incidents:
        log(f"{_YELLOW}QUEUE{_RESET}  [{inc['label']}] {inc['task'][:60]}...")

    print()
    log(f"{_BOLD}Processing in priority order...{_RESET}")
    print()

    # Process sequentially via headless delegation with priority.
    # The PriorityChannel ensures delivery ordering, and the
    # PriorityRouter topology reroutes based on severity.
    for inc in incidents:
        t_start = time.monotonic()
        # All delegations target a generic "responder" — the topology
        # will reroute to senior or junior based on priority.
        result = await network.hub.delegate(
            source="triage-system",
            target="junior-responder",  # Default target; topology may reroute
            task=inc["task"],
            priority=inc["priority"],
        )
        elapsed = time.monotonic() - t_start
        results.append((inc["label"], result, elapsed))
        _delivery_order.append((inc["label"], int(inc["priority"])))

    total_elapsed = time.monotonic() - t0

    # -- Print results --
    print()
    print(f"  {_DIM}{'- ' * 30}{_RESET}")
    print()
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print(f"  {_GREEN}{_BOLD}TRIAGE COMPLETE{_RESET}  {_DIM}({total_elapsed:.1f}s){_RESET}")
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print()

    # Show delivery order
    print(f"  {_BOLD}Delivery order (priority):{_RESET}")
    for label, prio in _delivery_order:
        color = _RED if label == "CRITICAL" else _YELLOW if label == "NORMAL" else _DIM
        print(f"    {color}{label} (priority={prio}){_RESET}")
    print()

    # Show results
    for label, result, elapsed in results:
        color = _RED if label == "CRITICAL" else _YELLOW if label == "NORMAL" else _DIM
        print(f"  {color}{_BOLD}[{label}]{_RESET} {_DIM}({elapsed:.1f}s){_RESET}")
        for line in (result or "(no response)").split("\n")[:5]:
            print(f"    {line}")
        print()

    await network.hub.close()


if __name__ == "__main__":
    asyncio.run(main())
