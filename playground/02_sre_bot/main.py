#!/usr/bin/env python3
"""SRE Bot — autonomous DevOps monitoring agent with scheduled health checks.

Category 2: Single Actor + Tools + Watch + Scheduler (Autonomous Agent)

This demo shows an agent that operates autonomously — reacting to events and
running on schedules, NOT waiting for human prompts.

Usage:
    # Scenario 1: Scheduled health checks (runs autonomously for 25s)
    python playground/02_sre_bot/main.py

    # Scenario 2: Interactive incident investigation
    python playground/02_sre_bot/main.py --scenario 2

    # Scenario 3: Scheduled + interactive combined
    python playground/02_sre_bot/main.py --scenario 3

    # Custom interval and model
    python playground/02_sre_bot/main.py --interval 8 --model gemini-3-flash-preview
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from datetime import datetime

from autogen.beta.config.gemini import GeminiConfig
from autogen.beta.events import (
    ModelResponse,
    ToolCallEvent,
    ToolResultEvent,
)
from autogen.beta.network import (
    Actor,
    IntervalWatch,
    LoopDetector,
    Network,
    Signal,
    TokenMonitor,
)
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.final import tool

# ======================================================================
# ANSI colors
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
# Simulated infrastructure state
# ======================================================================

SERVICES = ["api-gateway", "user-service", "payment-service", "order-service", "notification-service"]


# ======================================================================
# SRE Tools
# ======================================================================


@tool
async def check_service_health(service_name: str) -> str:
    """Check the health and metrics of a specific service.

    Args:
        service_name: Name of the service to check (e.g. "api-gateway", "user-service").
    """
    # ~30% chance a service is degraded
    is_degraded = random.random() < 0.30
    if is_degraded:
        status = "degraded"
        latency = random.randint(800, 3000)
        error_rate = round(random.uniform(5.0, 25.0), 1)
        uptime = round(random.uniform(95.0, 99.0), 2)
    else:
        status = "healthy"
        latency = random.randint(20, 150)
        error_rate = round(random.uniform(0.0, 1.0), 2)
        uptime = round(random.uniform(99.5, 99.99), 2)

    active_conns = random.randint(50, 500)
    pod_count = random.randint(2, 8)

    return (
        f"Service: {service_name}\n"
        f"  Status:       {status.upper()}\n"
        f"  Latency:      {latency}ms (p99)\n"
        f"  Error rate:   {error_rate}%\n"
        f"  Uptime:       {uptime}%\n"
        f"  Active conns: {active_conns}\n"
        f"  Pod count:    {pod_count}/8\n"
        f"  Last deploy:  2h 15m ago"
    )


@tool
async def restart_service(service_name: str) -> str:
    """Restart a service with rolling deployment (zero-downtime restart).

    Args:
        service_name: Name of the service to restart.
    """
    downtime_ms = random.randint(0, 200)
    pods_restarted = random.randint(2, 6)
    return (
        f"Rolling restart of {service_name} complete.\n"
        f"  Pods restarted: {pods_restarted}\n"
        f"  Downtime: {downtime_ms}ms\n"
        f"  New health check: PASSING\n"
        f"  Deployment ID: deploy-{random.randint(10000, 99999)}"
    )


@tool
async def query_logs(service_name: str, severity: str = "ERROR", minutes: int = 10) -> str:
    """Query recent logs for a service filtered by severity.

    Args:
        service_name: Name of the service to query logs for.
        severity: Log severity filter (DEBUG, INFO, WARN, ERROR, FATAL).
        minutes: How many minutes back to search.
    """
    error_messages = [
        "Connection pool exhausted, rejecting new connections",
        "Database query timeout after 30000ms",
        "Circuit breaker OPEN for downstream dependency",
        "OOM killed — container exceeded 2Gi memory limit",
        "TLS handshake failed: certificate expired",
        "Request rate exceeded 5000 rps throttle",
        "Kafka consumer lag exceeded 10000 messages",
        "Health check endpoint returned 503",
    ]
    warn_messages = [
        "Connection pool at 85% capacity",
        "Response latency p99 above SLO threshold (500ms)",
        "Retry count elevated: 12 retries in last minute",
        "Disk usage at 78% on /data volume",
        "GC pause exceeded 200ms",
    ]
    info_messages = [
        "Deployment rollout complete",
        "Auto-scaled from 3 to 5 pods",
        "Configuration reload successful",
        "Cache invalidation completed in 45ms",
    ]

    if severity.upper() == "ERROR":
        pool = error_messages
    elif severity.upper() == "WARN":
        pool = warn_messages
    else:
        pool = info_messages

    count = random.randint(2, 6)
    entries = []
    base_time = datetime.now()
    for i in range(count):
        ts = base_time.strftime("%H:%M:%S")
        msg = random.choice(pool)
        entries.append(f"  [{ts}] [{severity.upper()}] {service_name}: {msg}")

    return (
        f"Log query: service={service_name}, severity={severity}, last {minutes}m\n"
        f"Found {count} entries:\n" + "\n".join(entries)
    )


@tool
async def send_alert(channel: str, message: str, severity: str = "warning") -> str:
    """Send an alert notification to a communication channel.

    Args:
        channel: Alert channel (slack, pagerduty, email).
        message: Alert message content.
        severity: Alert severity (info, warning, critical).
    """
    alert_id = f"ALERT-{random.randint(10000, 99999)}"
    return (
        f"Alert sent to {channel.upper()}.\n"
        f"  ID: {alert_id}\n"
        f"  Severity: {severity.upper()}\n"
        f"  Message: {message}\n"
        f"  Acknowledged: pending\n"
        f"  Escalation: {'auto-escalate in 15m' if severity == 'critical' else 'manual'}"
    )


@tool
async def create_incident(title: str, severity: str, affected_services: str) -> str:
    """Create an incident ticket in the incident management system.

    Args:
        title: Incident title.
        severity: Incident severity (SEV1, SEV2, SEV3, SEV4).
        affected_services: Comma-separated list of affected services.
    """
    inc_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    return (
        f"Incident created.\n"
        f"  ID: {inc_id}\n"
        f"  Title: {title}\n"
        f"  Severity: {severity}\n"
        f"  Affected: {affected_services}\n"
        f"  Status: OPEN\n"
        f"  On-call: SRE Team (auto-assigned)\n"
        f"  Runbook: https://runbooks.internal/incident/{inc_id}"
    )


@tool
async def get_system_metrics() -> str:
    """Get the overall system dashboard metrics (CPU, memory, disk, requests)."""
    return (
        f"System Dashboard ({datetime.now().strftime('%H:%M:%S')})\n"
        f"  CPU usage:       {random.randint(30, 85)}% (cluster avg)\n"
        f"  Memory usage:    {random.randint(45, 80)}%\n"
        f"  Disk usage:      {random.randint(40, 75)}%\n"
        f"  Active requests: {random.randint(500, 5000)} rps\n"
        f"  Error rate:      {round(random.uniform(0.1, 3.0), 2)}% (global)\n"
        f"  Open incidents:  {random.randint(0, 3)}\n"
        f"  Services:        {len(SERVICES)} registered, {random.randint(len(SERVICES) - 1, len(SERVICES))} healthy"
    )


SRE_TOOLS = [
    check_service_health,
    restart_service,
    query_logs,
    send_alert,
    create_incident,
    get_system_metrics,
]

# ======================================================================
# SRE system prompt
# ======================================================================

SRE_PROMPT = """\
You are an autonomous SRE (Site Reliability Engineering) monitoring bot responsible \
for maintaining the health and reliability of a microservices platform.

Your services: api-gateway, user-service, payment-service, order-service, notification-service.

## Scheduled Health Checks
When triggered for a routine health check:
1. Call get_system_metrics to get the overall dashboard.
2. Check each service with check_service_health.
3. For any DEGRADED service:
   a. Query its ERROR logs with query_logs to understand the issue.
   b. If the error rate is above 10%, restart the service with restart_service.
   c. If the error rate is above 15%, also create an incident with create_incident \
and send a critical alert with send_alert to pagerduty.
   d. If the error rate is between 5-10%, send a warning alert to slack.
4. Provide a brief summary of overall system health.

## Incident Investigation
When asked to investigate a specific incident:
1. Check the affected service(s) health.
2. Query logs at ERROR and WARN severity.
3. Get system metrics for context.
4. If remediation is needed, restart the service.
5. Create an incident ticket if severity warrants it.
6. Send appropriate alerts.
7. Provide a detailed investigation report with root cause analysis and actions taken.

## Guidelines
- Be concise but thorough.
- Only create incidents and send alerts when there are actual problems.
- Healthy services need no action — just note they are healthy.
- Always provide a final summary of actions taken and current system state.
"""

# ======================================================================
# Event logging
# ======================================================================


def log(msg: str, *, style: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"  {_DIM}{ts}{_RESET}  {style}{msg}{_RESET}")


def subscribe_event_logging(stream: MemoryStream) -> None:
    """Subscribe to a stream to log tool calls, responses, and observer signals."""

    async def _on_event(event: object) -> None:
        if isinstance(event, ToolCallEvent):
            try:
                args = event.serialized_arguments
                parts = [f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in args.items()]
                args_str = ", ".join(parts)
                if len(args_str) > 120:
                    args_str = args_str[:120] + "..."
            except Exception:
                args_str = event.arguments[:120]
            log(f"{_YELLOW}{_BOLD}TOOL{_RESET} {_YELLOW}{event.name}{_RESET}({_DIM}{args_str}{_RESET})")

        elif isinstance(event, ToolResultEvent):
            preview = event.content[:160].replace("\n", " | ")
            log(f"  {_DIM}-> {preview}{'...' if len(event.content) > 160 else ''}{_RESET}")

        elif isinstance(event, ModelResponse) and event.content:
            preview = event.content[:200].replace("\n", " | ")
            log(
                f"{_GREEN}{_BOLD}RESPONSE:{_RESET} {_GREEN}{preview}{'...' if len(event.content) > 200 else ''}{_RESET}"
            )

        elif isinstance(event, Signal):
            log(f"{_RED}{_BOLD}ALERT [{event.severity.upper()}]{_RESET} {_RED}{event.message}{_RESET}")

    stream.subscribe(_on_event)


# ======================================================================
# Actor creation
# ======================================================================


def create_sre_actor(model: str = "gemini-3-flash-preview") -> Actor:
    return Actor(
        "sre-bot",
        prompt=SRE_PROMPT,
        config=GeminiConfig(model=model, temperature=0.3),
        tools=SRE_TOOLS,
        observers=[
            TokenMonitor(warn_threshold=10_000, alert_threshold=30_000),
            LoopDetector(repeat_threshold=3),
        ],
    )


# ======================================================================
# Header
# ======================================================================


def print_header(scenario: int, interval: int, model: str) -> None:
    scenario_names = {
        1: "Scheduled Health Checks (Autonomous)",
        2: "Incident Investigation (Interactive)",
        3: "Scheduled + Interactive (Combined)",
    }
    print()
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print(f"  {_BOLD}SRE BOT{_RESET}  {_CYAN}Autonomous DevOps Monitoring Agent{_RESET}")
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print()
    print(f"  {_BOLD}Scenario:{_RESET}  {scenario} — {scenario_names[scenario]}")
    print(f"  {_BOLD}Model:{_RESET}     {model}")
    if scenario in (1, 3):
        print(f"  {_BOLD}Interval:{_RESET}  {interval}s between health checks")
    print()
    print(f"  {_BOLD}Tools:{_RESET}")
    for t in SRE_TOOLS:
        print(f"    {_YELLOW}{t.schema.function.name}{_RESET}")
    print()
    print(f"  {_BOLD}Observers:{_RESET}")
    print(f"    {_MAGENTA}TokenMonitor{_RESET}  (warn=10k, alert=30k)")
    print(f"    {_MAGENTA}LoopDetector{_RESET}  (threshold=3)")
    print()
    print(f"  {_BOLD}Log legend:{_RESET}")
    print(f"    {_YELLOW}TOOL{_RESET}       tool call + arguments")
    print(f"    {_GREEN}RESPONSE{_RESET}   LLM final response")
    print(f"    {_RED}ALERT{_RESET}      observer signal")
    print(f"    {_CYAN}SCHEDULE{_RESET}   scheduler trigger")
    print()


# ======================================================================
# Scenario runners
# ======================================================================


async def scenario_health_checks(model: str, interval: int) -> None:
    """Scenario 1: Periodic autonomous health checks."""
    actor = create_sre_actor(model)
    network = Network()
    await network.register(actor, capabilities=["monitoring", "remediation"], description="SRE monitoring bot")

    network.schedule(
        IntervalWatch(interval),
        target="sre-bot",
        task=(
            "Perform a routine health check. Get system metrics, check all services "
            "(api-gateway, user-service, payment-service, order-service, notification-service), "
            "investigate and remediate any degraded services, and provide a status summary."
        ),
    )

    # We want 2-3 cycles: run for interval*2.5 seconds
    run_duration = interval * 2.5
    cycle = 0

    # Instrument hub._delegate to track scheduler-driven delegations.
    # The Scheduler calls hub._delegate (not hub.ask), so we wrap _delegate
    # to add logging and cycle tracking.
    original_delegate = network.hub._delegate

    async def _instrumented_delegate(to_agent, task, *, source="", **kwargs):
        nonlocal cycle
        if source == "scheduler":
            cycle += 1
            print()
            print(f"  {_CYAN}{_BOLD}{'=' * 60}{_RESET}")
            print(f"  {_CYAN}{_BOLD}=== SCHEDULED HEALTH CHECK (cycle {cycle}) ==={_RESET}")
            print(f"  {_CYAN}{_BOLD}{'=' * 60}{_RESET}")
            print()
            t0 = time.monotonic()
            # Attach event logging to a stream for this delegation
            stream = MemoryStream()
            subscribe_event_logging(stream)
            kwargs["stream"] = stream

        result = await original_delegate(to_agent, task, source=source, **kwargs)

        if source == "scheduler":
            elapsed = time.monotonic() - t0
            print()
            log(f"{_BOLD}--- Cycle {cycle} completed ({elapsed:.1f}s) ---{_RESET}")
            print()
        return result

    network.hub._delegate = _instrumented_delegate  # type: ignore[assignment]

    print(f"  {_CYAN}Starting scheduler... health checks every {interval}s{_RESET}")
    print(f"  {_CYAN}Running for {run_duration:.0f}s (expect {int(run_duration // interval)} cycles){_RESET}")
    print()

    async with network:
        await asyncio.sleep(run_duration)

    print()
    print(f"  {_BOLD}Scheduler stopped. {cycle} health check cycle(s) completed.{_RESET}")
    print()


async def scenario_investigation(model: str) -> None:
    """Scenario 2: Interactive incident investigation."""
    actor = create_sre_actor(model)
    network = Network()
    await network.register(actor, capabilities=["monitoring", "remediation"], description="SRE monitoring bot")

    incident_description = (
        "Database connection pool exhaustion on user-service. "
        "Error rate spiked to 15% in the last 10 minutes. "
        "Multiple timeout errors reported by downstream consumers. "
        "Investigate the root cause, check affected services, and remediate."
    )

    print(f"  {_MAGENTA}{_BOLD}INCIDENT:{_RESET} {_WHITE}{incident_description}{_RESET}")
    print()

    stream = MemoryStream()
    subscribe_event_logging(stream)

    async with network:
        t0 = time.monotonic()
        reply = await network.ask(actor, incident_description, stream=stream)
        elapsed = time.monotonic() - t0

    print()
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print(f"  {_GREEN}{_BOLD}INVESTIGATION COMPLETE{_RESET} ({elapsed:.1f}s)")
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print()
    if reply.body:
        for line in reply.body.split("\n"):
            print(f"  {_GREEN}{line}{_RESET}")
    print()


async def scenario_combined(model: str, interval: int) -> None:
    """Scenario 3: Scheduled health checks + interactive investigation running together."""
    actor = create_sre_actor(model)
    network = Network()
    await network.register(actor, capabilities=["monitoring", "remediation"], description="SRE monitoring bot")

    network.schedule(
        IntervalWatch(interval),
        target="sre-bot",
        task=(
            "Perform a routine health check. Get system metrics, check all services "
            "(api-gateway, user-service, payment-service, order-service, notification-service), "
            "investigate and remediate any degraded services, and provide a status summary."
        ),
    )

    cycle = 0
    original_delegate = network.hub._delegate

    async def _instrumented_delegate(to_agent, task, *, source="", **kwargs):
        nonlocal cycle
        if source == "scheduler":
            cycle += 1
            print()
            print(f"  {_CYAN}{_BOLD}{'=' * 60}{_RESET}")
            print(f"  {_CYAN}{_BOLD}=== SCHEDULED HEALTH CHECK (cycle {cycle}) ==={_RESET}")
            print(f"  {_CYAN}{_BOLD}{'=' * 60}{_RESET}")
            print()
            t0 = time.monotonic()
            # Attach event logging to a stream for this delegation
            stream = MemoryStream()
            subscribe_event_logging(stream)
            kwargs["stream"] = stream

        result = await original_delegate(to_agent, task, source=source, **kwargs)

        if source == "scheduler":
            elapsed = time.monotonic() - t0
            print()
            log(f"{_BOLD}--- Cycle {cycle} completed ({elapsed:.1f}s) ---{_RESET}")
            print()
        return result

    network.hub._delegate = _instrumented_delegate  # type: ignore[assignment]

    incident = (
        "Database connection pool exhaustion on user-service. "
        "Error rate spiked to 15% in the last 10 minutes. "
        "Investigate and remediate."
    )

    print(f"  {_CYAN}Starting scheduler... health checks every {interval}s{_RESET}")
    print(f"  {_MAGENTA}Also running interactive investigation:{_RESET}")
    print(f"  {_WHITE}{incident}{_RESET}")
    print()

    async with network:
        # Run interactive investigation while scheduler also fires
        t0 = time.monotonic()
        reply = await network.ask(actor, incident)
        elapsed = time.monotonic() - t0

        # Let the scheduler run a bit more after the investigation completes
        remaining = max(0, interval * 1.5 - elapsed)
        if remaining > 0:
            print(f"  {_CYAN}Investigation done. Letting scheduler run {remaining:.0f}s more...{_RESET}")
            print()
            await asyncio.sleep(remaining)

    print()
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print(f"  {_BOLD}Session complete. {cycle} scheduled cycle(s) + 1 investigation.{_RESET}")
    print(f"  {_BOLD}{'=' * 60}{_RESET}")
    print()
    if reply.body:
        print(f"  {_GREEN}{_BOLD}Investigation result:{_RESET}")
        for line in reply.body.split("\n"):
            print(f"  {_GREEN}{line}{_RESET}")
    print()


# ======================================================================
# CLI
# ======================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SRE Bot — autonomous DevOps monitoring agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Scenarios:\n"
            "  1  Scheduled health checks — runs autonomously for ~25s\n"
            "  2  Incident investigation — interactive, single investigation\n"
            "  3  Scheduled + interactive — both running together\n"
        ),
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Scenario to run (default: 1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="Model to use (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Seconds between health checks for scenario 1/3 (default: 10)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    print_header(args.scenario, args.interval, args.model)

    if args.scenario == 1:
        await scenario_health_checks(args.model, args.interval)
    elif args.scenario == 2:
        await scenario_investigation(args.model)
    elif args.scenario == 3:
        await scenario_combined(args.model, args.interval)


if __name__ == "__main__":
    asyncio.run(main())
