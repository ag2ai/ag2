"""Planet-Satellite Architecture: Incident Response / Root Cause Analysis

Demonstrates a planet agent acting as incident commander, investigating a
simulated production latency spike. The planet triages using diagnostic tools,
then spawns satellites to investigate different hypotheses in parallel.

Tools return realistic mock data simulating a production environment — no
external services needed.

Architecture:
    Planet (main model)     -- triages incident, forms hypotheses, delegates
    Satellites (lighter)    -- each investigates one hypothesis independently
    TokenMonitor            -- tracks token usage during investigation
    LoopDetector            -- flags if investigation goes in circles

Usage:
    export OPENAI_API_KEY=sk-...
    python playground/planet_satellite_incident.py
"""

import asyncio
import os
import sys

from autogen.beta.config.openai import OpenAIConfig
from autogen.beta.events import ModelMessageChunk
from autogen.beta.satellites import (
    LoopDetector,
    PlanetAgent,
    SatelliteFlag,
    TaskSatelliteRequest,
    TaskSatelliteResult,
    TokenMonitor,
)
from autogen.beta.stream import MemoryStream
from autogen.beta.tools import tool

# ---------------------------------------------------------------------------
# Mock tools — realistic production diagnostic data
# ---------------------------------------------------------------------------


@tool
def query_logs(service: str, severity: str = "ERROR", limit: int = 20) -> str:
    """Query application logs for a service.

    Args:
        service: Service name (api-gateway, user-service, payment-service,
                 order-service, inventory-service, database-proxy).
        severity: Log level filter (ERROR, WARN, INFO).
        limit: Maximum number of log entries to return.
    """
    logs = {
        "api-gateway": [
            "2025-06-15T14:23:01Z ERROR upstream timeout after 30000ms to user-service:8080/api/v2/users/batch",
            "2025-06-15T14:23:05Z ERROR upstream timeout after 30000ms to order-service:8080/api/v1/orders",
            "2025-06-15T14:23:12Z WARN retry attempt 3/3 for user-service — circuit breaker OPEN",
            "2025-06-15T14:23:15Z ERROR 503 Service Unavailable — all upstreams exhausted for /api/v2/users/batch",
            "2025-06-15T14:23:18Z WARN request queue depth 2,847 (threshold: 500)",
            "2025-06-15T14:23:22Z ERROR upstream timeout after 30000ms to payment-service:8080/api/v1/charge",
            "2025-06-15T14:23:30Z INFO health check OK for inventory-service",
            "2025-06-15T14:23:45Z WARN connection pool exhausted for user-service (max: 100)",
        ],
        "user-service": [
            "2025-06-15T14:20:33Z ERROR slow query detected: SELECT * FROM users WHERE id IN (...) — 28,450ms (threshold: 500ms)",
            "2025-06-15T14:20:45Z ERROR connection pool exhaustion: 0/50 available — 312 requests waiting",
            "2025-06-15T14:21:01Z WARN GC pause 4,200ms — heap usage 7.8GB/8GB",
            "2025-06-15T14:21:15Z ERROR OOM kill narrowly avoided — allocated 7.95GB of 8GB heap",
            "2025-06-15T14:21:30Z ERROR slow query detected: SELECT u.*, p.* FROM users u JOIN profiles p ON u.id = p.user_id WHERE u.region = 'eu-west' — 35,200ms",
            "2025-06-15T14:22:00Z WARN thread pool saturated: 200/200 worker threads busy",
            "2025-06-15T14:22:15Z INFO deployment user-service:v2.14.0 started at 14:15:00Z (rolling update in progress)",
            "2025-06-15T14:22:30Z ERROR query plan changed: full table scan on users (estimated 12M rows) — missing index on region column",
        ],
        "payment-service": [
            "2025-06-15T14:23:00Z WARN elevated latency to Stripe API: p99=2,100ms (baseline: 400ms)",
            "2025-06-15T14:23:10Z INFO retry succeeded for charge txn_abc123 after 2 attempts",
            "2025-06-15T14:23:20Z WARN request queue growing: 145 pending (normal: <20)",
            "2025-06-15T14:23:30Z INFO health check OK — upstream dependencies: Stripe(degraded), DB(OK)",
        ],
        "order-service": [
            "2025-06-15T14:22:45Z ERROR timeout calling user-service /api/v2/users/batch — 30,001ms",
            "2025-06-15T14:22:50Z ERROR failed to enrich order with user data — falling back to cache (MISS)",
            "2025-06-15T14:23:00Z WARN order processing backlog: 1,247 orders queued (normal: <50)",
            "2025-06-15T14:23:10Z ERROR cascade failure: order-service -> user-service timeout -> payment-service delay",
        ],
        "database-proxy": [
            "2025-06-15T14:20:15Z WARN connection count spike: 280 active (normal: 80-120)",
            "2025-06-15T14:20:30Z ERROR slow query log threshold exceeded 50 times in 60s",
            "2025-06-15T14:21:00Z WARN replication lag primary->replica-2: 12s (threshold: 2s)",
            "2025-06-15T14:21:30Z INFO active connections by service: user-service=180, order-service=45, payment-service=30, other=25",
            "2025-06-15T14:22:00Z WARN query plan regression detected after schema migration #847 (applied 14:10:00Z)",
        ],
    }
    entries = logs.get(service, [f"No logs found for service: {service}"])
    if severity != "INFO":
        entries = [e for e in entries if severity in e or "ERROR" in e]
    return "\n".join(entries[:limit])


@tool
def check_metrics(service: str) -> str:
    """Get current performance metrics for a service.

    Args:
        service: Service name.
    """
    metrics = {
        "api-gateway": (
            "api-gateway metrics (last 30 min):\n"
            "  Request rate:   12,400 req/min (baseline: 8,200)\n"
            "  Error rate:     23.4% (baseline: 0.1%)\n"
            "  p50 latency:   1,200ms (baseline: 45ms)\n"
            "  p99 latency:   31,000ms (baseline: 180ms)\n"
            "  CPU:           45% (baseline: 20%)\n"
            "  Memory:        62% (baseline: 40%)\n"
            "  Active conns:  4,200 (baseline: 1,500)"
        ),
        "user-service": (
            "user-service metrics (last 30 min):\n"
            "  Request rate:   3,800 req/min (baseline: 2,100)\n"
            "  Error rate:     45.2% (baseline: 0.05%)\n"
            "  p50 latency:   8,500ms (baseline: 25ms)\n"
            "  p99 latency:   35,000ms (baseline: 120ms)\n"
            "  CPU:           92% (baseline: 35%)\n"
            "  Memory:        98% (baseline: 55%)\n"
            "  DB query time:  28,000ms avg (baseline: 15ms)\n"
            "  Active threads: 200/200 (saturated)\n"
            "  GC pause freq:  12/min (baseline: 1/min)\n"
            "  Version:        v2.14.0 (deployed 14:15:00Z, prev: v2.13.2)"
        ),
        "payment-service": (
            "payment-service metrics (last 30 min):\n"
            "  Request rate:   1,200 req/min (baseline: 900)\n"
            "  Error rate:     3.1% (baseline: 0.2%)\n"
            "  p50 latency:   850ms (baseline: 120ms)\n"
            "  p99 latency:   4,500ms (baseline: 500ms)\n"
            "  CPU:           38% (baseline: 25%)\n"
            "  Memory:        52% (baseline: 45%)\n"
            "  Stripe API latency: 2,100ms p99 (baseline: 400ms)"
        ),
        "order-service": (
            "order-service metrics (last 30 min):\n"
            "  Request rate:   2,800 req/min (baseline: 2,000)\n"
            "  Error rate:     31.5% (baseline: 0.1%)\n"
            "  p50 latency:   12,000ms (baseline: 80ms)\n"
            "  p99 latency:   30,500ms (baseline: 250ms)\n"
            "  CPU:           55% (baseline: 30%)\n"
            "  Memory:        60% (baseline: 45%)\n"
            "  Queue depth:    1,247 (baseline: 10)"
        ),
        "database-proxy": (
            "database-proxy metrics (last 30 min):\n"
            "  Active connections: 280 (baseline: 100)\n"
            "  Queries/sec:       8,500 (baseline: 3,200)\n"
            "  Slow queries/min:  340 (baseline: 2)\n"
            "  Replication lag:   12s (baseline: 0.5s)\n"
            "  CPU:              78% (baseline: 30%)\n"
            "  IOPS:             45,000 (baseline: 12,000)\n"
            "  Buffer pool hit:   72% (baseline: 99.2%)"
        ),
    }
    return metrics.get(service, f"No metrics found for service: {service}")


@tool
def check_deployments(hours: int = 24) -> str:
    """Check recent deployments and infrastructure changes.

    Args:
        hours: How many hours back to look (default: 24).
    """
    return (
        "Recent deployments and changes (last 24h):\n"
        "\n"
        "  2025-06-15 14:15:00  user-service v2.13.2 -> v2.14.0 (rolling update)\n"
        "                       Change: Added batch user lookup endpoint /api/v2/users/batch\n"
        "                       Author: jsmith — PR #1847 'Batch user API for order enrichment'\n"
        "                       Note: includes new SQL query for batch lookups\n"
        "\n"
        "  2025-06-15 14:10:00  database migration #847 applied\n"
        "                       Change: Added `last_login` column to users table (12M rows)\n"
        "                       Author: jsmith — PR #1845 'Track user last login'\n"
        "                       Note: ALTER TABLE completed in 4m 23s\n"
        "\n"
        "  2025-06-15 10:00:00  order-service v3.8.1 -> v3.8.2\n"
        "                       Change: Bug fix for duplicate order detection\n"
        "                       Author: alee — PR #922\n"
        "\n"
        "  2025-06-14 22:00:00  infrastructure: scaled api-gateway from 3 -> 4 pods\n"
        "                       Reason: planned capacity increase for morning traffic\n"
        "\n"
        "  2025-06-14 16:00:00  payment-service v2.5.0 -> v2.5.1\n"
        "                       Change: Updated Stripe SDK to 2025.1\n"
        "                       Author: bchen — PR #445"
    )


@tool
def check_alerts() -> str:
    """Get current active alerts from the monitoring system."""
    return (
        "Active alerts:\n"
        "\n"
        "  [FIRING] P1 — api-gateway error rate >5% for 10 min\n"
        "           Started: 14:20:00Z | Duration: 8 min\n"
        "\n"
        "  [FIRING] P1 — user-service p99 latency >5s for 5 min\n"
        "           Started: 14:18:00Z | Duration: 10 min\n"
        "\n"
        "  [FIRING] P2 — database-proxy slow queries >100/min for 5 min\n"
        "           Started: 14:20:00Z | Duration: 8 min\n"
        "\n"
        "  [FIRING] P2 — user-service memory >90% for 3 min\n"
        "           Started: 14:21:00Z | Duration: 7 min\n"
        "\n"
        "  [FIRING] P3 — order-service queue depth >100\n"
        "           Started: 14:22:00Z | Duration: 6 min\n"
        "\n"
        "  [RESOLVED] P3 — payment-service Stripe latency >1s\n"
        "             Resolved: 14:25:00Z (auto-resolved, Stripe recovered)"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

planet_config = OpenAIConfig(
    model="gpt-4o",
    api_key=os.environ.get("OPENAI_API_KEY"),
    streaming=True,
)

satellite_config = OpenAIConfig(
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY"),
    streaming=True,
)

# ---------------------------------------------------------------------------
# Planet agent
# ---------------------------------------------------------------------------

PLANET_PROMPT = """\
You are a Senior SRE acting as Incident Commander for a production incident.
An alert has fired: elevated error rates and latency across multiple services.

Your investigation workflow:
1. Check active alerts with `check_alerts`.
2. Triage — use `check_metrics` and `query_logs` for the most affected services.
3. Check `check_deployments` for recent changes that correlate with the incident.
4. Based on evidence, form 3-4 hypotheses about the root cause.
5. Use `spawn_tasks` to delegate each hypothesis investigation to a satellite.
   Include ALL the evidence you've gathered in each task description so the
   satellite can reason about it.
6. Synthesise satellite findings into an incident report:
   - Timeline of events
   - Root cause (with confidence level)
   - Contributing factors
   - Immediate remediation steps
   - Follow-up action items

Be systematic. Follow the evidence. Don't jump to conclusions.
"""

planet = PlanetAgent(
    "Incident Commander",
    prompt=PLANET_PROMPT,
    config=planet_config,
    satellite_config=satellite_config,
    satellite_prompt=(
        "You are an SRE specialist investigating one specific hypothesis about "
        "a production incident. Analyse the provided evidence carefully. "
        "Consider what supports and contradicts this hypothesis. Rate your "
        "confidence (High/Medium/Low). Suggest specific remediation steps if "
        "this is the root cause. Be concise and evidence-based (300-500 words)."
    ),
    satellites=[
        TokenMonitor(warn_threshold=30_000, alert_threshold=70_000),
        LoopDetector(window_size=10, repeat_threshold=3),
    ],
    tools=[query_logs, check_metrics, check_deployments, check_alerts],
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def main() -> None:
    print(f"{'=' * 70}")
    print("Incident Response: Production Latency Spike")
    print(f"{'=' * 70}\n")

    stream = MemoryStream()
    _speaker = ""

    async def _on_event(event: object) -> None:
        nonlocal _speaker
        if isinstance(event, TaskSatelliteRequest):
            _speaker = ""
            label = event.task[:70].replace("\n", " ")
            print(
                f"\n  \033[32m[spawn]\033[0m {event.satellite_name}: {label}...",
                flush=True,
            )
        elif isinstance(event, TaskSatelliteResult):
            _speaker = ""
            print(
                f"\n  \033[32m[done]\033[0m  {event.satellite_name}: "
                f"{len(event.result)} chars",
                flush=True,
            )
        elif isinstance(event, SatelliteFlag):
            print(
                f"\n  \033[33m[flag]\033[0m  [{event.severity}] {event.message}",
                flush=True,
            )
        elif isinstance(event, ModelMessageChunk):
            if _speaker != "planet":
                _speaker = "planet"
                print(f"\n\033[1;36m  [Planet: Incident Commander] >\033[0m\n", flush=True)
            sys.stdout.write(event.content)
            sys.stdout.flush()

    stream.subscribe(_on_event)

    print("Investigating incident...\n", flush=True)
    conversation = await planet.ask(
        "INCIDENT: Multiple P1 alerts firing. Elevated error rates and latency "
        "across api-gateway, user-service, and order-service. Started ~14:18 UTC. "
        "Customer impact confirmed. Investigate and determine root cause.",
        stream=stream,
    )
    print()

    for sat in planet._satellites:
        if isinstance(sat, TokenMonitor):
            print(f"\n[Token usage: {sat.total_tokens:,} total tokens]")


if __name__ == "__main__":
    asyncio.run(main())
