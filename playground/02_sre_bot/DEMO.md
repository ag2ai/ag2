# SRE Bot — Autonomous DevOps Monitoring Agent

**Category 2: Single Actor + Tools + Watch + Scheduler (Autonomous Agent)**

This demo shows an agent that operates **autonomously** — reacting to events and
running on schedules, NOT waiting for human prompts. The SRE bot monitors
service health, auto-remediates degraded services, and sends alerts when issues
arise, all on a timer-based schedule.

## What This Demonstrates

- **IntervalWatch + Scheduler**: The agent runs periodic health checks without
  human intervention. The `Network` arms an `IntervalWatch` and the scheduler
  delegates tasks to the agent when it fires.
- **Autonomous decision-making**: When the agent finds a degraded service, it
  decides on its own whether to restart it, create an incident, or send an
  alert — based on error rate thresholds.
- **Observers**: `TokenMonitor` tracks LLM token usage across cycles.
  `LoopDetector` catches repetitive tool-call patterns.
- **Combined mode**: Scenario 3 shows scheduled autonomous cycles and an
  interactive investigation running concurrently through the same Network.

## Prerequisites

1. Set your Gemini API key:
   ```bash
   export GOOGLE_API_KEY="your-key-here"
   ```

2. Activate the beta virtual environment:
   ```bash
   source .venv-beta/bin/activate
   ```

## How to Run

### Scenario 1: Scheduled Health Checks (default)

The agent runs autonomously for ~25 seconds. You will see 2-3 health check
cycles fire automatically, each checking all services and remediating any
degraded ones.

```bash
python playground/02_sre_bot/main.py
```

### Scenario 2: Incident Investigation

Interactive mode. The agent investigates a simulated database connection pool
exhaustion incident on `user-service`, queries logs, checks metrics, and
produces an investigation report.

```bash
python playground/02_sre_bot/main.py --scenario 2
```

### Scenario 3: Scheduled + Interactive

Both modes at once. The scheduler fires health checks on a timer while the
agent simultaneously investigates an incident. Shows how autonomous and
interactive work coexist in the same Network.

```bash
python playground/02_sre_bot/main.py --scenario 3
```

### Options

```
--scenario 1|2|3   Scenario to run (default: 1)
--model MODEL      LLM model (default: gemini-3-flash-preview)
--interval SEC     Seconds between health checks (default: 10)
```

Example with shorter interval:
```bash
python playground/02_sre_bot/main.py --scenario 1 --interval 8
```

## What to Watch For

### Autonomous Cycles (Scenario 1 and 3)
- **Cycle markers**: Each scheduled trigger prints
  `=== SCHEDULED HEALTH CHECK (cycle N) ===` so you can track how many
  autonomous cycles have fired.
- **Tool usage pattern**: The agent should call `get_system_metrics` first, then
  `check_service_health` for each service, then investigate only the degraded
  ones.
- **Selective remediation**: Healthy services get no action. Degraded services
  trigger `query_logs`, and depending on error rate:
  - 5-10%: warning alert to Slack
  - 10%+: `restart_service`
  - 15%+: also `create_incident` + critical alert to PagerDuty
- **Randomness**: The health check tool has a ~30% chance of returning
  "degraded" for each service, so each cycle produces different results.

### Interactive Investigation (Scenario 2)
- The agent receives a specific incident and should produce a structured
  investigation report.
- Watch for the agent to check the affected service, pull logs at multiple
  severity levels, and take remediation actions.

### Observer Alerts
- If token usage crosses 10k tokens, you will see a `[WARNING]` alert from
  TokenMonitor.
- If the agent calls the same tool 3+ times with identical arguments, you will
  see a `[WARNING]` from LoopDetector.

### Combined Mode (Scenario 3)
- The interactive investigation runs immediately while the scheduler fires in
  the background.
- You should see interleaved markers for `INTERACTIVE INVESTIGATION` and
  `SCHEDULED HEALTH CHECK` cycles.

## Architecture

```
Network
  |
  +-- Hub (discovery + routing)
  |     |
  |     +-- sre-bot (Actor)
  |           +-- Tools: check_service_health, restart_service, query_logs,
  |           |          send_alert, create_incident, get_system_metrics
  |           +-- Observers: TokenMonitor, LoopDetector
  |
  +-- Scheduler
        +-- IntervalWatch(N seconds) --> target="sre-bot"
```

The key insight: the `Scheduler` owns an `IntervalWatch` that fires every N
seconds. When it fires, the scheduler tells the `Hub` to delegate a task to
`sre-bot`. The agent then runs autonomously with full tool access, just as if a
human had asked it a question — but no human is involved.
