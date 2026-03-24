# Smart Building Manager

Four autonomous agents manage a commercial building: HVAC, Security, Energy, Maintenance. Agents run on schedules AND react to events, with the Scheduler driving periodic tasks through the Hub. Two ways to run it.

## Mode 1: In-Process (Single Terminal)

All agents in one process. Good for quick testing.

```bash
source .venv-beta/bin/activate
export GOOGLE_API_KEY="..."

# Scenario 1: Autonomous (scheduler-only, 40s default)
python playground/05_smart_building/main.py

# Scenario 2: Security breach (interactive, no scheduler)
python playground/05_smart_building/main.py --scenario 2

# Scenario 3: Combined (scheduler + HVAC failure emergency)
python playground/05_smart_building/main.py --scenario 3

# Custom duration
python playground/05_smart_building/main.py --duration 60
```

## Mode 2: Distributed (3 Terminals)

Agents on separate servers. The controller connects to both and runs the Scheduler, which fires tasks to remote agents over HTTP.

```
  Terminal 1                Terminal 2                Terminal 3
  ┌──────────────────┐     ┌────────────────────┐   ┌──────────────────────┐
  │  CLIMATE SERVER   │     │  OPERATIONS SERVER │   │  CONTROLLER          │
  │  HVAC + Energy    │     │  Security + Maint. │   │  Scheduler           │
  │  :8911            │     │  :8912             │   │  (no local agents)   │
  └────────┬─────────┘     └────────┬───────────┘   └────┬───┬─────────────┘
           │                        │                     │   │
           ◄────────────────────────┼─────────────────────┘   │
                                    ◄─────────────────────────┘
                          Hub.connect() + Scheduler over HTTP
```

### Terminal 1 — Climate Server (HVAC + Energy)

```bash
python playground/05_smart_building/climate_server.py
```

### Terminal 2 — Operations Server (Security + Maintenance)

```bash
python playground/05_smart_building/operations_server.py
```

### Terminal 3 — Controller

```bash
# Autonomous building — scheduler fires tasks to remote agents
python playground/05_smart_building/controller.py

# Security breach — sends task to remote security agent
python playground/05_smart_building/controller.py --scenario 2

# Combined — scheduler + emergency to remote maintenance
python playground/05_smart_building/controller.py --scenario 3

# Longer run
python playground/05_smart_building/controller.py --duration 120
```

### What to Watch

1. **Terminal 3 (controller)** shows scheduler triggers (cyan) and delegation flow (magenta/green)
2. **Terminal 1 (climate)** shows HVAC checking temperatures and Energy adjusting power — triggered remotely by the controller's Scheduler
3. **Terminal 2 (operations)** shows Security camera sweeps and Maintenance equipment checks
4. In scenario 2, Security on Terminal 2 delegates to Energy (Terminal 1) and Maintenance (Terminal 2) — cross-server and local delegation
5. The Scheduler runs on the controller but fires tasks to agents running on completely different servers

## Scheduled Watches

| Agent | Interval | Task |
|-------|----------|------|
| HVAC | 15s | Check zone temperatures, adjust out-of-range |
| Security | 20s | Camera sweep all zones, log motion |
| Energy | 30s | Closing-time: eco mode, reduce lighting |
| Maintenance | 45s | Check critical equipment, create work orders |

## Scenarios

| # | Name | What Happens |
|---|------|-------------|
| 1 | Autonomous Building | Scheduler-only. All four watches fire on intervals. No interactive task. |
| 2 | Security Breach | Emergency response. Security investigates, delegates to Energy + Maintenance. No scheduler. |
| 3 | Combined | Scheduler runs while HVAC failure emergency is dispatched to Maintenance. |
