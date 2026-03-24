# Emergency Dispatch Network

Distributed emergency response system. Five actor types across separate servers coordinate via HTTP delegation.

## Architecture

```
  Terminal 1              Terminal 2              Terminal 3              Terminal 4 (later)
  ┌────────────────┐     ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
  │ MEDICAL SERVER  │     │ POLICE SERVER  │     │ DISPATCH CENTER│     │ FIRE DEPT      │
  │ EMS + Hospital  │     │ Police         │     │ Dispatch       │     │ Fire Chief     │
  │ :8901           │     │ :8902          │     │ :8900          │     │ :8903          │
  └───────┬────────┘     └───────┬────────┘     └──┬──────┬──────┘     └───────┬────────┘
          │                      │                  │      │                    │
          ◄──────────────────────┼──────────────────┘      │                    │
                                 ◄─────────────────────────┘                    │
                                                    ◄───────────────────────────┘
                                                     hot-connected at runtime
```

Emergencies arrive via `call.py` — a separate client that POSTs to the dispatch server.

## Quick Start

```bash
source .venv-beta/bin/activate
export GOOGLE_API_KEY="..."
```

### Terminal 1 — Medical Server (EMS + Hospital)

```bash
python playground/04_emergency/medical_server.py
```

### Terminal 2 — Police Server

```bash
python playground/04_emergency/police_server.py
```

### Terminal 3 — Dispatch Center (always-on)

```bash
python playground/04_emergency/dispatch_server.py
```

Auto-connects to medical + police on startup.

### Terminal 4 — Send Emergencies

```bash
# Single emergency (default: critical highway accident)
python playground/04_emergency/call.py

# Specific scenario
python playground/04_emergency/call.py --scenario 2

# Custom emergency
python playground/04_emergency/call.py "Chemical spill at warehouse..."

# Multiple emergencies at once (shows concurrent handling)
python playground/04_emergency/call.py --batch
```

## Demo: Hot-Adding Fire Department

Shows live service discovery — a new server joins while everything is running.

```bash
# Terminal 5 — start fire server
python playground/04_emergency/fire_server.py

# Terminal 4 — connect fire to dispatch
python playground/04_emergency/call.py --connect http://localhost:8903

# Now fire-related emergencies route to fire dept
python playground/04_emergency/call.py --scenario 5
```

## Demo: Concurrent Incidents with Priority

Sends 3 emergencies simultaneously with different severity levels:

```bash
python playground/04_emergency/call.py --batch
```

This sends:
1. Critical highway accident (full response: EMS + Police)
2. Minor fender bender (minimal response: Police only)
3. Critical multi-casualty pileup (full response: EMS + Police)

Watch the dispatch server terminal — all 3 process concurrently with priority-appropriate responses.

## Demo: Live EMS-Hospital Communication

EMS delegates to Hospital **twice** during each incident:

1. **Initial alert** — injury details, triage, ETA → Hospital prepares trauma bay + specialists
2. **Transport update** — revised vitals, updated ETA → Hospital adjusts preparations

Watch the medical server terminal to see both delegation rounds.

## Scenarios

| # | Name | Severity | Services |
|---|------|----------|----------|
| 1 | Critical Highway Accident | CRITICAL | EMS + Police |
| 2 | Multi-Casualty Pileup | CRITICAL | EMS + Police |
| 3 | Industrial Spinal Injury | SERIOUS | EMS + Police |
| 4 | Minor Fender Bender | MINOR | Police |
| 5 | Warehouse Fire | CRITICAL | Fire + EMS + Police |

## Utility Commands

```bash
# List all scenarios
python playground/04_emergency/call.py --list

# Check dispatch status (connected agents, incident count)
python playground/04_emergency/call.py --status
```

## What to Watch

1. **Dispatch terminal** — delegation flow (magenta = delegation, green = completion)
2. **Medical terminal** — EMS tools + two rounds of EMS→Hospital delegation
3. **Police terminal** — traffic control and incident reports
4. **Fire terminal** — fire engines, hazard assessment, perimeters (after connecting)
5. **call.py output** — final coordinated response

## Delegation Flow

```
call.py --HTTP POST--> dispatch_server (:8900)
                         dispatch actor
                           ├── delegate_to("ems")    --HTTP--> medical_server (:8901)
                           │                                     ems actor
                           │                                       ├── delegate_to("hospital") --local-->
                           │                                       │     hospital: prepare trauma bay
                           │                                       ├── update_patient_status (en-route)
                           │                                       └── delegate_to("hospital") --local-->
                           │                                             hospital: transport update
                           ├── delegate_to("police") --HTTP--> police_server (:8902)
                           │                                     police actor
                           └── delegate_to("fire")   --HTTP--> fire_server (:8903)  [if connected]
                                                                 fire actor
```
