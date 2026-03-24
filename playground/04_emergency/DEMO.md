# Emergency Dispatch Network

Four agents coordinate emergency response: Dispatch, EMS, Police, Hospital. Two ways to run it.

## Mode 1: In-Process (Single Terminal)

All agents in one process. Good for quick testing.

```bash
source .venv-beta/bin/activate
export GOOGLE_API_KEY="..."

python playground/04_emergency/main.py
python playground/04_emergency/main.py --scenario 2
python playground/04_emergency/main.py --scenario 3
python playground/04_emergency/main.py "Chemical spill at warehouse..."
```

## Mode 2: Distributed (3 Terminals)

Agents on separate servers, cross-server HTTP delegation via `Hub.serve()` + `Hub.connect()`. This is the real distributed demo.

```
  Terminal 1                Terminal 2              Terminal 3
  ┌──────────────────┐     ┌──────────────────┐   ┌──────────────────┐
  │  MEDICAL SERVER   │     │  POLICE SERVER   │   │  DISPATCH CLIENT │
  │  EMS + Hospital   │     │  Police          │   │  Dispatch        │
  │  :8901            │     │  :8902           │   │  (local)         │
  └────────┬─────────┘     └────────┬─────────┘   └────┬───┬─────────┘
           │                        │                   │   │
           │    ┌───────────────────┘                   │   │
           │    │                                       │   │
           ◄────┼───────────────────────────────────────┘   │
                ◄───────────────────────────────────────────┘
                        Hub.connect() over HTTP
```

### Terminal 1 — Medical Server (EMS + Hospital)

```bash
python playground/04_emergency/medical_server.py
```

EMS and Hospital run on the same Hub. When EMS delegates to Hospital, it happens locally on this server.

### Terminal 2 — Police Server

```bash
python playground/04_emergency/police_server.py
```

### Terminal 3 — Dispatch Client

```bash
python playground/04_emergency/dispatch.py
python playground/04_emergency/dispatch.py --scenario 2
python playground/04_emergency/dispatch.py --scenario 3
```

Dispatch runs locally, connects to both remote servers via `Hub.connect()`, and sends the emergency. The delegation chain:

```
dispatch --HTTP--> ems (medical:8901)
                       ems --local--> hospital (same server)
dispatch --HTTP--> police (police:8902)
```

### What to Watch

1. **Terminal 3 (dispatch)** shows the delegation flow — magenta lines for delegations, green for completions
2. **Terminal 1 (medical)** shows EMS running tools, then locally delegating to Hospital
3. **Terminal 2 (police)** shows Police running traffic control tools and filing reports
4. The framework handles all the wiring — `discover_agents` and `delegate_to` are injected automatically

## Scenarios

| # | Name | Situation |
|---|------|-----------|
| 1 | Single Critical Injury | Car accident on Highway 101. Driver trapped, lanes blocked. |
| 2 | Multi-Casualty Pileup | Truck jackknifed on I-280. 4 cars, 3 injured, debris, fluid leak. |
| 3 | Industrial Spinal Injury | Worker fell 30 feet. Can't move legs, back deformity. |

## Comparison: Old vs New

| | Old (`playground/emergency/`) | New (distributed) |
|---|---|---|
| Files | 7 files, 550+ lines | 5 files, ~350 lines total |
| Terminals | 6 (hub + 4 agents + caller) | 3 (medical + police + dispatch) |
| HTTP wiring | Manual aiohttp servers, manual registration, manual tool injection | `Hub.serve()` + `Hub.connect()` — framework handles everything |
| Discovery | Manual HTTP GET to hub | `discover_agents()` injected automatically |
| Delegation | Manual HTTP POST relay | `delegate_to()` injected automatically |
| Local delegation | N/A (all agents separate) | EMS → Hospital happens locally on same server |
