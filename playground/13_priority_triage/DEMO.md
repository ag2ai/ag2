# Priority Triage

**Category:** Priority & Conflict Resolution

Demonstrates AG2's priority system end-to-end with real LLM-powered agents.

## What it shows

| Feature | How it's used |
|---------|--------------|
| **PriorityChannel** | Hub auto-creates one when `priority_scheme` is provided. Higher-priority envelopes are delivered first. |
| **PriorityScheme** | Custom `IncidentPriorityScheme` with LOW=0, NORMAL=1, CRITICAL=2. Plugs into `Network()`. |
| **Topology routing** | `PriorityRouter` plugin inspects `envelope.priority` and reroutes CRITICAL incidents to the senior responder. |
| **ConflictResolver** | `HighestPriorityWins` — if two envelopes compete, the higher-priority one wins. |
| **Headless delegation** | `hub.delegate()` with `priority=` parameter — no initiating LLM call for the Hub itself. |

## Architecture

```
                    ┌─────────────────────────────┐
                    │     PriorityChannel (heap)   │
  INC-003 (LOW)  -->│  ┌───┐ ┌───┐ ┌────────┐    │
  INC-002 (NORM) -->│  │LOW│ │NRM│ │CRITICAL│ ◄──│── delivered first
  INC-001 (CRIT) -->│  └───┘ └───┘ └────────┘    │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │    PriorityRouter topology   │
                    │  priority >= CRITICAL?       │
                    │    YES → senior-responder    │
                    │    NO  → junior-responder    │
                    └─────────────────────────────┘
```

## Running

```bash
python playground/13_priority_triage/main.py
python playground/13_priority_triage/main.py --model gemini-3-flash-preview
```

## Key code

```python
# Custom priority scheme
class IncidentPriorityScheme:
    def compare(self, a, b):
        return int(a) - int(b)

# Topology routes by priority
class PriorityRouter(BasePlugin):
    async def process(self, envelope, ctx):
        if int(envelope.priority) >= IncidentPriority.CRITICAL:
            envelope.recipient = self._senior
        else:
            envelope.recipient = self._junior
        return envelope

# Network auto-wires PriorityChannel
network = Network(
    topology=Pipeline(priority_router),
    priority_scheme=IncidentPriorityScheme(),
    conflict_resolver=HighestPriorityWins(),
)

# Headless delegation with priority
await hub.delegate(
    source="triage-system",
    target="junior-responder",
    task="CRITICAL: DB connection timeouts...",
    priority=IncidentPriority.CRITICAL,
)
```
