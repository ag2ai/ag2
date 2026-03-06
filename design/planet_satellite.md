# Planet-Satellite Architecture

## Motivation

Multi-agent orchestration in AG2 v1 uses GroupChat — agents take turns in a round-robin or selector-driven conversation. This works for structured dialogue but has fundamental limitations:

1. **No clear authority.** Democratic conversation confuses LLMs. When multiple agents "discuss," each tries to be helpful, leading to redundant work, contradictions, and context pollution.

2. **No passive observation.** Every agent in GroupChat is an active participant. There is no way to have a monitor that watches the conversation without contributing to it.

3. **No concurrent delegation.** GroupChat is sequential — one agent speaks at a time. There is no mechanism to fan out independent subtasks in parallel.

The event-driven stream architecture in beta makes a different pattern possible: **one agent decides, others observe and execute.**

## Core Idea

Inspired by orbital mechanics:

- **Planet agent** — the single authority. Owns analysis, planning, tool use, and user interaction. Drives the LLM loop.
- **Natural-born satellites** — passive observers that monitor the event stream. They do not interact with the user or other agents. They flag issues (cost, loops, safety) via events. Not all require an LLM.
- **Task satellites** — short-lived workers spawned by the planet for subtasks. Given a focused task, they execute independently and return results. Analogous to subagents.

Key insight: instead of constraining the planet via complex system prompts, satellites monitor behavior externally and flag issues through the event bus. Safety through observation, not restriction.

## Architecture

```
                    User
                      |
                      v
              ┌───────────────┐
              │  PlanetAgent  │  ← owns the LLM loop
              │               │  ← has spawn_task / spawn_tasks tools
              └──────┬────────┘
                     │
            ┌────────┴────────┐          Event Stream
            │   MemoryStream  │  ◄──── all events flow here
            └────────┬────────┘
                     │
        ┌────────────┼────────────────┐
        │            │                │
   ┌────▼────┐  ┌────▼─────┐  ┌──────▼──────┐
   │  Token  │  │   Loop   │  │   Custom    │    Natural-born
   │ Monitor │  │ Detector │  │  Satellite  │    satellites
   └─────────┘  └──────────┘  └─────────────┘    (observe only)

        Planet spawns task satellites on demand:

   ┌──────────────┐  ┌──────────────┐
   │ TaskSatellite│  │ TaskSatellite│   Own stream each,
   │              │  │              │   bridge progress
   └──────────────┘  └──────────────┘   to parent stream
```

## Event Types

Satellite-specific events extend `BaseEvent`:

```
SatelliteFlag             source, severity, message
SatelliteStarted          name
SatelliteCompleted        name
TaskSatelliteRequest      task, satellite_name
TaskSatelliteProgress     satellite_name, content
TaskSatelliteResult       task, satellite_name, result, usage
```

Severity levels: `INFO`, `WARNING`, `CRITICAL`, `FATAL`

These compose naturally with existing events (`ModelResponse`, `ToolCall`, `ModelMessageChunk`, etc.) — satellites subscribe to whatever they need.

## API

### Basic usage

```python
from autogen.beta.config.openai import OpenAIConfig
from autogen.beta.satellites import PlanetAgent, TokenMonitor, LoopDetector

config = OpenAIConfig(model="gpt-4o", streaming=True)

planet = PlanetAgent(
    "researcher",
    prompt="You are a research analyst. Use spawn_tasks to delegate.",
    config=config,
    satellite_config=config.copy(model="gpt-4o-mini"),  # lighter model for satellites
    satellites=[
        TokenMonitor(warn_threshold=20_000, alert_threshold=50_000),
        LoopDetector(window_size=10, repeat_threshold=3),
    ],
)

conversation = await planet.ask("Research quantum computing advances")
print(conversation.message.message.content)
```

### Streaming progress

```python
from autogen.beta.events import ModelMessageChunk
from autogen.beta.satellites import (
    TaskSatelliteProgress,
    TaskSatelliteRequest,
    TaskSatelliteResult,
    SatelliteFlag,
)

stream = MemoryStream()

async def on_event(event):
    if isinstance(event, TaskSatelliteRequest):
        print(f"[spawn] {event.satellite_name}: {event.task[:60]}")
    elif isinstance(event, TaskSatelliteProgress):
        sys.stdout.write(event.content)  # real-time satellite output
    elif isinstance(event, TaskSatelliteResult):
        print(f"[done] {event.satellite_name}")
    elif isinstance(event, ModelMessageChunk):
        sys.stdout.write(event.content)  # real-time planet output
    elif isinstance(event, SatelliteFlag):
        print(f"[{event.severity}] {event.message}")

stream.subscribe(on_event)
await planet.ask("Research topic", stream=stream)
```

### Custom satellite plug-in

Any class implementing the `Satellite` protocol can be attached:

```python
from autogen.beta.satellites import BaseSatellite, SatelliteFlag, Severity, OnEvent
from autogen.beta.events import ModelMessage

class ProfanityFilter(BaseSatellite):
    def __init__(self, words: list[str]):
        super().__init__("profanity-filter", trigger=OnEvent(ModelMessage))
        self._words = [w.lower() for w in words]

    async def process(self, events, ctx):
        for event in events:
            text = event.content.lower()
            for w in self._words:
                if w in text:
                    return SatelliteFlag(
                        source=self.name,
                        severity=Severity.WARNING,
                        message=f"Output contains '{w}' — review before sending.",
                    )
        return None

planet = PlanetAgent(
    ...,
    satellites=[
        TokenMonitor(),
        ProfanityFilter(["confidential", "internal only"]),
    ],
)
```

### Spawn tools

PlanetAgent automatically has two tools available to the LLM:

```
spawn_task(task: str) -> str
    Spawn a single task satellite. Returns its response.

spawn_tasks(tasks: list[str], parallel: bool = True) -> str
    Spawn multiple task satellites. Returns combined results.
    Runs concurrently by default.
```

The planet's LLM decides when to delegate:

```
User: "Write a market analysis of AI chip manufacturers"

Planet thinks: I'll delegate three specialist reviews...
Planet calls: spawn_tasks(
    tasks=["NVIDIA market position", "AMD AI strategy", "Emerging competitors"],
    parallel=true,
)

[3 satellites run concurrently using satellite_config]
[Progress streams to parent via TaskSatelliteProgress events]
[Results return to planet via tool return values]

Planet synthesizes satellite results into final report.
```

## Triggers

Triggers control when a satellite's `process()` fires:

```python
from autogen.beta.satellites import OnEvent, EveryNEvents

# Fire immediately for each matching event
OnEvent(ModelResponse)
OnEvent(ModelResponse | ToolCall)

# Buffer N events, then fire with the batch
EveryNEvents(10, condition=ModelResponse)
```

Both implement the `Trigger` protocol — users can implement custom triggers for time-based, threshold-based, or any other strategy.

## Flag Injection

When satellites emit `SatelliteFlag` events, the planet agent collects them in a queue. Before each LLM call, the queue is drained and injected as a temporary system prompt:

```
[SATELLITE MONITORING ALERTS]
- [WARNING] (token-monitor): Token usage warning: 22,000 tokens (threshold: 20,000).
- [WARNING] (loop-detector): Potential loop detected: tool 'search' called 3 times.
```

The planet's LLM sees these alerts and can adjust its behavior naturally — no special handling code required. The flag prompt is removed after the LLM call so it doesn't persist into subsequent turns.

## Design Decisions

**Satellites are stream subscribers.** No changes to the core `MemoryStream` or `Agent` were needed. The satellite module is purely additive.

**Task satellites get their own stream.** This prevents their internal `ModelRequest`/`ModelResponse` events from triggering the planet's subscriptions. Progress is bridged to the parent via `TaskSatelliteProgress` events.

**State persists after detach.** Calling `detach()` unsubscribes from the stream but preserves counters (e.g., `TokenMonitor.total_tokens`). Call `reset()` explicitly for a fresh session. This lets callers read satellite state after a conversation completes.

**No LLM-based satellites yet.** Built-in satellites (TokenMonitor, LoopDetector) are rule-based for predictability and zero extra cost. LLM-powered satellites (safety monitor, output validator) are a natural extension but deferred.

## Module Structure

```
autogen/beta/satellites/
├── __init__.py              # public API (18 exports)
├── events.py                # SatelliteFlag, Severity, lifecycle events
├── triggers.py              # Trigger protocol, OnEvent, EveryNEvents
├── satellite.py             # Satellite protocol, BaseSatellite ABC
├── planet.py                # PlanetAgent
└── builtins/
    ├── __init__.py
    ├── token_monitor.py     # TokenMonitor — tracks usage, flags at thresholds
    └── loop_detector.py     # LoopDetector — detects repetitive tool calls
```

## Future Work

- **LLM-powered satellites** — safety monitor (evaluate tool calls), output validator (check task outcomes)
- **Periodic trigger** — time-based firing for long-running sessions
- **Session manager** — orchestrate planet + satellite lifecycle across multi-turn conversations with persistent state
- **Event persistence tiers** — durable vs. ephemeral events for replay and debugging
- **Concurrent fan-out** — `MemoryStream.send()` currently processes subscribers sequentially; concurrent dispatch would improve latency for LLM-based satellites
- **Satellite-to-satellite communication** — flag escalation chains (e.g., cost monitor triggers safety review)
