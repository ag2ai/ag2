---
name: ag2-observers-and-alerts
description: Monitor an AG2 beta agent's stream — log events, detect repeated tool calls, track token spend, build trigger-driven observers, route observer alerts to the model, and halt on FATAL conditions. Covers `@observer(...)` (stateless), `BaseObserver` (stateful), built-ins (`TokenMonitor`, `LoopDetector`), `Watch` primitives (`EventWatch`, `CadenceWatch`, `DelayWatch`, `IntervalWatch`, `CronWatch`, `AllOf`, `AnyOf`, `Sequence`), `ObserverAlert` (`Severity.INFO/WARNING/CRITICAL/FATAL`), `AlertPolicy`, and `HaltEvent`. Use when the user wants observability, runtime safety guards, alerts, or batch/time-based reactive logic.
license: Apache-2.0
---

# Observers, watches, and alerts

## When to use

- **Observability** — log model responses, tool calls, token usage.
- **Runtime safety** — block dangerous tool arguments, halt the agent.
- **Reactive metrics** — fire on every Nth response, or every M seconds.
- **Loop / repetition detection** — catch infinite tool-call loops.
- **Stateful monitoring** — anything that needs to remember prior events to decide what to do next.

## Two observer shapes

| Shape | When | Use |
|---|---|---|
| Stateless function | One-off event hook (logging, metrics) | `@observer(EventType)` |
| Stateful class | Counters / windows / thresholds / composed triggers | Subclass `BaseObserver` |

Both are stream subscribers under the hood — registered on the agent rather than directly on the stream.

## 60-second recipe — `@observer`

```python
from autogen.beta import Agent, observer
from autogen.beta.config import OpenAIConfig
from autogen.beta.events import ModelResponse

@observer(ModelResponse)
async def log_response(event: ModelResponse) -> None:
    print(f"Model said: {event.content}")

agent = Agent(
    "assistant",
    config=OpenAIConfig(model="gpt-4o-mini"),
    observers=[log_response],
)
```

Or attach after construction with `@agent.observer(...)`. Per-call observers also supported (`agent.ask("...", observers=[...])`).

Observer callbacks support full dependency injection (`Context`, `Inject`, `Variable`, `Depends`). Filter by event type, multiple types (`ModelRequest | ModelResponse`), or field value (`ToolCallEvent.name == "search"`). Use `interrupt=True` to modify or suppress events before regular subscribers see them.

## Built-in stateful observers

```python
from autogen.beta import Agent
from autogen.beta.observer import LoopDetector, TokenMonitor

agent = Agent(
    "assistant",
    config=config,
    observers=[
        TokenMonitor(warn_threshold=50_000, alert_threshold=100_000),
        LoopDetector(window_size=10, repeat_threshold=3),
    ],
)
```

- **`TokenMonitor`** — tracks cumulative tokens across `ModelResponse` and `TaskCompleted`. Emits `WARNING` / `CRITICAL` `ObserverAlert`s as thresholds are crossed. Read state via `monitor.total_tokens`.
- **`LoopDetector`** — sliding window of recent tool calls. Emits a `WARNING` alert when `repeat_threshold` consecutive identical calls are seen.

## Custom `BaseObserver`

A `BaseObserver` pairs a `Watch` (when to fire) with a `process()` method (what to do):

```python
from autogen.beta import Context
from autogen.beta.observer import BaseObserver
from autogen.beta.watch import CadenceWatch
from autogen.beta.events import BaseEvent, ModelResponse
from autogen.beta.events.alert import ObserverAlert, Severity

class AvgCompletionObserver(BaseObserver):
    """Every N responses, emit an INFO alert with avg completion-token count."""

    def __init__(self, window: int = 5) -> None:
        super().__init__("avg-completion", watch=CadenceWatch(n=window, condition=ModelResponse))
        self._window = window

    async def process(self, events: list[BaseEvent], ctx: Context) -> ObserverAlert | None:
        tokens = [e.usage.completion_tokens for e in events if isinstance(e, ModelResponse) and e.usage]
        if not tokens:
            return None
        return ObserverAlert(
            source=self.name,
            severity=Severity.INFO,
            message=f"Avg completion tokens over last {self._window}: {sum(tokens) / len(tokens):.0f}",
        )
```

If `process()` returns an `ObserverAlert`, the base class emits it onto the stream. You can also send events manually via `await ctx.send(...)`.

## Watch primitives — picking *when* to fire

| You need | Use |
|---|---|
| Every matching event | `EventWatch(EventType)` or just `stream.subscribe(fn, condition=...)` |
| Every N matching events | `CadenceWatch(n=N, condition=EventType)` |
| Every T seconds (buffered events) | `CadenceWatch(max_wait=T, condition=EventType)` |
| Either threshold | `CadenceWatch(n=N, max_wait=T, condition=EventType)` |
| Once after delay | `DelayWatch(seconds)` |
| Periodic timer | `IntervalWatch(seconds)` |
| Cron schedule | `CronWatch("0 9 * * MON")` |
| All sub-watches must fire | `AllOf(w1, w2)` |
| Any sub-watch fires | `AnyOf(w1, w2)` |
| In order | `Sequence(w1, w2)` |

All importable from `autogen.beta.watch`. Callback signature is uniform: `async def cb(events: list[BaseEvent], ctx: Context) -> None`. Time-driven watches pass `events=[]`.

## `ObserverAlert` — the alert type

```python
from autogen.beta.events.alert import ObserverAlert, Severity

ObserverAlert(
    source="my-observer",
    severity=Severity.WARNING,    # INFO, WARNING, CRITICAL, FATAL
    message="What happened",
)
```

**Important:** `ObserverAlert` is on the stream and persisted in history, but the default provider mappers **do not render it back to the LLM**. To make the agent see alerts, add `AlertPolicy()` to `assembly=[...]`:

```python
from autogen.beta.policies import AlertPolicy
agent = Agent("assistant", config=config, assembly=[AlertPolicy()])
```

## FATAL alerts → `HaltEvent` → short-circuit

`AlertPolicy` does two things on `Severity.FATAL`:

1. Emits a `HaltEvent` on the stream.
2. Appends a halt notice to the system prompt.

When `assembly=[...]` is non-empty, the harness automatically wires `_HaltCheckMiddleware` which sees the `HaltEvent` and short-circuits the next LLM call with a synthetic `HALTED: ...` response.

```python
from autogen.beta import Context
from autogen.beta.observer import BaseObserver
from autogen.beta.events import BaseEvent, ToolCallEvent
from autogen.beta.events.alert import HaltEvent, ObserverAlert, Severity
from autogen.beta.policies import AlertPolicy
from autogen.beta.watch import EventWatch

class PathGuardian(BaseObserver):
    def __init__(self) -> None:
        super().__init__("path-guardian", watch=EventWatch(ToolCallEvent))

    async def process(self, events: list[BaseEvent], ctx: Context) -> ObserverAlert | None:
        for event in events:
            if not isinstance(event, ToolCallEvent) or event.name != "write_file":
                continue
            if "/etc/" in event.arguments or "/usr/" in event.arguments:
                return ObserverAlert(
                    source=self.name,
                    severity=Severity.FATAL,
                    message=f"blocked dangerous write: {event.arguments}",
                )
        return None

agent = Agent(
    "safe-shell",
    prompt="...",
    config=config,
    tools=[write_file],
    observers=[PathGuardian()],
    assembly=[AlertPolicy()],   # routes FATAL → HaltEvent
)
```

The first dangerous tool call triggers FATAL → halt; the agent's next ask is short-circuited. Full runnable demo: `assets/safety_guard.py`.

## Subscribing to alerts and halts from outside

```python
from autogen.beta import MemoryStream
from autogen.beta.events.alert import HaltEvent, ObserverAlert

stream = MemoryStream()
stream.where(ObserverAlert).subscribe(lambda e: print(f"[{e.severity}] {e.source}: {e.message}"))
stream.where(HaltEvent).subscribe(lambda e: print(f"HALT: {e.reason}"))
await agent.ask("...", stream=stream)
```

## Observers vs Middleware vs Stream subscribers

| Feature | Observer | Middleware | Stream subscriber |
|---|---|---|---|
| Registered on | Agent | Agent | Stream |
| Lifecycle | Scoped to execution | Scoped to execution | Manual |
| Boilerplate | Function (or `BaseObserver`) | `BaseMiddleware` class | Function |
| Can modify events | `interrupt=True` | Yes (wraps execution) | `interrupt=True` |
| DI support | Yes | Yes | Yes |
| Use case | Monitoring, metrics, alerts | Cross-cutting (retry, auth, rate limit) | Low-level event wiring |

## Going deeper

- `assets/token_watchdog.py` — three observers (`TokenMonitor`, `LoopDetector`, custom `AlertConsole`) on one agent. Mirrors `code_examples/04`.
- `assets/safety_guard.py` — `PathGuardian` → FATAL → `AlertPolicy` → `HaltEvent` → short-circuit. Mirrors `code_examples/08`.
- Source docs:
  - `website/docs/beta/advanced/observers.mdx` — `@observer`, `BaseObserver`, registration, built-ins, `ObserverAlert`.
  - `website/docs/beta/advanced/watches.mdx` — every Watch primitive, composition rules.
  - `website/docs/beta/advanced/stream.mdx` — Stream API, `where`, `subscribe`, interrupters, `RedisStream`.
  - `website/docs/beta/advanced/assembly.mdx` — `AlertPolicy` ordering and dedup.

## Common pitfalls

- **Alerts not reaching the model** — `ObserverAlert` events are on the stream but invisible to the LLM by default. Add `AlertPolicy()` to `assembly=[...]`.
- **FATAL not halting** — `AlertPolicy` is what creates `HaltEvent`. Without `assembly=[..., AlertPolicy(), ...]` (or any non-empty assembly chain enabling `_HaltCheckMiddleware`), nothing halts.
- **Sharing one `AlertPolicy()` across agents** — dedup state lives on the instance. Give each agent its own.
- **Watch callback assumes `events` is non-empty** — for time-driven watches (`DelayWatch`, `IntervalWatch`, `CronWatch`), `events` is always `[]`.
- **Forgetting `process()` is async** — `BaseObserver.process` must be `async def`.
- **Subscribing with `subscribe(fn)` when you wanted `subscribe()` decorator** — both work; the bare-call form is `stream.subscribe(fn)`, the decorator form is `@stream.subscribe()` (with parens).
- **`CadenceWatch` with no `n` and no `max_wait`** — invalid; at least one is required.
