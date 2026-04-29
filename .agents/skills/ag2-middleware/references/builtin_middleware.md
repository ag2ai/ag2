# Built-in middleware reference

All importable from `autogen.beta.middleware` (and `autogen.beta.middleware.builtin` for `TelemetryMiddleware`).

## `LoggingMiddleware`

```python
from autogen.beta.middleware import LoggingMiddleware

agent = Agent(..., middleware=[LoggingMiddleware()])
```

Logs:
- when a turn starts and finishes
- each LLM call and its response time
- each tool execution and its result

Use for quick debugging or application-level observability. For production-grade traces use `TelemetryMiddleware` (`ag2-telemetry`) instead.

No constructor args.

## `RetryMiddleware`

```python
from autogen.beta.middleware import RetryMiddleware

agent = Agent(..., middleware=[RetryMiddleware(max_retries=2)])
```

Retries failed LLM calls up to `max_retries` times. Defaults to retrying any `Exception`; narrow with `retry_on=`:

```python
RetryMiddleware(max_retries=3, retry_on=httpx.HTTPError)
```

Use for transient provider failures, network blips, occasional rate-limit responses.

## `HistoryLimiter`

```python
from autogen.beta.middleware import HistoryLimiter

agent = Agent(..., middleware=[HistoryLimiter(max_events=100)])
```

Trims event history to `max_events` before each LLM call. Preserves the first `ModelRequest` when possible and avoids leaving leading orphaned `ToolResultsEvent` entries.

Use when you want a simple, deterministic, **count-based** cap on context. For richer history shaping (token-budget, working-memory injection, sliding window with summary) use **assembly policies** instead — see `ag2-knowledge-and-memory`.

## `TokenLimiter`

```python
from autogen.beta.middleware import TokenLimiter

agent = Agent(..., middleware=[TokenLimiter(max_tokens=1000, chars_per_token=4)])
```

Char-based estimate (`len(str(event)) / chars_per_token`) — cheap, not perfectly accurate. Trims to fit the budget.

Use as a **safety net** alongside other history shaping, not as an exact meter. For accurate token counting use a model-specific tokenizer in a custom middleware.

## `TelemetryMiddleware`

OpenTelemetry instrumentation following the GenAI semantic conventions.

```python
from autogen.beta.middleware.builtin import TelemetryMiddleware

agent = Agent(
    "assistant",
    config=...,
    middleware=[
        TelemetryMiddleware(
            tracer_provider=tracer_provider,
            agent_name="assistant",
            capture_content=True,   # default; False for privacy-sensitive contexts
        ),
    ],
)
```

Full setup, span attributes, content-capture controls — see the `ag2-telemetry` skill.

## Choosing between built-ins

| Want to | Reach for |
|---|---|
| See what's happening at runtime | `LoggingMiddleware` |
| Survive flaky providers | `RetryMiddleware` |
| Hard cap on history length | `HistoryLimiter` |
| Hard cap on history size in tokens (rough) | `TokenLimiter` |
| Production-grade traces, GenAI semconv | `TelemetryMiddleware` |
| Trim history but keep a summary of dropped events | `SummarizeCompact` (see `ag2-knowledge-and-memory`) |
| Inject working memory before LLM call | `WorkingMemoryPolicy` (see `ag2-knowledge-and-memory`) |
| Approve a single tool call | `approval_required()` (see `ag2-hitl`) |

## Stacking pattern (typical)

```python
agent = Agent(
    "assistant",
    config=config,
    middleware=[
        TelemetryMiddleware(tracer_provider=tracer, agent_name="assistant"),
        LoggingMiddleware(),
        RetryMiddleware(max_retries=2),
        HistoryLimiter(max_events=200),  # last line of defence
    ],
)
```

Order: tracing on the outside (sees retries as separate spans), logging next (logs each retry), retry inner (so trim doesn't undo a successful retry), history limit closest to the LLM call (operates on what would actually be sent).

For richer history strategies (summarisation, sliding window, token-budget assembly, working memory) prefer the **assembly + compaction** pipeline documented in `ag2-knowledge-and-memory` over `HistoryLimiter` / `TokenLimiter`.
