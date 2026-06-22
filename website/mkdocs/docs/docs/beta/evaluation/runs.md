---
title: "Datasets & Runs"
sidebarTitle: "Runs"
---

This page covers the core of the offline eval pipeline: building a `Suite` of tasks, calling `run_agent()`, reading the `RunResult`, and grading traces that already exist. (Saving runs and comparing them over time has its own page — [Persistence & tracking](persistence.md).)

## The trace is the foundation

Grading is a pure function of one thing: a **`Trace`** — the typed record of what the agent did on a task (model responses, tool calls, tool results, human input) plus token usage, duration, and any exception. Each of those steps is emitted as an OpenTelemetry span while the agent runs, and the framework reconstructs the `Trace` from those spans. Scorers only ever read the `Trace`.

It's the **same** reconstruction no matter where the spans came from — a `run_agent` call here, a folder of saved traces, or live production telemetry in Grafana Tempo. That's exactly what lets the framework's two halves — *producing* a trace and *grading* one — share one code path and the same scorers.

Three concrete consequences:

- **Multi-turn just works.** A `reply.ask(...)` continuation re-enters the agent loop, so its spans land in the same trace. Scorers see the whole conversation, not just the first turn.
- **Offline vs online is just where the trace came from.** Offline = run a curated dataset with `run_agent`. Online = grade traces captured from production with `evaluate_traces` (e.g. from Grafana Tempo). *Same scorers, different source.*
- **No special primitives for trajectory scoring.** Assembly-policy correctness, compaction faithfulness, sub-task tree quality — all event-pattern questions, answered by the events already in the trace.

You don't manage any of this: `run_agent` attaches a telemetry middleware, collects the spans the agent emits, and reconstructs the `Trace` for you. But the frame matters when you write a custom scorer — you're always reading a `Trace`, never a live stream.

## Datasets — the `Suite`

A `Suite` is an immutable collection of `Task` records. Build one from a JSONL file or inline.

### From JSONL (recommended)

```python linenums="1"
from autogen.beta.eval import Suite

suite = Suite.from_jsonl("eval/dataset.jsonl")
```

Each line in the file is a JSON object:

```json
{"task_id": "weather-001", "inputs": {"input": "What's the weather in Tokyo?"}, "reference_outputs": {"city": "Tokyo"}, "tags": ["happy-path"]}
{"task_id": "weather-002", "inputs": {"input": "Weather in Paris?"}, "reference_outputs": {"city": "Paris"{{ "}}" }}
```

Fields:

| Field | Required | What it is |
|---|---|---|
| `task_id` | optional | Stable identifier. If absent, auto-filled as `task-0000`, `task-0001`, …. |
| `inputs` | **required** | Dict containing at least `"input"` — the prompt passed to `agent.ask(...)`. |
| `reference_outputs` | optional | Expected output for reference-based scorers. |
| `tags` | optional | List of labels. Slice results by them — `result.pass_rate(key, tag="happy-path")`. |
| `metadata` | optional | Free-form dict. Surfaces in the persisted run JSON. |

!!! tip
    JSONL is the canonical format because it's `grep`-friendly, line-diffable, and HuggingFace-compatible. Blank lines are skipped; malformed lines raise with the line number for fast diagnosis.

### From a list (for quick experimentation)

```python linenums="1"
from autogen.beta.eval import Suite

suite = Suite.from_list([
    {"inputs": {"input": "What's the weather in Tokyo?"}, "reference_outputs": {"city": "Tokyo"{{ "}}" }},
    {"inputs": {"input": "Weather in Paris?"}, "reference_outputs": {"city": "Paris"{{ "}}" }},
])
```

Same shape as JSONL lines. Use this in notebooks or for ad-hoc suites that don't deserve a file yet.

## The agent — an `Agent` instance

`run_agent(agent=...)` takes a built **`Agent` instance**, reused across every task. Each task runs on a fresh stream, so conversation history never leaks between tasks — agents are effectively stateless across `ask` calls (history lives on the per-call stream), so reusing one instance is safe for any normal agent.

```python linenums="1"
from autogen.beta import Agent
from autogen.beta.config import GeminiConfig

weather_agent = Agent(
    "weather",
    prompt="You are a weather assistant. Use get_weather.",
    config=GeminiConfig(model="gemini-3-flash-preview"),
    tools=[get_weather],
)
```

Need a different model per task — a live `ModelConfig`, or a `TestConfig` cassette for deterministic CI? Pass `model_config=` to `run_agent()`; it is forwarded to `ask` and **overrides** the agent's own config for that task. One instance covers the whole suite — no per-task rebuild, no factory.

!!! note "Multi-agent flows go through `evaluate_traces`"
    `run_agent` is for an ask-shaped target — an `Agent`. Multi-agent / network / workflow flows aren't driven by a single `ask(prompt)`, so they don't go through `run_agent`. Run them however they run — they emit OpenTelemetry spans — and grade the reconstructed trace with `evaluate_traces`. Grading is decoupled from production via the `Trace`, so the same scorers work either way.

## Running

`run_agent` produces traces via OpenTelemetry, so it requires the tracing extra — `pip install "ag2[tracing]"`. Grading pre-existing traces with `evaluate_traces` does not need it.

```python linenums="1"
from autogen.beta.eval import BudgetThresholds, run_agent

result = await run_agent(
    suite,                                 # a Suite, or a bare str (single-prompt suite)
    agent=weather_agent,
    scorers=[...],                         # list of Scorer instances
    store_dir=Path("./runs"),              # optional — where the JSON lands (omit to skip persisting)
    model_config=cassettes,                # optional — None / ModelConfig / dict[task_id, ModelConfig]
    budgets=BudgetThresholds(              # optional — observational, doesn't abort
        max_tokens_per_task=2_000,
        max_seconds_per_task=15.0,
    ),
    concurrency=4,                         # parallel task cap
    repeats=1,                             # run each task N times (consistency); pools the pass-rate
    run_id="2026-05-11-weather-suite",     # optional — overrides the UUID4 default
    label="weather-eval",                  # optional — groups runs of the same eval over time
    stream=my_stream,                      # optional — observe lifecycle events (see "Observing a run")
    span_attributes={"ag2.org.id": "..."}, # optional — stamp extra attrs on every span (see "External telemetry")
    span_processors=[...],                 # optional — also export spans to your own backend (see "External telemetry")
)
```

### `model_config` modes

The same parameter is overloaded three ways:

| Value | Behavior |
|---|---|
| `#!python None` (default) | The agent's own config is used. |
| A single `ModelConfig` | Same config for every task (overrides the agent's). |
| A `#!python dict[task_id, ModelConfig]` | Per-task config. Standard pattern for cassette-based CI. |

### Repeats — consistency

`repeats=N` runs **each task N times** — the simplest way to ask "does my agent do this *consistently*?". The per-key `pass_rate` / `score_stats` pool across every run (so 8 of 10 passing shows as `80%`), and each run gets a distinct `task_id` suffix (`"weather-001#1"`, `"weather-001#2"`, …).

```python
result = await run_agent(suite, agent=weather_agent, scorers=[...], store_dir="runs", repeats=10)
```

(At `temperature=0` consistency is near-trivial; `repeats` earns its keep when there's real nondeterminism.)

### Labels

`label="weather-eval"` stamps a **user-defined** identifier on the run, recorded at the top of the run JSON. Unlike `run_id` (unique per run), a `label` is meant to be *shared* across runs of the same eval — so a sequence of runs can be grouped and trended over time. The framework never fills it in.

### Concurrency

Tasks run in parallel up to `concurrency`, bounded by an `asyncio.Semaphore`. Default is 4. Raise it for I/O-bound suites against fast models; lower it when you're rate-limited.

### Budgets

`BudgetThresholds` records violations on each task's `budget_violation` flag but never aborts a task that goes over. The aggregate count surfaces in `result.aggregates.budget_violations` — useful as a CI regression signal ("zero tasks may exceed budget").

!!! warning
    Budgets are **observational** in v0. If you need a hard kill switch, use [observers](../advanced/observers.md) (`TokenMonitor` with `AlertPolicy(severity=FATAL)`) — the agent halts via a `HaltEvent` at runtime.

### External telemetry — exporting spans to your own backend

`run_agent` produces each task's `Trace` from OpenTelemetry spans (the same substrate `evaluate_traces` grades). Two parameters let a host platform fold those spans into its own observability stack — both default to off, so a plain run behaves exactly as before.

- **`span_attributes`** — a dict stamped on **every** span the agent emits. The run is auto-seeded with `ag2.eval.run_id`, and — when set — `ag2.eval.variant` / `ag2.eval.label`; each task additionally gets `ag2.eval.task_id`. Your own keys (e.g. `{"ag2.org.id": org_id}`) are added on top, so spans can be scoped per-org / per-run / per-task in the backend. Caller keys win on conflict.
- **`span_processors`** — extra OpenTelemetry `SpanProcessor`s attached to each task's tracer provider, **alongside** the in-memory exporter grading reads. Export is additive: the in-memory processor is never replaced, so grading output is identical whether or not you pass this. Typically a `BatchSpanProcessor(OTLPSpanExporter(...))`.

```python linenums="1"
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

result = await run_agent(
    suite, agent=weather_agent, scorers=[...], store_dir="runs",
    run_id="evalrun_abc", variant="v3", label="checkout-suite",
    span_attributes={"ag2.org.id": org_id},                   # stamped on every span
    span_processors=[BatchSpanProcessor(OTLPSpanExporter())],  # also export to your backend
)

for tr in result.tasks:
    save_row(task_id=tr.task.task_id, trace_id=tr.trace_ref.trace_id)  # real OTEL id → deep-link
```

Each task's `TaskResult.trace_ref.trace_id` is the **real OpenTelemetry trace id** of that task's spans — so a stored result row deep-links straight to the trace in your backend (Grafana Tempo, Cloud Trace, …). Multiple traces per task are possible (sub-agents); the root span's trace id is captured.

!!! warning "Flush before the process exits"
    `BatchSpanProcessor` batches spans and flushes on a timer. A short-lived eval script can exit before the last batch ships — call `processor.force_flush()` (or `shutdown()`) after `run_agent` returns when you need the spans guaranteed in the backend.

## What you get back — `RunResult`

```python linenums="1"
result = await run_agent(...)

# Run-level metadata
result.run_id              # str (UUID4 hex unless you set it) — unique per run
result.label               # str | None — user-defined; groups runs of the same eval over time
result.schema_version      # "0.1"
result.created_at          # ISO-8601 UTC
result.duration_ms         # int
result.suite               # the Suite that was executed

# Per-task records
result.tasks               # tuple[TaskResult, ...]
result.tasks[0].task       # Task
result.tasks[0].trace      # Trace
result.tasks[0].feedback   # tuple[Feedback, ...]
result.tasks[0].budget_violation  # bool
result.tasks[0].trace_ref  # TraceRef | None — .trace_id is the real OTEL trace id (deep-link)

# Aggregates
result.pass_rate("scorer_name")    # float — boolean scorers
result.score_stats("scorer_name")  # ScoreStats(mean, p50, p95, n) — numeric scorers
result.value_counts("scorer_name") # dict[label, count] — categorical scorers
result.pass_rate("scorer_name", tag="hard")  # any accessor takes tag= to slice to one segment
result.tags                        # frozenset[str] — the tags present, for slicing
result.aggregates                  # Aggregates — everything together
result.aggregates.tokens           # TokenUsage(input, output, cache_creation, cache_read)
result.diff(baseline)              # RunDiff vs a prior run — see the Persistence & tracking page
result.aggregates.errors           # int — tasks where the agent raised
result.aggregates.budget_violations  # int

# Human-readable
result.summary()           # printable multi-line table
result.save(path=None)     # re-save the run JSON (path defaults to store_dir/<run_id>.json)
```

### `summary()`

Returns a multi-line string suitable for a CI log:

```
Run 25be826dc1a94a4b9d50a4f94449139e
  Schema:      0.1
  Created:     2026-05-11T01:38:04.157919+00:00
  Duration:    5292ms
  Suite:       dataset (5 tasks, source: eval/dataset.jsonl)
  Runs:        5
  Concurrency: 4
  Errors:      0
  Budget violations: 0
  Tokens:      input=1544 output=174 total=1718

Pass rates:
  called_get_weather_once   100.0% (5/5)
  final_answer_matches      100.0% (5/5)
  no_tool_errors            100.0% (5/5)
  token_budget              100.0% (5/5)
  tool_called[get_weather]  100.0% (5/5)

Score stats:
  extra_tool_calls  mean=0.00 p50=0.00 p95=0.00 n=5

Value counts:
  termination_reason  completed=5
```

## Grading existing traces — `evaluate_traces`

`run_agent` is the *produce-and-grade* path: it runs your agent, then grades the trace. `evaluate_traces` is the *grade-only* path — for traces that already exist, captured elsewhere. The grading is identical; only the source differs.

```python linenums="1"
from autogen.beta.eval import DirectoryTraceSource, evaluate_traces

result = await evaluate_traces(
    DirectoryTraceSource("./captured-traces"),   # a folder of saved traces
    scorers=[...],
    suite=suite,            # optional — only needed for reference-based scorers (joined by task_id)
    store_dir="runs",
)
```

A **`TraceSource`** is anything that yields traces. Three ship:

| Source | Reads from |
|---|---|
| `InMemoryTraceSource` | traces you already hold in memory |
| `DirectoryTraceSource` | a folder of saved trace files (`save_trace` writes them) |
| `TempoTraceSource` | Grafana Tempo over OTLP — grade real production telemetry |

Because grading depends only on the reconstructed `Trace`, the **same scorers** work whether the trace came from `run_agent`, a directory, or production.

Reconstruction understands **two span dialects**, auto-detected per trace, so traces from a range of tools and frameworks grade unchanged:

- the [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/){.external-link target="_blank"} — `gen_ai.*` spans, what AG2's own `TelemetryMiddleware` emits, and
- [OpenInference](https://github.com/Arize-ai/openinference){.external-link target="_blank"} — `openinference.span.kind` + `llm.*` / `tool.*`, emitted by the Arize/Phoenix instrumentors.

## Observing a run

Just like `agent.ask(stream=...)`, `run_agent`, `run_variants`, and `run_pairwise` all accept a `stream`. Pass one and the runner publishes **eval lifecycle events** to it as the run unfolds — so you observe an evaluation with the same machinery you use on an agent (`subscribe`, `where`, the watch system, persistent backends). Nothing is printed for you; you attach the observer and render however you like.

```python linenums="1"
from autogen.beta.stream import MemoryStream
from autogen.beta.eval import run_variants
from autogen.beta.eval.events import VariantCompleted

stream = MemoryStream()

async def on_variant(event: VariantCompleted) -> None:
    print(f"{event.variant}: {event.result.pass_rate('final_answer_matches'):.0%}")

stream.where(VariantCompleted).subscribe(on_variant)

board = await run_variants(suite, variants=variants, scorers=[...], store_dir="runs", stream=stream)
```

The events live in `autogen.beta.eval.events` and are all transient — observational only, since the durable record of a run is its persisted JSON:

| Event | Emitted by | When |
|---|---|---|
| `EvalStarted` | `run_agent` | a run begins |
| `TaskEvaluated` | `run_agent` | each task finishes (carries its `feedback`) |
| `EvalCompleted` | `run_agent` | a run finishes (carries the `RunResult`) |
| `VariantStarted` / `VariantCompleted` | `run_variants` | around each variant (carries the variant's `RunResult`) |
| `PairwiseStarted` / `PairwiseCompared` / `PairwiseCompleted` | `run_pairwise` | around the run and each head-to-head comparison |

!!! tip "Ready-made console output"
    For a quick console view without writing a callback, subscribe the built-in `console_reporter` (from `autogen.beta.eval`): `#!python stream.subscribe(console_reporter)`. It's just an opt-in observer — no `run_agent()` flag — so swap in your own callback whenever you want different output.

## Exceptions don't abort the run

The framework catches exceptions at three levels and records them on the task instead of aborting:

| Level | What happens |
|---|---|
| `agent.ask()` raises | The exception lands on `trace.exception` (no events), and other tasks continue. |
| A scorer raises | The scorer's feedback becomes `Feedback(score=None, comment="scorer raised: ...")`. Other scorers run normally. |

The aggregate count of agent-level errors surfaces in `result.aggregates.errors`. Treat it as a CI signal — a healthy suite has zero.

## Where to next

- **[Scorers](scorers.md)** — what goes into `scorers=[...]`.
- **[Variants](variants.md)** — run several builds over the suite and rank them on a leaderboard.
- **[Pairwise](pairwise.md)** — judge two builds head-to-head.
- **[Persistence & tracking](persistence.md)** — compare runs over time, catch regressions, and track an agent across iterations.
- **[Testing](../testing.md)** — `TestConfig` cassettes for deterministic CI runs.
- **[Observers](../advanced/observers.md)** — runtime safety guards (token budgets that actually halt, loop detection).
