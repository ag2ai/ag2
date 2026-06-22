---
title: "Evaluation"
sidebarTitle: "Overview"
---

Evaluation is how you measure whether your agent actually works. The framework runs your agent over a dataset of tasks, scores each run on multiple properties, and gives you aggregate metrics you can track over time.

One idea ties the whole framework together: a scorer grades the **recorded trace** of a run — the typed log of what the agent did. That's why the same scorers work whether you produce the trace by running the agent here, or grade a trace captured from production — and why runs persist as data you can compare over time.

## Testing vs evaluation

These sound similar but aren't.

- **Testing asserts correctness.** A unit test passes or fails. Either the function returned the right value or it didn't.
- **Evaluation measures performance.** Most agent outputs don't have a single right answer, so you grade them on many properties — did the agent call the right tool, did the final answer mention the right thing, did it stay under a token budget — and watch how the *aggregate* moves between releases.

The framework supports both. You can wrap eval metrics in pytest assertions ("pass rate must be ≥ 0.95"), but the underlying mechanism — scorers producing structured feedback that rolls up into aggregates — is built for measurement, not assertion.

## What you'll write

Every eval suite has four pieces. That's it.

1. A **dataset** — a `Suite` of tasks with inputs and (optional) expected outputs.
2. An **agent** — an `Agent` instance.
3. One or more **scorers** — functions that grade a run.
4. A call to **run_agent** — the framework's entry point.

The rest of these docs cover each piece in depth. Read the quick-start below first.

!!! note "Two ways in"
    `run_agent` runs your agent and grades it. If the traces already exist — captured from production, or produced by another tool — grade them directly with `evaluate_traces`, no agent run required. It reads both OpenTelemetry GenAI-semconv and OpenInference spans. Same scorers either way; see [Runs](runs.md).

## Quick start

The simplest plausible eval. Two tasks, one custom scorer, one prebuilt.

```python linenums="1" hl_lines="11-16 23-27 30-32 36-44"
import asyncio
from pathlib import Path

from autogen.beta import Agent, tool
from autogen.beta.config import GeminiConfig
from autogen.beta.events import ToolCallEvent
from autogen.beta.eval import Suite, run_agent, scorer
from autogen.beta.eval.scorers import tool_called

# 1. Dataset — inline tasks (or Suite.from_jsonl("tasks.jsonl"))
dataset = Suite.from_list([
    {"task_id": "t1", "inputs": {"input": "What's the weather in Tokyo?"},
     "reference_outputs": {"city": "Tokyo"{{ "}}" }},
    {"task_id": "t2", "inputs": {"input": "Weather in Paris?"},
     "reference_outputs": {"city": "Paris"{{ "}}" }},
])

# 2. Agent — one instance, reused across tasks
@tool
async def get_weather(city: str) -> str:
    return f"Sunny, 72F in {city}"

agent = Agent(
    "weather",
    config=GeminiConfig(model="gemini-3-flash-preview"),
    tools=[get_weather],
)

# 3. Scorer — a plain function with @scorer
@scorer
def called_get_weather(trace) -> bool:
    return len(trace.events_of(ToolCallEvent, name="get_weather")) == 1

# 4. Run
async def main():
    result = await run_agent(
        dataset,
        agent=agent,
        scorers=[
            called_get_weather,
            tool_called("get_weather"),  # prebuilt
        ],
        store_dir=Path("./runs"),
    )
    print(result.summary())

asyncio.run(main())
```

Output is a printed summary table plus a JSON file under `./runs/`.

```
Run abc123def
  Suite:       inline (2 tasks, source: inline)
  Runs:        2
  Duration:    3120ms
  Tokens:      input=423 output=78 total=501

Pass rates:
  called_get_weather        100.0% (2/2)
  tool_called[get_weather]  100.0% (2/2)
```

That's the whole framework in 30 lines. Everything else in these docs is variations on this shape.

!!! tip
    New to evals and this moved fast? The [Get started](getting-started.md) tutorial builds the same thing up one concept at a time.

## Determinism for CI

You don't want your eval suite to depend on a live LLM in pre-merge CI — too slow, too expensive, too flaky. The framework uses [TestConfig](../testing.md) cassettes to mock the model deterministically:

```python linenums="1" hl_lines="3-6 12"
from autogen.beta.testing import TestConfig

cassettes = {
    "t1": TestConfig(ToolCallEvent(name="get_weather", arguments='{"city":"Tokyo"}'), "Tokyo is sunny."),
    "t2": TestConfig(ToolCallEvent(name="get_weather", arguments='{"city":"Paris"}'), "Paris is sunny."),
}

result = await run_agent(
    dataset,
    agent=agent,
    scorers=[called_get_weather],
    model_config=cassettes,  # dict keyed by task_id
    store_dir=Path("./runs"),
)
```

Same suite, same scorers, no API key required. Output is identical on every machine.

## What you get back

Every `run_agent()` returns a `RunResult` and writes a JSON file to `store_dir`. The result exposes:

- `result.summary()` — a printable text table (above)
- `result.pass_rate("scorer_name")` — float between 0 and 1 for boolean scorers
- `result.score_stats("scorer_name")` — `mean / p50 / p95 / n` for numeric scorers
- `result.value_counts("scorer_name")` — `{label: count}` for categorical scorers
- `result.pass_rate("scorer_name", tag="hard")` — any accessor takes `tag=` to slice to one segment (`result.tags` lists them)
- `result.aggregates` — everything together
- `result.tasks` — per-task records with full `Trace`, feedback list, and budget status
- `result.diff(load_run("runs/old.json"))` — compare against a prior run; `.regressions` gates CI (`assert not diff.regressions`)

The JSON file is the persistence format. It's schema-versioned (`"0.1"`) and is what a future hosted dashboard will render. See [Runs](runs.md) for the full shape.

## Where to next

- **[Get started](getting-started.md)** — a paced, step-by-step first eval if this quick-start moved too fast.
- **[Scorers](scorers.md)** — how to write them, the six prebuilts (including `agent_judge` and `failure_attribution`), return-shape rules, exception handling.
- **[Runs](runs.md)** — `Suite`, the `run_agent()` signature, `RunResult` deep dive, `evaluate_traces` for existing traces, `repeats=`, and `stream=` to observe a run live.
- **[Variants](variants.md)** & **[Pairwise](pairwise.md)** — compare builds on a leaderboard, or head-to-head.
- **[Persistence & tracking](persistence.md)** — the run JSON, comparing runs for regressions, and tracking an agent across iterations.

## What's in scope

v0 ships **offline evaluation** (run a curated dataset with `run_agent`) **and grading of existing traces** (`evaluate_traces` over a directory or Grafana Tempo) — so you can already grade captured production traces with the same scorers. Still on the roadmap: *continuous* online evaluation and a hosted dashboard.
