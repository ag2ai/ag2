---
title: "Get started"
sidebarTitle: "Get started"
---

A hands-on walk-through: by the end you'll have run a real evaluation and understood every line. We build it up one concept at a time — no prior eval experience assumed.

## What an evaluation is

An evaluation is a repeatable test for an agent. You give it a suite of **tasks**, let your agent answer each one, then run **scorers** that grade the answers. The result is a scorecard you can track as your agent changes.

It's like unit testing, with one difference: most agent answers don't have a single right value, so instead of one pass/fail you grade several properties — did it call the right tool? is the answer correct? did it stay under budget? — and watch how the aggregate moves between versions.

Four pieces, that's all: a **dataset**, your **agent**, one or more **scorers**, and a call to **`run_agent`**.

## Step 1 — your first run

Start as small as possible: two questions, an agent, one check. This calls a real model, so set a key first (`export OPENAI_API_KEY=...`); Step 5 below shows how to run with none.

!!! note "Install the `tracing` extra"
    `run_agent` produces each task's trace via OpenTelemetry, so it requires the tracing extra: `pip install "ag2[tracing]"`. (Grading already-captured traces with `evaluate_traces` does not need the extra.)

```python linenums="1"
import asyncio

from autogen.beta import Agent
from autogen.beta.config import OpenAIConfig
from autogen.beta.eval import Suite, run_agent
from autogen.beta.eval.scorers import final_answer_matches

# 1. the dataset — a couple of tasks, each with the expected answer
suite = Suite.from_list([
    {"task_id": "france", "inputs": {"input": "Capital of France?"}, "reference_outputs": {"answer": "Paris"{{ "}}" }},
    {"task_id": "japan", "inputs": {"input": "Capital of Japan?"}, "reference_outputs": {"answer": "Tokyo"{{ "}}" }},
])

# 2. the agent under test
agent = Agent("geographer", prompt="Answer with the capital city.", config=OpenAIConfig(model="gpt-4o-mini"))

async def main():
    # 3 + 4. score each answer against its expected value, and run
    result = await run_agent(
        suite,
        agent=agent,
        scorers=[final_answer_matches(field="answer", matcher="contains")],
        store_dir="./runs",
    )
    print(result.summary())

asyncio.run(main())
```

Run it and you get a scorecard:

```
Run a1b2c3d4
  Suite:       inline (2 tasks, source: inline)
  Runs:        2
Pass rates:
  final_answer_matches  100.0% (2/2)
```

That's a complete evaluation. The next steps unpack each piece.

## Step 2 — read the result

`run_agent` returns a `RunResult` and saves a JSON file under `store_dir`. A few accessors:

```python linenums="1"
result.summary()                            # the printable table above
result.pass_rate("final_answer_matches")    # 1.0 — the fraction of tasks that passed
```

Every scorer becomes a column, looked up by its **key** (here `"final_answer_matches"`). Boolean scorers give a pass-rate; numeric ones give `score_stats` (mean / p50 / p95); categorical ones give `value_counts`.

## Step 3 — the expected answer (`reference_outputs`)

Notice each task carried a `reference_outputs` — the **gold answer**, the thing you grade against:

```python
{"task_id": "france", "inputs": {"input": "Capital of France?"}, "reference_outputs": {"answer": "Paris"{{ "}}" }}
```

It's a small labelled record, not a bare string, so a scorer can pick out the field it cares about: `final_answer_matches(field="answer")` reads `reference_outputs["answer"]` and compares it to the agent's answer. `inputs` is what goes *in* (the prompt lives under `"input"`); `reference_outputs` is what *should* come out. Tasks graded purely from the trace ("did it call the tool?") don't need one.

!!! note "Matchers"
    `matcher="contains"` passes if the gold value appears anywhere in the answer — right for free-text replies like "The capital is Paris." Use `"casefold"` or `"exact"` when the answer should be exactly the value.

## Step 4 — ask more questions (scorers)

One check is rarely enough. A scorer is just a function that asks **one** question about a run. Mix the prebuilt ones with your own — a function decorated with `@scorer` that declares what it needs by name (`trace`, `outputs`, `reference_outputs`, …; here just `outputs`, the final answer):

```python linenums="1"
from autogen.beta.eval import scorer
from autogen.beta.eval.scorers import no_tool_errors, token_budget

@scorer
def answered_briefly(outputs) -> bool:
    return len(outputs["body"]) < 100      # outputs["body"] is the final answer text

scorers = [
    final_answer_matches(field="answer", matcher="contains"),
    no_tool_errors(),
    token_budget(2_000),
    answered_briefly,
]
```

Keep each scorer to a single, specific question — three small checks tell you *what* broke when one fails; one big check just says "something's wrong." See [Scorers](scorers.md) for the prebuilts (including the `agent_judge` LLM scorer) and the return-type rules.

## Step 5 — run it with no API key (for CI)

You don't want pre-merge CI calling a live model — slow, costly, flaky. Swap the model for a `TestConfig` cassette: a canned reply per task, so the run is deterministic and free.

```python linenums="1"
from autogen.beta.testing import TestConfig

# Same agent as Step 1 — a canned reply per task_id
canned = {"france": TestConfig("Paris"), "japan": TestConfig("Tokyo")}

result = await run_agent(suite, agent=agent, scorers=scorers, model_config=canned, store_dir="./runs")
```

One thing changed: `model_config` supplies the canned replies keyed by `task_id`. It's passed through to `ask`, which **overrides** the agent's own config for that task — so the same `agent` from Step 1 runs deterministically with no API key, no separate build needed. Same suite, same scorers, identical output on every machine — wrap `result.pass_rate(...)` in a `pytest` assertion and you have a CI gate.

## Where to next

You've seen the whole loop. From here:

- **[Scorers](scorers.md)** — the six prebuilts, writing custom scorers, and `agent_judge` for grading subjective quality.
- **[Runs](runs.md)** — datasets and the `RunResult` in depth, grading existing traces (`evaluate_traces`), and observing a run live.
- **[Variants](variants.md)** & **[Pairwise](pairwise.md)** — compare builds on a leaderboard, or head-to-head.
- **[Persistence & tracking](persistence.md)** — save runs and compare them over time to catch regressions and confirm improvements.
