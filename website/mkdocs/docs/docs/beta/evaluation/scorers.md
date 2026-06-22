---
title: "Scorers"
sidebarTitle: "Scorers"
---

A scorer is a function that grades one agent run and produces a structured feedback record. Run it across N tasks and you get N feedback records — which the framework aggregates into pass rates, score distributions, and label counts.

## Writing a scorer

Decorate a function with `#!python @scorer`. The framework calls it once per task; the return value becomes feedback.

```python linenums="1" hl_lines="4-6"
from autogen.beta.events import ToolCallEvent
from autogen.beta.eval import scorer

@scorer
def called_get_weather(trace) -> bool:
    return len(trace.events_of(ToolCallEvent, name="get_weather")) == 1
```

That's it. Three things to notice:

1. **It takes only what it needs.** The decorator inspects the signature and injects only the parameters you declared.
2. **It returns a `#!python bool`.** The framework turns that into a pass rate.
3. **It's pure.** No I/O, no global state. Scorers run concurrently across tasks; impure scorers race.

## What you can ask for

A scorer can declare any subset of these five parameters, by name:

| Parameter | Type | What it is |
|---|---|---|
| `inputs` | `#!python dict[str, Any]` | The task's input payload — typically `{"input": "the user's prompt"}`. |
| `outputs` | `#!python dict[str, Any]` | The agent's final answer projected from the trace, mirroring the reply API: `{"body": <final text>, "content": <typed answer>}`. `body` is the text (like `#!python reply.body`); `content` is the parsed value when the answer is JSON — e.g. a `response_schema` agent — otherwise the text (like `#!python await reply.content()`). Read structured fields via `#!python outputs["content"]["answer"]`. |
| `reference_outputs` | `#!python dict[str, Any] \| None` | The task's expected output, if the dataset provided one. |
| `trace` | `Trace` | The typed events (model responses, tool calls / results, …) plus tokens, duration, and exception. |
| `task` | `Task` | The task record (id, tags, metadata). |

Declare only what you use:

```python linenums="1"
@scorer
def called_get_weather(trace) -> bool: ...                 # reference-free

@scorer
def city_argument_correct(trace, reference_outputs) -> bool: ...  # reference-based

@scorer
def answer_mentions_city(outputs, reference_outputs) -> bool: ... # output-only
```

## Three return shapes → three aggregation behaviors

```python linenums="1"
@scorer
def called_get_weather(trace) -> bool: ...        # → pass_rate

@scorer
def extra_tool_calls(trace) -> int: ...           # → score_stats (mean / p50 / p95)

@scorer
def termination_reason(trace) -> str: ...         # → value_counts
```

The framework routes by return type:

- **`#!python bool`** lands in `result.pass_rate("scorer_name")` as `passes / total`.
- **`#!python int` / `#!python float`** lands in `result.score_stats("scorer_name")` as `ScoreStats(mean, p50, p95, n)`.
- **`#!python str`** lands in `result.value_counts("scorer_name")` as `{label: count}`. Useful for slicing — "of 100 runs, 95 completed, 5 errored".
- **`#!python None`** is treated as "skip — no feedback recorded for this task".
- A `Feedback` instance or `#!python list[Feedback]` lets you set the key explicitly, attach a comment, or emit multiple records from one call.

!!! note
    `#!python bool` is a subclass of `#!python int` in Python, so the framework checks `#!python isinstance(value, bool)` first — `#!python True` always becomes a pass-rate feedback, never a numeric one. On the flip side, returning `#!python 1` / `#!python 0` (an `#!python int`) is a *numeric* score and lands in `score_stats`, **not** `pass_rate` — return `#!python True` / `#!python False` for pass/fail.

## Reference-based vs reference-free

A load-bearing distinction:

- **Reference-based** scorers compare what happened to what *should have* happened. They need `reference_outputs` from a labelled dataset — so they can't grade arbitrary production traffic that has no gold answer.
- **Reference-free** scorers judge from the trace alone — so the same code grades `run_agent` traces, stored traces, and live production traces (via `evaluate_traces`).

Write scorers reference-free whenever you can — the same code then runs everywhere.

!!! note "Guard reference-based scorers against missing references"
    A task without `reference_outputs` injects `reference_outputs=None`. Guard for it — `#!python if reference_outputs is None: return None` to skip the task, or `#!python return False` to count it as a fail. An unguarded `#!python reference_outputs["city"]` raises on `None`, which the framework records as `#!python score=None` (the run continues, but the task neither passes nor counts).

## One question per scorer

Don't bundle. A god-scorer like

```python linenums="1"
@scorer
def everything_is_fine(trace, outputs, reference_outputs) -> bool:
    return (
        len(trace.events_of(ToolCallEvent, name="get_weather")) == 1
        and reference_outputs["city"] in (outputs.get("body") or "")
        and trace.tokens.total < 2000
    )
```

…tells you nothing when it fails. Three scorers, one each, give you three signals you can trace.

```python linenums="1"
@scorer
def called_get_weather_once(trace) -> bool:
    return len(trace.events_of(ToolCallEvent, name="get_weather")) == 1

@scorer
def answer_mentions_city(outputs, reference_outputs) -> bool:
    return reference_outputs["city"] in (outputs.get("body") or "")

@scorer
def under_token_budget(trace) -> bool:
    return trace.tokens.total < 2000
```

## Prebuilt scorers

Six ship under `autogen.beta.eval.scorers`. Four are simple, deterministic checks (below); two richer ones — `agent_judge` and `failure_attribution` — get their own sections after.

| Scorer | Question | Type | When to use |
|---|---|---|---|
| `tool_called(name, *, exactly=None)` | Did the agent call this tool? | bool | Most tool-use scenarios. `exactly=N` for strict count. |
| `no_tool_errors()` | Were there zero `ToolErrorEvent`s? | bool | Catch tools that exploded. |
| `final_answer_matches(field, matcher)` | Does the answer match `reference_outputs[field]`? | bool | Closed-form correctness. Matcher: `"exact"`, `"casefold"`, `"contains"`. |
| `token_budget(max_tokens)` | Did the run stay under `max_tokens` total? | bool | Cost discipline as a pass/fail signal. |

Each is a *factory* — calling it returns a `Scorer`. Drop them straight into the `scorers=` list:

```python linenums="1" hl_lines="8-13"
from autogen.beta.eval.scorers import (
    final_answer_matches,
    no_tool_errors,
    token_budget,
    tool_called,
)

scorers = [
    tool_called("get_weather"),
    no_tool_errors(),
    final_answer_matches(field="city", matcher="contains"),
    token_budget(2_000),
]
```

!!! tip
    Two distinct `tool_called(...)` calls produce distinct keys (`tool_called[get_weather]` vs `tool_called[get_news]`), so multiple instances coexist in one run.

## Agent-as-a-judge — `agent_judge`

Some properties can't be checked with `==`: *is the answer helpful? well-reasoned? on-brand?* `agent_judge` hands the answer plus a criterion you write to a judge model, which returns a numeric score. This is a more capable take on a LLM-as-a-judge evaluation.

```python linenums="1"
from autogen.beta.config import OpenAIConfig
from autogen.beta.eval.scorers import agent_judge

scorers = [
    agent_judge(OpenAIConfig(model="gpt-4o-mini"), criterion="The answer resolves the user's request.", key="helpfulness"),
    agent_judge(OpenAIConfig(model="gpt-4o-mini"), criterion="The answer is concise.", key="conciseness"),
]
```

One judge = one criterion = one column. The score lands in `result.score_stats(key)`, so a *list* of judges is a multi-dimension **scorecard** — each criterion scored and aggregated independently. The numeric range defaults to `(0.0, 1.0)` and is **enforced** (out-of-range scores are clamped); pass `scale=(1, 5)` for a Likert range. The judge is an ordinary `Agent`, so it can be made deterministic using a `TestConfig` in CI.

By default the judge sees the task's gold answer (rendered as a `## Reference` section whenever `reference_outputs` is present) — correct for a *correctness* judge, where matching the reference is the point. But for dimensions that must grade the answer **on its own** — *faithfulness*, *grounding* — leaking the gold answer lets the judge reward an answer for matching it rather than for being grounded in the agent's tool results. Pass `include_reference=False` to withhold the reference from those judges:

```python linenums="1"
scorers = [
    agent_judge(config, criterion="The answer matches the reference.", key="correctness"),
    agent_judge(config, criterion="Every claim is grounded in the tool results.",
                key="faithfulness", include_reference=False),
]
```

## Pass / fail thresholds — `threshold`

A judge returns a *number*, but for automation you often want a hard verdict — *reject anything below 0.7*. Pass `threshold=` to gate the judge: its column then lands in `result.pass_rate(key)` (pass iff `score >= threshold`) instead of `score_stats`, and the raw number is recorded in the feedback's `detail`.

```python linenums="1"
agent_judge(OpenAIConfig(model="gpt-4o-mini"), criterion="The answer is helpful.", key="helpfulness", threshold=0.7)
```

`threshold=0.7` is shorthand for wrapping the judge in the generic `threshold(...)` combinator, which gates **any** numeric scorer (a judge, or your own `@scorer`) into a Pass/Fail — `scorer → scorer`:

```python linenums="1"
from autogen.beta.eval.scorers import agent_judge, threshold

quality = agent_judge(config, criterion="...", key="quality")   # numeric scorer
gate = threshold(quality, at_least=0.7)                          # → Pass/Fail scorer
```

A gated criterion emits one `Feedback`: `score` is the boolean (so it feeds `pass_rate` and shows up in `result.diff(baseline).regressions` when it flips), and the number + bounds are kept in `detail`. A scorer that produces no numeric grade — a judge with no verdict, or one that raised — counts as a **fail**. Bounds are inclusive; use `at_most` (or both) for "lower is better" metrics. In CI: `assert result.pass_rate("helpfulness") == 1.0`.

!!! note
    Gating is opt-in: without `threshold=`, `agent_judge` keeps its numeric `score_stats` behavior. For a per-task token/time *resource* gate (a different axis), see [`BudgetThresholds`](runs.md).

## Failure attribution — `failure_attribution`

When a task fails, *why?* `failure_attribution` labels each run with a failure mode (a `str`, so it rolls up in `result.value_counts(key)`):

```python linenums="1"
from autogen.beta.eval.scorers import failure_attribution

scorers = [failure_attribution(key="failure_mode")]
# result.value_counts("failure_mode")  ->  {"none": 41, "tool_failure": 6, "crash": 3}
```

Out of the box it runs **deterministic detectors** (a crash/exception, no final answer, a tool error) — no model needed. Pass a `config` to add an **LLM attributor** that classifies the subtler, semantic failures the detectors can't see (wrong answer, gave up, looped):

```python linenums="1"
failure_attribution(OpenAIConfig(model="gpt-4o-mini"), key="failure_mode")
```

## Exception handling

A scorer that raises does NOT fail the run. The framework catches the exception, records a `Feedback` with `#!python score=None` and a comment explaining what blew up, logs a warning, and moves on:

```python linenums="1"
@scorer
def fragile(outputs) -> bool:
    return outputs["body"].startswith("Tokyo")  # raises if there's no final answer
```

If the trace had no final answer (`outputs` has no `"body"`) this raises `KeyError`. The feedback for that task becomes:

```python
Feedback(key="fragile", score=None, comment="scorer raised: KeyError: ...")
```

This is by design — one broken scorer should never kill an eval over 100 tasks. But it also means you should treat `#!python score=None` as a signal worth investigating, not noise.

## Tips for good scorers

- **Be specific.** "Did the agent call `get_weather`?" beats "Was the output good?"
- **Be cheap.** Scorers run for every task. No API calls inside a scorer.
- **Prefer structured fields over text.** `trace.events_of(ToolCallEvent, name="get_weather")` beats regex-matching `outputs["body"]`.
- **Don't share state.** Two scorers grade independently. No mutable globals.
- **Return `#!python None` when you can't grade.** A scorer that doesn't apply to a task should skip rather than fail.

## Sync vs async

Either works. The framework detects async with `#!python inspect.iscoroutinefunction` and awaits accordingly:

```python linenums="1"
@scorer
async def llm_judge(trace, reference_outputs) -> bool:
    # ... call a judge model ...
    return verdict
```

Most scorers consume an in-memory `Trace` and stay sync; async is there for scorers that call out — `agent_judge`, for instance, runs its judge model under the hood.

## Custom keys with `Scorer` directly

If you want a key independent of the function name, construct a `Scorer` yourself:

```python linenums="1"
from autogen.beta.eval import Scorer, Trace

def _check(trace: Trace) -> bool:
    return len(trace.events_of(ToolCallEvent, name="get_weather")) == 1

my_scorer = Scorer(_check, key="weather-tool-used")
```

The prebuilts use this pattern internally — `tool_called("get_weather")` returns a `Scorer` with `key="tool_called[get_weather]"`.

## Where to next

- **[Runs](runs.md)** — the `Suite` API, the `run_agent()` signature, `RunResult` aggregation methods, and the persistence format.
