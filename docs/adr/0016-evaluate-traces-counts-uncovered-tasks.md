---
status: accepted
date: 2026-09-02
---

# `evaluate_traces` counts a suite task with no trace, rather than dropping it

## Context

`evaluate_traces(source, suite=…)` graded whatever the source listed: one
`TaskResult` per `TraceRef`, joined back to a `Suite` task by `task_id` for
reference-based scorers. The join was one-directional. A trace with no matching
task was tolerated — graded reference-free against a synthesized `Task` — but a
*task* with no matching trace contributed nothing at all.

`Aggregates` is computed over the graded tasks, so an uncovered task shrank the
pass-rate denominator instead of registering a failure. The failure mode is
backwards for the thing evals exist to do: an agent that crashed before emitting
any span, or a run whose export never landed, made the suite score *higher*. A CI
gate on `pass_rate(...) >= 0.9` reads green precisely when the run went worst.

The producing side already had the right shape. `run_agent` catches a task that
raised and records it as `Trace(events=(), exception=…)` (`_error_trace_pair`),
so the task is graded and counted as an error. Only the trace-source path could
lose a task silently, and it does so for every `TraceSource` implementation.

## Decision

`evaluate_traces` takes a keyword-only `on_missing_task: Literal["error",
"ignore", "raise"]`, applied to every suite task no `TraceRef` carries the
`task_id` of. `"raise"` aborts before grading anything; `"ignore"` is the old
behaviour plus a warning naming the uncovered ids; `"error"` — the default —
materializes the task as a `TaskResult` whose trace is
`Trace(events=(), exception=MissingTraceError(task_id), duration_ms=0)`, graded
through the same scorer path as any other trace, so it lands in
`aggregates.errors` and in the pass-rate denominator. It mirrors what
`run_agent` already does for a task that raised.

The asymmetry stays deliberate in the other direction: a trace whose `task_id`
is not in the suite is still graded reference-free. A source may legitimately
carry more than the suite describes (captured production traffic); a suite
describes exactly what was meant to run.

## Consequences

- **This changes existing numbers.** A run where the source did not cover the
  whole suite now reports more tasks, more errors, and usually a lower pass rate.
  That is the point — but a `RunResult` graded before this change is not
  comparable with one graded after if any task was uncovered, and `diff()` will
  see tasks that "appeared". Nothing in the docs, tests, or an earlier ADR
  promised that uncovered tasks were skipped, so no behaviour contract is being
  broken; callers who want the old numbers pass `on_missing_task="ignore"`.
- Ordering is the simplest deterministic rule: graded traces first in source
  order, then missing tasks in suite order. Callers keying off positional index
  should key off `task_id`.
- A scorer now sees an empty `Trace` for such a task. That is not a new shape —
  it is exactly what `run_agent` hands a scorer when `ask` raised — so scorers
  need no change, and one that raises on it is captured as
  `Feedback(score=None)` as always.
- `MissingTraceError` is public (`ag2.eval`), so a caller can distinguish "no
  trace" from a real agent failure recorded on the trace.
- `run_agent` is unaffected in practice: it produces one trace per task,
  including for tasks that raised, so its suite is always fully covered.
