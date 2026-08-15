---
status: accepted
date: 2026-08-15
---

# Anything that adds up tokens reads `UsageEvent`

Surfaced while tracing why a coordinator that delegates one sub-task reported 110 of the
1100 tokens it actually spent. Three consumers of token accounting read three different
events, and only one of them was right. This ADR records the rule that keeps them from
drifting apart again.

## Context

`UsageEvent` is the framework's accounting record: one event per unit of billable work,
emitted onto the stream at the point the tokens are spent — a main-loop LLM call, a live
session, history compaction, memory aggregation, and a sub-agent rollup. Its docstring
already said so, and `UsageReport.from_events` already read it and nothing else.

The rule lived only in that docstring, so the other two consumers each invented their own
source:

- **`TokenMonitor`** (the budget guard) accumulated `ModelResponse.usage` and
  `TaskCompleted.usage` with `+=`. Three defects, all the same defect: a sub-task that
  billed and *then* failed moved the guard by zero, because only `TaskCompleted` carries a
  usage field; repeated delegations to a worker on a reused stream read 660 instead of 330,
  because `TaskCompleted.usage` is a **cumulative snapshot** of that stream and adding
  snapshots re-counts everything before them; and compaction, aggregation and the live
  clients produce no `ModelResponse` at all, so their spend was never counted.
- **`Trace.tokens`** (eval) summed `ModelResponse` off reconstructed spans. Delegated spend
  never became an LLM span in the parent's trace, and maintenance calls run outside the
  middleware hooks so they can never produce one.

Both failures are the same mistake: reading a field that answers *"what has this stream
spent so far?"* to answer *"what did this step just spend?"*, or reading an artifact that
happens to accompany most billable work instead of the record that accompanies all of it.

## Decision

**Every consumer that accumulates tokens reads `UsageEvent`. Snapshot fields — `AgentReply.usage`,
`TaskResult.usage`, `TaskCompleted.usage` — are for direct inspection by a caller who holds
that one object, and must never be summed.**

- `TokenMonitor` watches `UsageEvent` instead of `ModelResponse | TaskCompleted`. All three
  defects above close together, with no new field on `TaskFailed` and no change to the
  sub-task runner — the rollup was already emitted on both the success and the failure path,
  before the terminal lifecycle event.
- `TelemetryMiddleware` subscribes to `UsageEvent` and records each one as a `record_usage`
  span. This is the only route by which delegated and maintenance spend reaches a trace at
  all; no downstream change can recover data that was never captured.
- `Trace.tokens` aggregates `UsageEvent` through `UsageReport`, so eval and `agent.usage()`
  report the same number by construction rather than by coincidence.
- The eval results schema goes `0.1` → `0.2`. A run loaded from disk keeps the version it was
  written with, so a reader can tell which accounting produced its numbers.

Non-obvious choices, each of which looks like a bug until you know why:

- **The telemetry usage watcher deliberately outlives the turn.** Middleware that reports
  usage does so *after* its own `call_next` returns — compaction summarises what the finished
  turn produced — and agent-level middleware wraps middleware passed to `ask`, which is how
  the eval runner installs telemetry. Scoping the subscription to the turn dropped exactly
  the maintenance spend it exists to capture. Exactly-once comes from *replacing* the
  per-stream watcher in a process-wide registry, not from unsubscribing; that registry
  mirrors the per-stream turn-lock registry in `ag2/agent.py`, and turns on a shared stream
  are serialised by that lock.
- **Usage spans are parented explicitly at the turn span**, not at the ambient context. A
  late-arriving event fired after the turn span closed would otherwise start a *new trace*,
  and a backend grouping by trace id would lose the spend entirely.
- **The AG2 span convention must never synthesize usage from LLM spans.** A main-loop call
  produces both a usage span and an LLM span, so reading both double-counts every direct
  call. Synthesis is switched on only for traces containing no usage span at all — archived
  exports — and for foreign dialects (OpenInference), which have no accounting event of
  their own and would otherwise report zero for people who changed nothing.
- **The guard refuses a reported total below prompt plus completion.** Summed usage adds
  `total_tokens` field-wise, so one call from a provider that omits it drags the sum under
  its own parts. Taking the larger of the two keeps a partial total from understating the
  budget, while still believing a provider whose total legitimately exceeds the two counts.
  The fallback lives in the observer, not on `Usage`: whether a synthesized total is honest
  is a question about the shared value type.

## Consequences

- **A span tree can contain the same tokens twice, and the reader has to cancel one.** When a
  sub-agent is itself instrumented, its per-call accounting flattens into the *same* trace as
  the parent's `"subtask"` rollup. `_nested_agent_spend` totals the spend recorded under each
  nested agent subtree and `_drop_duplicated_rollups` cancels a matching rollup — as a pass over
  the reconstructed events, because whether a rollup duplicates is a fact about the whole span
  tree and threading it through the per-span readers made them stateful and single-use.
  Matching is by **value, not by name** —
  `gen_ai.agent.name` is optional and defaults to `"unknown"`, while the rollup's label is the
  real agent name, so the two cannot be compared; the rollup is by construction the sum of
  exactly those events, which makes value equality the reliable signal. Known limit: two
  workers with identical spend, one instrumented and one not, could cancel the wrong rollup.
  Each entry cancels at most one rollup, so the error is bounded.
- **`0.1` and `0.2` token counts are not comparable.** A baseline recorded before this change
  understates delegated and maintenance spend. The version is carried through `load_run`
  rather than restamped so that a regression against an old baseline is legible as a schema
  change and not read as a real jump in cost.
- **Three near-synonymous questions now describe usage data** — is it persisted, is it
  conversation ([ADR 0010](0010-history-management-keys-on-conversational-not-transient.md)),
  and is it the additive record. The natural reaction to that is to collapse two of them; ADR
  0010 makes the same argument about its own pair. The combination that forces this third
  apart is a snapshot field that is genuinely useful to a direct caller and genuinely wrong
  to a consumer that accumulates.
- **A new consumer of token counts has one place to look.** Adding a field to a lifecycle
  event to expose spend is the move this ADR exists to prevent — the event is already on the
  stream before it.
- **`ag2/network/task_mirror.py` still drops usage**: it forwards state, result and error
  across the network boundary on both terminal events, so a mirrored sub-task's rollup never
  reaches the parent. Pre-existing, out of scope here, and genuinely separate work.
