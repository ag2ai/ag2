---
status: accepted
date: 2026-08-15
---

# Anything that adds up tokens reads `UsageEvent`

> Subagents v2 update (2026-10-29):
> [ADR 0029](0029-account-descendant-usage-from-source-events.md) supersedes the
> original rollup mechanism with direct consumption of source descendant events.
> The decision below reflects that implementation. Rollup matching is retained
> only for reading archived traces.

Surfaced while tracing why a coordinator that delegates one sub-task reported 110 of the
1100 tokens it actually spent. Three consumers of token accounting read three different
events, and only one of them was right. This ADR records the rule that keeps them from
drifting apart again.

## Context

`UsageEvent` is the framework's accounting record: one event per unit of billable
work, emitted when tokens are spent by a main-loop LLM call, a live session,
history compaction, or memory aggregation. Descendants' original records now
reach ancestors as plain `UsageEvent` events. The original delegation
implementation emitted a sub-agent rollup instead; new runs no longer do.

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

**Every consumer that accumulates tokens reads source `UsageEvent` records.
Snapshot fields — `AgentReply.usage`,
`TaskResult.usage`, `TaskCompleted.usage` — are for direct inspection by a caller who holds
that one object, and must never be summed.**

- Consumers subscribe to `UsageEvent` for both local and descendant spend.
  This preserves existing type-based observer and stream subscriptions;
  there is no `ChildEvent` envelope to unwrap or extra condition to enable.
  Accounting arrives before `TaskCompleted` and does not execute the
  descendant's operational handlers in the parent.
- Deduplicate by `UsageEvent.event_id`, the identity of one source publication.
  Preserve that identity through forwarding, persistence, and usage spans.
  Repeated observations of one record count once; equal-cost calls with
  different identities remain separate costs. A `DownEvent` audit is its own
  source; an accounting-shaped payload is not counted from that audit.
- `TelemetryMiddleware` records each source record as a `record_usage` span,
  preserving its identity, kind, provider, and model. This captures delegated
  and maintenance spend without generating replacement rollups.
- `Trace.tokens` aggregates `UsageEvent` through `UsageReport`, so eval and `agent.usage()`
  report the same number by construction rather than by coincidence.
- Accounting history survives failure, cancellation, and compaction. Reports
  read retained source records rather than reconstructing spend from terminal
  snapshots. Budget monitors see descendant spend as the source events arrive.
- The original accounting correction changed the eval results schema from
  `0.1` to `0.2`. A run loaded from disk keeps the version it was written with,
  so a reader can tell which accounting produced its numbers.

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

- **Archived rollup traces can contain the same tokens twice.** In the original
  delegation implementation, when a
  sub-agent is itself instrumented, its per-call accounting flattens into the *same* trace as
  the parent's `"subtask"` rollup. `_nested_agent_spend` totals the spend recorded under each
  nested agent subtree and `_drop_duplicated_rollups` cancels a matching rollup — as a pass over
  the reconstructed events, because whether a rollup duplicates is a fact about the whole span
  tree and threading it through the per-span readers made them stateful and single-use.
  Matching is by **value *and* name**. Value alone assumed every nested instrumented agent is
  also covered by a rollup on its parent — true of `run_task`/`as_tool`, but not of a plain
  `await other.ask(...)` from inside a tool, which produces usage spans and no rollup at all.
  That agent's spend then had no rollup of its own to cancel and cancelled an unrelated
  worker's of equal value, losing those tokens outright. The name is read off the sub-agent's
  own `invoke_agent` span, where `TelemetryMiddleware` always writes `gen_ai.agent.name`, and
  compared against the rollup's `ag2.usage.label`. Spend whose agent span says `"unknown"`
  (the caller named no agent) identifies nobody and falls back to value-only matching. Each
  entry cancels at most one rollup, so the residual ambiguity is bounded: two *unnamed*
  workers with identical spend, one instrumented and one not, can still cancel the wrong one —
  but the total stays correct and only the label is misattributed. This matching
  remains for archived rollup records; current runs preserve source identity
  instead and never generate a parent `"subtask"` rollup.
- **The reader adapts to the trace, not to who asked.** Both trace-level adjustments — rollup
  dedupe, and synthesizing usage for a trace captured before AG2 recorded usage spans — apply
  whoever supplied the conventions. Choosing a reader says what a *span* means; it does not
  make the same delegation bill twice, nor move where an archived trace keeps its counts.
  Gating them on `conventions is None` meant `conventions=DEFAULT_CONVENTIONS` — which reads
  as a no-op, and is the natural way to spell "the defaults plus mine" — double-counted every
  instrumented sub-agent and reported zero for every archived trace. A caller-supplied
  `AG2GenAIConvention` is therefore re-created with the synthesis setting the trace calls for;
  a subclass or a foreign reader is passed through untouched.
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
- **In-process accounting is not a network usage protocol.**
  `ag2/network/task_mirror.py` consumes lifecycle observations and forwards task
  metadata, not source accounting records. Direct descendant accounting in
  [ADR 0029](0029-account-descendant-usage-from-source-events.md) does not imply
  that a remote task mirror transports its complete usage history.
