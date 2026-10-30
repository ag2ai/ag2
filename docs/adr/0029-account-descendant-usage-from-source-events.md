---
status: accepted
date: 2026-10-27
---

# 0029. Account for descendant usage from source events

## Context

Delegation currently replaces descendant accounting events with one subtask
rollup at each boundary. This delays ancestor budget monitoring until the child
finishes, hides mixed-model breakdowns and requires special handling on failure
and cancellation. Subagents v2 already makes descendant events available above.

## Decision

Remove generation of subtask rollups. Accounting consumers read original
`UsageEvent` records from both the local stream and descendants as those
records arrive. Forward these records as plain `UsageEvent` events, without
`ChildEvent`, so existing type-based observer and stream subscriptions keep
receiving descendant spend. Do not synthesize replacement accounting events
or sum snapshot fields on responses and task terminals.

One source publication has one stable accounting identity, preserved through
forwarding, persistence and usage spans. Count that publication once within a
report's scope, regardless of how many representations it has. Equal usage
values from distinct publications remain distinct costs.
Forwarding preserves that identity through the stream primitive; it does not
call the ordinary context publication method that allocates a fresh usage ID.

Preserve the original usage kind, provider and model, and record which
delegation spent the tokens. A forwarded `UsageEvent` carries serializable
`task_id`, `agent_name` and `path` fields, stamped once by the delegation that
published it and kept by every later hop (see ADR 0027). A local record leaves
them empty. These are new fields rather than a reuse of `label`: `label` is the
agent name on archived `kind="subtask"` rollups, and giving it a second meaning
would make an archived rollup and a source record look alike to a reader that
groups by it. The live `origin` handle is not serialized, so without these
fields stored history could not tell descendant spend from the root's own.
Ancestor monitors consume
the same records without waiting for the delegated run's terminal event.
Persist descendant accounting records or durable references needed to recover
them, and retain that information during compaction. Reports must remain
complete after a run ends and after history is compacted.

Use a shared accounting-event extraction path for reports, monitors, telemetry,
UI collectors and retention. Trace readers deduplicate by source identity,
rather than comparing rollup amounts or agent names.

This supersedes the rollup-specific mechanism described in ADR 0014.
ADR 0014's rule that token accumulators read `UsageEvent` rather than snapshot
fields remains valid.

All descendant events are forwarded unconditionally, including Usage, per
ADR 0027. Accounting has no separate delivery switch or opt-in requirement.

## Consequences

Ancestor totals become the sum of original local and descendant records.
There are no newly generated `kind="subtask"` records or `parts` summaries;
breakdowns retain `model_call`, `compaction`, `aggregation` and their original
provider/model attribution. Existing snapshots are inspection results, never
additional additive inputs.

Forwarding alone is insufficient: durable accounting history, compaction
retention and trace identity handling must change together. Readers continue
to accept archived observation envelopes without requiring live subscriptions
to match them.
Archived traces containing old rollups remain readable through the existing
legacy reader. This does not reintroduce runtime generation of rollups.
