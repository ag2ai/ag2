---
status: accepted
date: 2026-06-29
---

# History management keys on `__conversational__`, not `__transient__`

Surfaced while tracing why `UsageEvent` (token-usage telemetry) was being counted
as a real turn by history compaction. The fix is one new marker; this ADR records
why a second axis was needed rather than reusing `__transient__`.

## Context

Two `BaseEvent` flags now describe an event, and they answer different questions:

- **`__transient__`** (existing) answers *"should this be persisted?"* Transient
  events (`ModelMessageChunk`, `TaskProgress`) are streaming/lifecycle artifacts
  superseded by a durable event, so the storage layer (`ag2/stream.py`) drops
  them. Everything else persists.

- **`__conversational__`** (new) answers *"is this real conversation that should
  drive history management?"* — i.e. count toward the compaction trigger, occupy
  a slot in the retained window, and be fed to the summarizer.

The bug was that these two questions had been **conflated onto one flag**.
`UsageEvent` is deliberately non-transient: it persists so `UsageReport` can read
token spend back from history. But compaction asked "is this work?" by reusing
the persistence flag — `[e for e in events if not __transient__]` — so every
persisted `UsageEvent` counted as a turn. Concretely:

- **Trigger drift.** Each LLM call emits one `UsageEvent`, so it was ~⅓ of
  "conversational" events — compaction fired meaningfully early on `max_events`
  and the `max_tokens` char estimate.
- **Window shrinkage.** `TailWindowCompact(target=N)` retains by raw position, so
  interleaved `UsageEvent`s pushed real turns out of the kept window.
- **Summary pollution.** `SummarizeCompact` stringified `UsageEvent`s into the
  summarization prompt.

Meanwhile the **aggregation** path had already diverged the other way — it counts
a hard-coded allowlist `(ModelRequest, ModelResponse, ToolCallsEvent,
ToolResultsEvent)` and explicitly excludes telemetry. Two filters, same question,
two answers — the classic drift that an inspection-only equivalence invites.

## Decision

**Add `__conversational__` as a second, orthogonal axis, and route every
history-management decision through one predicate.**

- `BaseEvent.__conversational__ = True` by default; telemetry sets it `False`.
  `UsageEvent.__conversational__ = False`.
- `is_conversational(event)` (`ag2/events/base.py`, exported from `ag2.events`)
  is the single predicate: `not __transient__ and __conversational__`. Both the
  compaction trigger and the retention budget call it, so "what counts as
  conversation" has one source of truth and cannot drift again.
- **Retention budgets by conversational count, but does not delete telemetry.**
  `_cut_for_target` (`ag2/compact.py`) finds the cut so the retained span holds
  the last `target` *conversational* events; interleaved telemetry rides along in
  that span for free. `TailWindowCompact(target=N)` again means N real turns.
- The summarizer prompt filters non-conversational events out of its input only;
  dropped telemetry is still persisted to the store like any other dropped event.

## Consequences / things that look wrong but are deliberate

- **Two near-synonym flags coexist on purpose.** `__transient__` and
  `__conversational__` look redundant but are independent axes. The combination
  that motivated the split — *persisted yet not conversational* (`UsageEvent`) —
  has no single-flag encoding. `is_conversational` ANDs both because a transient
  artifact is never conversation either.

- **Aggregation keeps its narrow allowlist; it is not migrated to the marker.**
  Aggregation's `every_n_events` counts a 4-type "turn of work" set, which is
  *narrower* than "conversational" (it excludes `HumanMessage`,
  `CompactionSummary`, …). Switching it to `is_conversational` would silently
  broaden its cadence. Both paths are now robust to *new* telemetry — compaction
  via the marker, aggregation via its allowlist — which is the drift the original
  bug was about; unifying the conversational-but-not-work distinction was not
  required to fix it.

- **Retained `UsageEvent`s stay in live history by design.** The tempting fix —
  strip telemetry from the list handed to `compact()` — is unsafe: compaction's
  result replaces history (`history.replace(compacted)`), so stripping would
  delete the `UsageEvent`s and (because `UsageReport` reads live history, not the
  dropped-event store) break usage accounting for the kept window. Keeping them in
  the retained span preserves that.

- **Usage for the *dropped* span is still lost across compaction — unchanged and
  out of scope.** `agent.usage()` reads live history only, so events compacted
  away (telemetry included) leave the report. This pre-existed the change; this
  ADR neither fixes nor worsens it. Closing it is a separate decision: document
  `usage()` as a live-window report, or have `UsageReport` also read the persisted
  dropped log. The marker design does not depend on which is chosen.
