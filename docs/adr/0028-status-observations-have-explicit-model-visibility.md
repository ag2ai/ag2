---
status: accepted
date: 2026-10-30
---

# Retained status observations have explicit model visibility

## Context

Status callbacks need both request-only observations and a retained progress
trail. Publishing retained status as ModelRequest would recursively call the
model and manufacture user turns. Marking it conversational just to expose it
to providers would change compaction triggers, retention counts, and summaries,
contradicting ADR 0010. Every provider already converts normal Input parts.

## Decision

Persist nonempty callback results as StatusObservation events through
Context.send (ADR 0004). The record contains copied Input parts and callback
identity, is non-transient and non-conversational, and is distinct from
ModelRequest. Equal observations are separate records; skipping adds none.

Before the middleware chain, explicitly project retained observations to
detached, transient ModelRequest views with service origin and non-conversational
classification. Fresh observations use the same projection after publication,
without rereading history or losing earlier wrapper insertions. This reuses
every provider's normal Input handling and ConversationPolicy's existing request
support, including after an invocation plugin leaves its scope (ADR 0022).
Provider replay grouping resets only on conversational requests, so a projected
service view cannot hide a missing native anchor or turn item. Request-only
(non-persistent) status uses the same service view rather than a plain user
request.

Non-conversational is not the same as weightless. Token estimates count status
like any other model input, so token limiters and the compaction token trigger
see the retained trail that is resent on every call; event counts still exclude
it. A history reducer never treats a span left with only service views as a
turn: it widens past them, so status appended after a tool round cannot orphan
and drop that round.

## Consequences

Storage and callbacks retain service identity; model requests use ordinary
provider content without a new role or provider-specific event mapping. Both
source data and provider views are copied so later mutations do not rewrite
saved observations. Existing telemetry, conversational counts, and summaries
keep their classification; ordinary reducers may remove observations.

Persistence precedes the wrapped model call. Once committed, an observation
survives later failure or cancellation. Status outside retry creates one snapshot
for its inner attempts. Persistent status inside retry publishes before every
attempt, so retry's publication guard refuses to repeat any failed attempt;
that order is documented as unsupported rather than special-cased in retry.
This provides no cross-backend transaction or crash-exactly-once claim.
