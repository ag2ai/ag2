---
status: accepted
date: 2026-10-27
---

# 0027. Subagent events use direct forwarding with the original source

## Context

Subagents v2 needs ancestor observers to see a whole delegation subtree and to
respond to an event's author. Buffered forwarding would require queue limits,
flush rules and policies for failed or stalled observers. The owner explicitly
chose ordinary event delivery over that isolation machinery.

The addressing use case is responding to the author of an observed event.
A separate selector-based API would add recipient discovery, broadcast, and
branch-visibility rules beyond that use case, including ambiguity when several
delegations have the same agent name.

## Decision

Forward all source events upward directly, awaiting the parent stream's normal
delivery pipeline. Forwarding is unconditional across the whole delegation
tree, including transient chunks; there are no `forward_events` or
`forward_transient` switches. Subscribers filter what they consume. Preserve
the direct author's identity and live address through every hop.

Forwarded operational events remain observation envelopes rather than local
execution events. They do not enter the ancestor's model transcript or trigger
its tool, model or HITL handlers. A transient source event has a transient
envelope, so default storage does not start retaining chunks merely because
they travelled upward. Ordinary `persist_all` behavior remains available.

Forward `UsageEvent` records directly, without `ChildEvent`, preserving their
source identity and accounting fields. Existing `UsageEvent` subscriptions
must continue to receive all descendant spend without changing their filters.
Usage is non-conversational telemetry and does not activate operational
handlers. No replacement rollup is generated; see
[ADR 0029](0029-account-descendant-usage-from-source-events.md).

Task lifecycle events (every `TaskEvent`) are forwarded the same way, for the
same reason: they are non-conversational, and no tool, model loop, HITL hook or
middleware uses them as a trigger. Before v2 a typed lifecycle subscription on
the parent received its child's `TaskStarted` and terminal event; wrapping them
would have silently emptied every such subscription. The envelope stays the
rule for all events that could start work in the parent. Only these inert
events, which share the `InertEvent` base, are exempt.

An inert event carries its provenance itself. The first forwarding hop, run
by the delegation that published it, stamps `path` and the live `origin`, and
on `UsageEvent` also `task_id` and `agent_name`. Later hops keep the stamp and
only rebind the live handle to their own sender. `ConversationContext.send`
clears the stamp, because a local publication is a new event with no source
above it.

Downlink accepts any event through the recipient's normal pipeline. The caller
chooses the event and is responsible for its effects, including reentrant model
calls; the transport adds no event-type restrictions or scheduling barrier.

Expose one addressed-delivery entry point: `event.origin.send(event)`. The
origin handle addresses the original author's invocation, including through
multiple ancestor hops. Keep the registry internal for cancellation and
cleanup; expose neither a global routing function nor lookup or broadcast by
task ID, agent name, or subtree.

The implementation brackets forwarding and lifecycle under the active stream
turn lock, including terminal publication after observer cleanup.

## Consequences

Upward delivery uses the stream's ordinary ordering and exception behavior.
There is no separate queue, drop policy, timeout or stalled-observer recovery;
a slow ancestor observer can delay its descendants. An observer can address
the event's author directly without unwrapping intermediate agents' identities.
Matching agent names cannot broaden a handle's recipient set: it always
addresses one invocation. A queued invocation has no live handle, and a saved
handle expires when its invocation ends. Selector-based delivery would require
a separate use case and design decision.
Broad subscribers and unfiltered cadence watches now see the whole subtree;
authors use subscription conditions when they need only a subset. This is an
intentional change to the previous opt-in proposal and v1 event visibility.
