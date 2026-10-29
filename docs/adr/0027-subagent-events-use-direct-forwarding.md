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

Forwarded events remain observation envelopes rather than local execution
events. They do not enter the ancestor's model transcript or trigger its tool,
model or HITL handlers. A transient source event has a transient envelope, so
default storage does not start retaining chunks merely because they travelled
upward. Ordinary `persist_all` behavior remains available.

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
