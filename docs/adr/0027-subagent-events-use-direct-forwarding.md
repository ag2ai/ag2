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

The implementation brackets forwarding and lifecycle under the active stream
turn lock, including terminal publication after observer cleanup.

## Consequences

Upward delivery uses the stream's ordinary ordering and exception behavior.
There is no separate queue, drop policy, timeout or stalled-observer recovery;
a slow ancestor observer can delay its descendants. An observer can address
the event's author directly without unwrapping intermediate agents' identities.
Broad subscribers and unfiltered cadence watches now see the whole subtree;
authors use subscription conditions when they need only a subset. This is an
intentional change to the previous opt-in proposal and v1 event visibility.
