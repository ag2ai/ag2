---
status: accepted
date: 2026-10-30
---

# Model request origin defines user-input history boundaries

## Context

An ordinary public Ask or Run and a framework inbox update both reach providers
as user content. `DrainedModelRequest` prevents recursive model calls during a
tool round, but the final inbox drain deliberately emits a regular ModelRequest
to continue execution. Neither provider role nor that execution marker reliably
identifies actual user input. Per-input history queries need a durable boundary
that remains meaningful when events are restored from storage.

## Decision

Persist `ModelRequest.origin`, with values `user` and `service`. Public invocation
requests retain `user`. Inbox messages carry the origin of their sender:
`AgentRun.enqueue` is a person steering the running turn, so it queues `user`;
`Context.enqueue` from tools, background tasks and other framework code keeps
the default `service`. An inbox drain, in either form, is `user` when any drained
message is, otherwise `service`. Merging old pending input into a new public
request preserves the new request's origin. Existing ordinary ModelRequest
records without origin default to `user`; existing DrainedModelRequest records
default to `service`.

Origin follows the sender rather than the delivery channel because the query
promises the latest actual user input. A channel-based rule would make a
mid-turn steer invisible to the boundary.

`History.since_last_user_input()` finds the latest retained request whose origin
is `user`, excludes it, then applies selection conditions. No retained boundary
means an empty interval. Origin does not change model dispatch, provider wire
roles, conversational classification, or content equality.

## Consequences

Storage preserves the distinction without caller-supplied turn IDs or inferred
timestamps. `DrainedModelRequest` keeps its separate execution purpose. Legacy
ordinary requests remain useful boundaries; old final inbox drains that were
stored without provenance cannot be distinguished retrospectively. Queries do
not attempt to recover boundaries removed by retention or compaction.
