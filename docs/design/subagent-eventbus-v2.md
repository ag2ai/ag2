# Subagent event routing and variables

Each delegation runs on a separate stream. Source operational events travel
directly to ancestors in a `ChildEvent`, including streaming chunks. Accounting
records travel as plain `UsageEvent` events, preserving existing type-based
subscriptions. There are no
`forward_events` or `forward_transient` flags. Subscription conditions select
what an observer consumes.

The envelope preserves the original author's `task_id`, `agent_name`, live
`origin`, and delegation `path`. Paths contain task identifiers rather than
agent names, so concurrent invocations of the same worker remain distinct.
An ancestor forwards a single observation rather than nesting envelopes.
Observations never enter the ancestor's model transcript or trigger its tools,
model loop or HITL hooks. Transient sources have transient envelopes and are
not retained by default; `persist_all=True` retains them normally.

Every event exposes the read-only `source_event` property. A local event returns
itself; a `ChildEvent` returns the original event inside the observation,
including after replay. `DownEvent` remains its own source: its inner payload
is an intervention being audited. The property is computed and is not a
serialized event field.

```python
from ag2 import Agent, Context, Scope, ScopedValue, observer
from ag2.events import BaseEvent, ChildEvent, HaltEvent


async def monitor(event: BaseEvent, ctx: Context) -> None:
    if isinstance(event, ChildEvent) and should_stop(event.inner):
        await event.origin.send(HaltEvent(reason="budget exceeded", source="monitor"))


root = Agent(
    "root",
    config=config,
    tools=[worker.as_tool(description="Research")],
    observers=[observer(callback=monitor)],
    variables={
        "findings": ScopedValue({}, scope=Scope.SHARED),
        "depth": 0,
        "private": ScopedValue("root only", scope=Scope.LOCAL),
    },
)
```

This sketch uses application-owned `config`, `worker`, and `should_stop`.

## Addressing and cancellation

`await event.origin.send(custom_event)` addresses the source author, even for
a grandchild. This is the single public entry point for addressed delivery.
Each handle identifies one invocation, so agents with the same name remain
distinct. An observer can retain the handle while that invocation is active.

`StreamHandle` is the live address returned by `event.origin`; applications
catch `DeliveryError` when delivery fails. These types are exported from
`ag2.tools.subagents`. The runner, its result type, and the delegation registry
remain internal implementation details. There is no public lookup or broadcast
by task ID, agent name, or subtree.

Forwarded `UsageEvent` records also expose the original author's live `origin`.
This permits budget callbacks subscribed directly to `UsageEvent` to address
the spender. A local usage record or a replayed record has no live origin.

A queued invocation has no live origin handle. Sending after the addressed
invocation has ended raises `DeliveryError` and emits `DeliveryFailed`;
replaying a stored observation does not recreate its live origin.

Downlink uses the recipient's ordinary pipeline without a turn lock or event
type restrictions. The caller owns the consequences, including reentrant
model calls. Interventions are audited with `DownEvent` and are not echoed
upward from the recipient.

`DownEvent.origin_task_id`, `origin_path`, and `origin_stream_id` identify the
ancestor that sent the intervention. A handle retains its invocation address;
if that invocation ends while delivery awaits a handler, delivery fails rather
than reaching a later invocation on the same persistent stream.

Halt cancels current asynchronous work and its subtree, settles child tasks,
and emits `TaskCancelled` after cleanup. Targeted cancellation returns a
cancelled delegation result; the parent and unrelated siblings continue.
Cancelling the parent also settles its background descendants. A running
synchronous tool in a thread requires cooperative stopping: cancelling its
awaiting coroutine cannot stop the Python thread.

Lifecycle and forwarding are scoped under the stream turn lock. A queued
invocation cannot capture another invocation's events, even when they reuse
the same stream object. The terminal follows its source events and observer
cleanup. Descendants owned by a delegated run settle before its completion.

## Variables

Use a single `variables` mapping on `Agent` or `ask`:

Variables must be serializable in every scope, including `SHARED`. Locks,
clients, connections, and other runtime objects belong in `dependencies` and
are accessed through `Inject` or `context.dependencies`. Delegation copies
the dependencies mapping shallowly, so those objects keep their identity.
`ScopedValue` controls variable inheritance, not whether a runtime object can
be passed as a variable.

| Declaration | Delegation behavior |
|---|---|
| Plain value or `ScopedValue(value, Scope.COPIED)` | Deep copy for each branch |
| `ScopedValue(value, Scope.LOCAL)` | Visible only at its owning level |
| `ScopedValue(value, Scope.SHARED)` | One binding shared throughout the tree |

Each fresh root invocation deep-copies the merged defaults and per-call
variable values. Its shared bindings are shared within that invocation's
tree; they do not alias the caller's original input containers or another
root invocation. Read the current values from `reply.context.variables`.

Only the root declares shared names. A descendant may change an existing
shared value, including rebinding it, but declaring a new top-level shared
name raises `ScopeError`. For arbitrary contributions, predeclare a shared
dict and add keys inside it. Shared changes survive failures and cancellation.
Control-plane keys beginning with `ag:` or `a2a:` are not inherited.

`Variable` injection returns the current value directly. A tool parameter
already injected before another tool rebinds that name retains its old object;
read the mapping again when the current binding is needed. Shared values are
not transactional or automatically synchronized: authors supply their own locks
through `dependencies` for compound concurrent updates, including synchronous
tools running in threads.
Acquire the lock before reading the binding used for a compound update.

## Accounting and HITL

`UsageReport`, `TokenMonitor`, telemetry, UI collectors and compaction read
original `UsageEvent` records from both local work and descendants. Descendant
accounting is forwarded without a `ChildEvent` wrapper, so both
`observer(UsageEvent, callback)` and `stream.where(UsageEvent).subscribe(callback)`
continue to receive it. Callbacks get the original accounting type and fields.
There are no generated subtask rollups. Source identity survives persistence
and usage spans, so one billable record is counted once even if multiple
ancestors recorded it. Equal-cost calls remain separate records. Provider,
model and usage kind stay attached to each original call. Archived rollup
records remain readable, but new runs do not generate them.

`TaskCompleted.usage` is a cumulative stream snapshot for inspection;
never add this snapshot to accounting events. A root
monitor sees descendant spend when its source record is published, without
waiting for the delegated run to finish. Already published accounting remains
available after failure, cancellation, compaction and replay.
`UsageEvent.event_id` identifies one source publication. If cancellation
interrupts forwarding before ordinary storage runs, the original accounting
record is retained in the affected histories before the terminal event.
Each forwarding hop preserves the same `event_id`; it does not create a new
publication. Readers still accept archived usage envelopes.
Explicitly publishing a usage record through `context.send` starts a new
publication, with a fresh ID and no inherited live origin address.

HITL keeps its original request/answer channel: a local hook takes priority;
otherwise the request bridges upward under the requester's context. Ancestor
observation envelopes are inert, so only one hook answers. Both the source
question and answer are available to ancestor observers, and the requester's
timeout covers all hops.

## Migration

Broad subscribers and unfiltered cadence watches now see descendant envelopes.
Filter by event type or source when only local events are relevant. Forwarding
awaits the parent pipeline directly: ancestor callback latency and errors retain
ordinary stream semantics, without queues, drop policies or stalled-observer
recovery. Network forwarding is outside this in-process design.

Lifecycle subscribers now receive `ChildEvent` observations of `TaskStarted`,
`TaskCompleted`, `TaskFailed`, or `TaskCancelled`. Subscribe to `ChildEvent` and
inspect `event.inner`, or use `event.source_event` in a consumer that
accepts both local events and observations. AG-UI, A2UI, and the network task
mirror consume these lifecycle observations directly.

See ADRs [0027](../adr/0027-subagent-events-use-direct-forwarding.md),
[0028](../adr/0028-root-declares-shared-variable-names.md) and
[0029](../adr/0029-account-descendant-usage-from-source-events.md).
