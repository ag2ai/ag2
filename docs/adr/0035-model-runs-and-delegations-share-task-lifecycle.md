---
status: accepted
date: 2026-11-03
---

# 0035. Model runs and delegations share the Task lifecycle

Amends [0034](0034-agent-task-deprecates-to-direct-task-construction.md)'s
scope and preserves [0030](0030-subagent-events-use-direct-forwarding.md)'s
terminal timing.
The deprecation of `Agent.task()` and its removal in 1.4.0 remain unchanged.

## Context

Delegations published lifecycle independently in `Forwarder`, while ordinary
model runs had no typed outcome. Keeping that split leaves cancellation,
expiry, and network observations dependent on how a model run was launched.
Standalone Tasks must still support arbitrary coroutines and checkpointing.

## Decision

Use the same Task transitions and event construction for standalone work,
model runs, and delegations. `ask` / `run`, their reply continuations, and
`resume` accept `capability` and `ttl_seconds`. Either option enables lifecycle
for that invocation; ordinary calls without either remain unchanged.
Delegation enables the same lifecycle automatically, using its existing task
ID, parent tool-call ID, and forwarding path. `emit_events=False` suppresses
that delegation's lifecycle only; source accounting and descendants still flow.
`Forwarder` owns addressing and forwarding, not a second set of transitions.

A model run's task title is the agent name, never the prompt: `TaskStarted`
reaches every observer and is mirrored to the hub. A delegation keeps its
objective as the title. Delegated `TaskStarted` now carries a `TaskSpec`.

An opted-in run settles its background descendants before its terminal, so the
terminal usage snapshot covers them. Opting in therefore also delays the
return of `ask()` until those descendants finish; untagged runs do not wait.

A run starts its lifecycle when driven, not merely on scope entry. An undriven
scope emits nothing; setup failure emits start/failure. Complete, failed,
cancelled and expired are immutable terminals. Cancellation propagates after
emitting `TaskCancelled` and settling running tools and descendants. A matching
external terminal is adopted without emitting a second terminal.

Publish the terminal after middleware and observer cleanup, while the stream
turn lock is still held. A cleanup failure determines the terminal and the
returned outcome, replacing any driver failure if it is the exception that
propagates. `run.result()` returns the model reply while its scope is still
open; the terminal follows on scope exit. `ask()` includes scope cleanup before
returning. External stream subscriptions observe the terminal after run-scoped
observers have been released. Retain captured source usage before publishing
terminals; terminal usage remains a snapshot (0032).

An ordinary call cancelled while waiting for its stream lock never opens a run
scope and emits no lifecycle. Delegations additionally report rejected attempts
to their parent through the same lifecycle, without entering the busy stream.

TTL remains an external observation deadline, not an execution timeout. The
hub's accepted terminal wins races with late owner events, including capability
statistics. A hub-only expiry need not cancel the local model loop; a local
observer can publish the matching `TaskExpired` to adopt that state locally.

Each opted-in continuation/resume creates a fresh identity. Capability and TTL
are per invocation and are not inherited. Runs do not inject a standalone Task
or replace an enclosing `TaskInject`. Run recovery uses trajectory plus
`resume`; standalone Tasks retain owner-defined checkpoint/recovery.

## Consequences

There is one state machine and one terminal selection path for model work.
Existing ordinary calls and descendant provenance/accounting remain stable.
Standalone Task cancellation now emits `TaskCancelled` rather than treating
`CancelledError` as failure. Explicit Task blocks remain the interface for
non-model work, progress injection, and owner-defined checkpoints.
