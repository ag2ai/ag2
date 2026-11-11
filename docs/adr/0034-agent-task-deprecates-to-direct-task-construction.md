---
status: accepted
date: 2026-11-03
---

# 0034. Deprecate Agent.task in favor of direct Task construction

> The run-lifecycle scope is amended by [0035](0035-model-runs-and-delegations-share-task-lifecycle.md).

## Context

`Agent.task()` only constructs a `TaskSpec` and a `Task` owned by `agent.name`.
Its callers wrap arbitrary coroutines, use `TaskInject`, and can checkpoint
owner-defined state. Replacing those blocks with `Agent.run()` would execute
the model loop and would not preserve that contract.

## Decision

Keep `Agent.task()` working in 1.3 with a call-site `DeprecationWarning` and a
type-level deprecation mark; remove the method in 1.4.0. The compatible
replacement is `Task(owner_id=agent.name, spec=TaskSpec(...), ...)` from
`ag2.task`. Move `title`, `description`, `payload`, and `capability` into the
spec, preserving the factory's shallow payload copy. Pass `context`,
`ttl_seconds`, `checkpoint_store`, and `resume_from` directly to `Task`.

## Consequences

Existing code keeps its behavior during the migration window. Direct
construction retains lifecycle events, dependency injection, external TTL
ownership, and checkpoint/recovery. `Task`, `TaskSpec`, and `TaskInject` remain
supported. The deprecation by itself does not add lifecycle events to model
runs; that is decided in 0035.
