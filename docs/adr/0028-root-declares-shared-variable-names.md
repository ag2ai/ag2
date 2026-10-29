---
status: accepted
date: 2026-10-27
---

# 0028. The root declares shared variable names

## Context

Subagents v2 shares variable bindings throughout a delegation tree. Allowing a
descendant to declare a new shared name would change its ancestors' and siblings'
namespace and require rules for collisions with their local or copied values.

## Decision

Declare shared names when constructing the root context of a delegation tree.
Descendants may mutate or rebind existing shared variables, but may not declare
new top-level shared names or promote local/copied names to shared. Reject such
declarations with `ScopeError`.

Variables hold serializable state in every scope. `ScopedValue` declares how
that state is inherited; `SHARED` does not make variables a channel for arbitrary
runtime objects. Locks, clients, and connections belong in `dependencies`,
accessed through `Inject` or `context.dependencies`. Delegation shallow-copies
the dependencies mapping, preserving the referenced objects.

A fresh root invocation deep-copies its merged variable defaults and per-call
values. Shared bindings belong to that invocation's tree, not to the caller's
original container or an independent root invocation. Continuation and plugin
overlays preserve existing scope declarations; values introduced as
`ScopedValue` declarations are copied when creating the scoped mapping.

For arbitrary descendant contributions, the root can predeclare a shared
container, for example `variables={"kv": ScopedValue({}, scope=Scope.SHARED)}`.
Adding keys inside this container does not declare new shared variable names.
Authors supply locks through `dependencies` when concurrent operations require
synchronization; declaring a shared variable supplies no locking.

## Consequences

Shared names remain explicit and cannot be introduced by another branch. Authors
who need open-ended contributions allocate a container at the root; descendants
cannot create a new tree-wide variable namespace on demand. Independent local
and copied values remain available, and shared mutations still have no rollback.
Callers inspect the invocation's current shared values through the returned
context rather than expecting their original input containers to be mutated.
