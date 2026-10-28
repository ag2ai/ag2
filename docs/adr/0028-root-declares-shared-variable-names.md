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

For arbitrary descendant contributions, the root can predeclare a shared
container, for example `variables={"kv": ScopedValue({}, scope=Scope.SHARED)}`. Adding keys
inside this container does not declare new shared variable names. Authors supply
their own locks when concurrent operations require synchronization.

## Consequences

Shared names remain explicit and cannot be introduced by another branch. Authors
who need open-ended contributions allocate a container at the root; descendants
cannot create a new tree-wide variable namespace on demand. Independent local
and copied values remain available, and shared mutations still have no rollback.
