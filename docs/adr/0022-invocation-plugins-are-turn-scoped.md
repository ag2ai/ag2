---
status: accepted
date: 2026-10-02
---

# Invocation plugins contribute to one turn

## Context

Long-lived agents need refreshed capabilities without rebuilding their resources.
Applying fresh plugins to the shared agent would accumulate contributions and
affect concurrent calls.

## Decision

Bind invocation plugins to the existing turn scope under its stream lock.
Restore the original prompt and remove newly supplied defaults on scope
exit, including failure and cancellation. Continuations require the caller to
pass the plugin again, so temporary capabilities cannot silently persist.

## Consequences

The agent and its resources remain reusable across calls with different plugins.
Callers own refresh timing: a fresh `SkillPlugin` rebuilds its catalog and schemas
after runtime cache invalidation; reusing the plugin keeps its snapshot.
Constructor plugins retain their existing lifetime.
