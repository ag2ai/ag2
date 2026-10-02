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
On scope exit, remove only the bound plugin prompt fragments and unreplaced
default values, including failure and cancellation. Track prompt fragments by
identity so equal text and unrelated conversation updates survive cleanup.
Continuations require the caller to pass the plugin again.

## Consequences

The agent and its resources remain reusable across calls with different plugins.
Callers own refresh timing: a fresh `SkillPlugin` rebuilds its catalog and schemas
after runtime cache invalidation; reusing the plugin keeps its snapshot.
Constructor plugins retain their existing lifetime and are not replaced: the same
kind of plugin attached in both places contributes twice (two skill catalogs).
Reassigned defaults and other conversation updates persist, except that a default
re-set to the identical object is indistinguishable from the plugin's own and is removed. Cleanup tracks
framework-bound contributions; it does not roll back arbitrary plugin code.
