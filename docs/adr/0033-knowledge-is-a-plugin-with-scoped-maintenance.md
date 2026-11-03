---
status: accepted
date: 2026-11-03
---

# 0033. Knowledge is a plugin with scoped maintenance

## Context

`Agent(knowledge=KnowledgeConfig(...))` owns a store dependency, a hidden tool,
bootstrap, event logging, compaction, and aggregation. Migrating only the tool
would lose maintenance and force applications to reconstruct the harness.
`on_turn` begins after schema resolution and middleware/observer setup and ends
before observer completion, so it cannot preserve the existing knowledge scope.

## Decision

Export `KnowledgePlugin` from `ag2.knowledge` as an ordinary Plugin collector
(ADR 0001). It contributes a dependency default, an optional `knowledge` tool,
and middleware factories. Knowledge owns its runtime state and strategies'
wiring; Agent retains only a temporary compatibility adapter.

Add the generic optional `ScopedMiddlewareFactory.scope(event, context)`
protocol. Agent enters these async context managers in factory registration
order before resolving tools or instantiating middleware and exits them in
reverse order after observer completion. This brackets setup and teardown
without reordering unrelated `on_turn` middleware or adding registration hooks.
Factory state survives turns; scope-local context and store references do not
leak between concurrent streams. Bootstrap honors the existing sentinel and
coordinates overlapping calls through the configured factory. Aggregation
precedes compaction, and final event logging follows observer completion.

Only the configured store's initialization is memoized. Temporary dependency
overrides use their sentinel on later calls; active scopes own the bootstrap
locks, with weak factory references, so the plugin does not extend those
stores' lifetime. Custom stores need neither hashing nor weakref support.

All new-plugin capabilities resolve the effective KnowledgeStore dependency.
Tools use ordinary precedence (ADR 0020) and approval middleware; automatic
maintenance is outside individual tool approvals. The compatibility adapter
preserves legacy dependency precedence and hidden-tool exclusion from legacy
subtask inheritance. Mixed legacy/new knowledge configuration is rejected
before store I/O.

Ship the replacement and a constructor-only DeprecationWarning in 1.3;
remove `Agent(knowledge=...)` in 1.4.0. Keep KnowledgeConfig and its existing
imports; the class is not independently deprecated.

## Consequences

A knowledge-free Agent runs no knowledge scope. Store construction remains
free of I/O. Custom bootstrap and strategy protocols, logging opt-out,
source usage retention, failure events, and non-fatal log failures remain.
Aggregation retains its finally-block behavior. Final logs still require a
normal exit of the enclosing scope: AgentRun closes scopes normally even on
background-turn failure, while a failing reply.ask propagates through its
scope. The migration preserves this entry-point distinction.

For 1.4.0 remove the `knowledge` constructor parameter from all overloads and
the implementation, `KNOWLEDGE_DEPRECATION`, `_legacy_knowledge` initialization
and validation, `ag2/knowledge/_legacy.py`, and the bound-store compatibility
branches in knowledge tools and middleware factories. Delete focused legacy
compatibility tests and the migration/deprecation notes once obsolete. Keep
KnowledgePlugin's runtime implementation, KnowledgeConfig and its exports,
ScopedMiddlewareFactory, and `_additional_tools` entries for legacy task tools
until that API's separate removal. Do not delete unrelated residents of shared
collections as part of knowledge removal.
