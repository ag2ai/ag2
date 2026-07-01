---
status: accepted
date: 2026-06-18
---

# A tool's dependency provider comes from the Context, not `set_provider`

Surfaced while giving `MemorySkill` scripts the same FastDepends resolution as
tools: the provider they needed was already on the live context, which raised the
question of why a separate per-tool provider channel existed at all.

## Context

`fast_depends` resolves `Depends(...)` against a *provider* (which also carries
`provider.override(...)`). The framework fed that provider to a tool two ways at
once:

- **Channel A — `set_provider` / `tool.provider`.** At *registration* time,
  `FunctionTool.ensure_tool(t, provider=agent.dependency_provider)` stored the
  provider on the tool, and `FunctionTool.__call__` passed `self.provider` to
  `asolve`. `Tool`, `FunctionTool`, and `Toolkit` each had a `set_provider`.
- **Channel B — `context.dependency_provider`.** At *run* time,
  `Agent.__build_context` (and the a2a / live / hitl / stream entry points) set
  `context.dependency_provider = agent.dependency_provider`.

Both channels carried the **same** `agent.dependency_provider` object, and
`tool.provider` was read in exactly **one** place — `FunctionTool.__call__`. So
Channel A was a parallel, registration-time copy of a value already present on
every context that executes a tool.

## Decision

**Resolve the provider from the live context. Delete Channel A.**

- `FunctionTool.__call__` passes `context.dependency_provider` to `asolve`.
- `set_provider` is removed from `Tool`, `FunctionTool`, and `Toolkit`; the
  `provider` slot/field is gone; `FunctionTool.ensure_tool` no longer takes a
  `provider`. Callers (`Plugin.add_tool`, `RealtimeAgent`) drop the argument.
- The provider is now a **run-time property of the `Context`**, resolved per call,
  with a single source of truth.

## Consequences / things that look wrong but are deliberate

- **Tools no longer carry a provider** — `ensure_tool` is pure shaping
  (deepcopy / wrap). A reader expecting a tool to "have" its provider will not
  find one; that is intentional. The provider lives on the context the tool runs
  against.
- **Every context that executes a tool must set `dependency_provider`.** The
  agent, a2a, live, hitl, and stream paths already do. A context built without
  one (e.g. a bare `ConversationContext(stream=...)`) resolves `Depends` against
  no provider — the same behavior the old `set_provider(None)` produced, so no
  regression. The full beta suite passes unchanged.
- **`MemorySkill` scripts and resources needed no special wiring.** Because they
  invoke through a `FunctionTool` and the runtime threads the live context, they
  pick up the agent's provider for free — this removal is the direction the
  MemorySkill work already standardized on (see
  [0006](./0006-memory-skills-in-process-scripts-schema-in-content.md)).
