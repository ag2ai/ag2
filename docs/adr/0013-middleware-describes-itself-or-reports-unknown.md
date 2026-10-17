---
status: accepted
date: 2026-07-31
---

# Middleware describes itself, or is reported as unknown — never guessed

## Context

Nothing could ask a middleware instance what it was or how it was configured. Consumers
that needed to log, compare, or snapshot middleware read `__code__.co_freevars` and
`__closure__` for tool-scoped hooks, or private `_`-prefixed attributes for agent-level
factories.

The two families differ:

- **Agent-level** (`TokenLimiter`, `RetryMiddleware`, `LoggingMiddleware`, …) were already
  classes holding configuration in named attributes — introspectable, but with no contract
  and no public accessor.
- **Tool-scoped** (`ToolMiddleware`, a bare `Callable` alias) is the closure case. The one
  built-in, `approval_required`, kept its settings in closure cells.

Cell names and ordering are an implementation detail of the producing function, so any
refactor breaks consumers silently. A consumer reading only some cells gets a partial
result indistinguishable from a complete one.

## Decision

`describe_middleware(mw)` returns `MiddlewareDescription(kind, config, complete, inner)`.
Middleware opts in via `describe()` (the `DescribableMiddleware` protocol); everything else
is reported as unknown. All seven built-ins opt in.

Non-obvious choices:

- **`__closure__` is never inspected**, though the configuration is reachable there. An
  honest `complete=False` is preferred over a partial answer that cannot be distinguished
  from a complete one.
- **`describe_middleware()` never raises.** A `describe()` that throws or returns the wrong
  type degrades to `complete=False`. The primary uses are logging and observability, so
  introspection must not break its caller.
- **`MiddlewareDescription` is unhashable** (`__hash__ = None`), despite being a frozen
  dataclass. `config` holds arbitrary values including nested descriptions, so hashing
  cannot work in general. Equality is supported; set and dict-key use is not.
- **`__post_init__` overrides an explicit `complete=True`** when any entry in `inner` is
  incomplete. Because every description normalises at its own construction, an inner is
  already `False` when a parent inspects it, so completeness propagates at any depth without
  the parent walking the tree. A helper would have to recurse and could be bypassed by
  direct construction.
- **`Middleware(...)` always reports `complete=False`** and only option names, not values.
  It cannot know whether an option is a credential, and `_cls` is only instantiated per
  turn, so there is no instance to delegate to at description time.
- **`kind` excludes the module path**, so an internal file move does not break a snapshot.
- **`approval_required` became a callable class** (`ApprovalRequired`, factory kept for
  compatibility). Attaching metadata to the closure would state the configuration twice and
  let behaviour and description drift.

## Consequences

- `config` is for settings only. Counters and caches belong in `context.variables` or the
  per-turn `BaseMiddleware` instance; a description carrying a moving number produces flaky
  snapshots. The built-ins already comply.
- Middleware wrapping other middleware reports it in `inner`, keeping `config` a flat
  settings mapping.
- `describe()` must not expose credentials. Built-ins summarise live objects: `LoggingMiddleware`
  reports its logger's name, `TelemetryMiddleware` only the keys of `span_attributes`,
  `MetricsMiddleware` registry presence as a bool.
- **Sharing structure is out of scope.** A description reports one instance; grouping by
  object identity belongs to whatever walks the agent.
- User-written closures keep working and are reported as unknown. A complete description
  requires a class with `describe()`.
