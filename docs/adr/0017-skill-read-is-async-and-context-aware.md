---
status: accepted
date: 2026-09-15
---

# `SkillRuntime.read` is async and context-aware, so a MemorySkill body can be rendered per read

Completes the move begun in
[0005](./0005-skill-runtimes-own-io-plugin-composes-multiple-runtimes.md) (runtimes
own IO) and [0006](./0006-memory-skills-in-process-scripts-schema-in-content.md)
(code-defined skills with in-process scripts). Those two made `read_resource` and
`execute` async and context-aware; `read` was left behind as the one sync,
context-free IO method. This records closing that gap and what it buys.

## Context

After 0006 a `MemorySkill` was dynamic everywhere except where it mattered most:

| member | invocation | dependency injection |
| --- | --- | --- |
| `@skill.resource` | `await` on every read | yes |
| `@skill.script` | `await` on every call | yes |
| `instructions` | a `str` frozen in `__init__` | no |

So a skill could serve a live roster as a *resource*, but could not say "deploy to
`{region}`" in its own body — the body was whatever string was passed at
construction, typically at import time, long before any conversation exists. The
workaround (stuff the variable part into a resource and instruct the model to go
read it) costs an extra tool round-trip and inverts progressive disclosure: the
part the model always needs becomes the part it has to ask for.

The obstacle was purely the protocol shape. `SkillRuntime.read(name) -> str` is
sync and takes no context, so there was nowhere to `await` a callable and nothing
to resolve `Context` / `Variable` / `Inject` against.

## Decision

**Bring `read` in line with `read_resource` and `execute`: `async def read(name,
context)`. A `MemorySkill`'s `instructions` may then be a callable, rendered on
every read through the same `FunctionTool` path as a resource.**

- **Protocol.** `read` becomes `async` and takes a **required** `context`
  positioned after `name`, mirroring `read_resource`. `LocalRuntime.read` accepts
  and ignores it, exactly as it already does for `read_resource` — a file read
  needs neither.
- **`instructions: str | Callable[..., str | Awaitable[str]]`.** A callable is
  wrapped with `tool()` at assignment, so a dynamic body runs through
  `CallModel.asolve` with `__ctx__` and the context's dependency provider — the
  same path resources and scripts already take. Sync callables run in a worker
  thread; `Depends` / `Variable` / `Inject` / `Context` all resolve.
- **`instructions` is a method that doubles as a decorator**, matching
  `@skill.resource` and `@skill.script`:

  ```python
  @skill.instructions
  async def body(region: Annotated[str, Variable("region")]) -> str: ...
  ```

  It accepts a string too (`skill.instructions("text")`), validates (a non-`str`,
  non-callable raises `TypeError`), and re-wraps on every set, so the cached
  `FunctionTool` can never drift from the value it renders.

- **One slot, one reader.** The body is stored as `str | FunctionTool` in a single
  slot and read back by `get_instructions()`, alongside the existing
  `get_resource()` / `get_script()`. Storing the two forms in two fields would put
  an "exactly one is non-`None`" invariant on the class that the sum type makes
  structurally impossible.
- **Only the body is dynamic.** The catalog entry — `name` and `description` in
  `<available_skills>` — stays the construction-time snapshot 0005 established.

## Consequences / things that look wrong but are deliberate

- **`read` gaining a required `context` is a breaking protocol change, taken
  rather than adding an optional parameter or a parallel `aread`.** There are
  exactly two implementations in-tree and exactly one call site
  (`SkillsToolkit._route_read`), so the cost is a handful of lines now versus a
  permanently forked read path. A protocol whose three IO methods have three
  different shapes is the thing 0005 set out to avoid.

- **`LocalRuntime.read` is `async` while doing entirely synchronous work.** That
  is the protocol being uniform, not the filesystem becoming async — same as
  `LocalRuntime.read_resource` since 0006. The alternative, letting each runtime
  pick its own colour, pushes the union back into the caller.

- **A dynamic body is not capped, while a resource read is** (`_RESOURCE_READ_CAP`,
  100k chars). The body is symmetric with `LocalRuntime`'s `SKILL.md` read, which
  is likewise uncapped: a skill's own instructions are authored content, not
  arbitrary data pulled in at read time. A runaway body is an authoring bug that
  should be visible, not silently truncated mid-sentence.

- **`skill.instructions` is no longer readable as a string** — it is now a bound
  method, and the value is read through `get_instructions()`.
  This is the price of spelling the decorator `@skill.instructions`, which is the
  spelling that matches `@skill.resource` and `@skill.script`; a property and a
  decorator cannot share one name without a hybrid str-subclass-that-is-callable,
  which is more magic than the ergonomics are worth. A constructor argument
  (`MemorySkill(instructions=...)`) still works and remains the shortest form for
  a static body; the decorator wins when both are given, since it runs later.

- **The decorator takes no `name=` / `description=`, unlike the other two.** A
  body has neither — it is one value, not an entry in a keyed collection. The
  called form `@skill.instructions()` is accepted anyway, purely so the habit of
  writing empty parens does not produce a baffling error.

- **The body is re-rendered on every `load_skill`, with no caching.** That is the
  feature — a body that is a construction-time snapshot is what a plain string
  already gives you. A callable that is expensive should cache internally, where
  its author knows the right key.
