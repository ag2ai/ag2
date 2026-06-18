---
status: accepted
date: 2026-06-18
---

# MemorySkill: in-process scripts run through the single `run_skill_script`, with their JSON-schema disclosed in loaded content

Builds on [0005](./0005-skill-runtimes-own-io-plugin-composes-multiple-runtimes.md),
which made read/execute polymorphic on the runtime and named a forthcoming
`MemoryRuntime`. This records how a **code-defined** skill — instructions,
resources, and *executable in-process callables*, all in RAM — fits that model
without forking the protocol.

## Context

A `MemorySkill` is authored in code, not on disk:

```python
skill = MemorySkill(name="unit-converter", description="…", instructions="…")

@skill.resource(...)         # static content or a live callable
async def conversion_table() -> str: ...

@skill.script(...)           # in-process callable, typed parameters
async def convert(value: float, factor: float) -> str: ...
```

The hard question was the **script**. The model needs a typed JSON-schema to call
`convert(value, factor)`, but our `SkillRuntime.execute(name, script, args)` was
CLI-shaped: a script *name*, positional *string* args, stdout *string* out. Two
ways to deliver a typed schema were considered:

1. **Harvest each in-process script as its own native `FunctionTool`**, hidden
   until `load_skill` via context-aware `schemas(context)`. Gives the provider a
   real per-script schema, but needs a new `script_tools()` protocol method, a
   dynamic-reveal mechanism, and floods the tool list with N tools after load —
   at which point a "skill script" is indistinguishable from a plain `@tool`,
   defeating progressive disclosure.

2. **One generic runner, schema disclosed as content.** A
   single `run_skill_script` tool whose `args` is a union; the per-script schema
   is generated from the callable's signature and embedded in the loaded
   `SKILL.md` content as a `<scripts><parameters_schema>` block. Disclosure is
   inherent (the schema only appears after `load_skill`), and there is one tool,
   always.

## Decision

**Single-runner + schema-in-content model. In-process scripts
stay on the uniform `execute` path; nothing leaves the runtime protocol.**

- **`execute` arg widening.** `runtime.execute(name, script, args)` accepts
  `dict[str, Any] | Sequence[str] | None`. `LocalRuntime` keeps the `list[str]`
  argv path (rejects a dict); `MemoryRuntime.execute` does `await callable(**args)`
  (rejects a list — inline scripts bind by name). `run_skill_script`'s `args` parameter widens to the
  same `oneOf: [object, array<string>, null]` union.
- **`Script` gains `parameters_schema: dict | None`.** `None` = file/CLI script
  (LocalRuntime); populated = in-process (MemoryRuntime). It stays inert *data* —
  the callable lives on the `MemorySkill`/`MemoryRuntime`, never on the
  descriptor. Schema is generated from the `@skill.script` signature via the
  existing `build_model` + `get_schema` machinery.
- **Schema is disclosed in `load_skill` content**, not the tool list.
  `MemoryRuntime.read` appends a `<scripts>` block carrying each script's
  `<parameters_schema>`. Argument validation happens runtime-side.
- **Resources (static + dynamic) stay on `read_resource`.** A `@skill.resource`
  callable is `await`ed at read time; only Tier-3 scripts touched `execute`.
- **Flat composition is sugar, and grouping is associative.** `SkillPlugin`
  accepts loose `MemorySkill`s and wraps each in its own single-skill
  `MemoryRuntime` at its declared position, so the existing last-to-first chain
  governs precedence with no new rule. `SkillPlugin(MemorySkill(a), MemorySkill(b))`,
  `SkillPlugin(MemoryRuntime(a), MemoryRuntime(b))`, and
  `SkillPlugin(MemoryRuntime(a, b))` are behaviorally identical.
- **`MemoryRuntime` is read-only.** `cleanup` is `False`, `ensure_storage()` is a
  no-op; `install` / `remove` / `lock_dir` raise — passing one to
  `SkillSearchToolkit` fails fast with a legible message.

## Consequences / things that look wrong but are deliberate

- **The provider never validates script args against the per-script schema.**
  `run_skill_script.args` is a generic `object`; the model fills it by reading the
  embedded `<parameters_schema>`, and the **runtime** validates. We accept weaker
  provider-side enforcement in exchange for one tool, no dynamic tool reveal, and
  no protocol growth. Choosing native per-script tools (option 1) would invert
  this trade-off.

- **A `MemorySkill`'s scripts populate `Skill.scripts`** (unlike an early draft
  that kept them off it). They must, so the `run_skill_script` capability gate
  (`any(s.scripts)`) fires and the `<scripts>` block renders. `parameters_schema`
  is what distinguishes an in-process script from a file one — not absence from
  the list.

- **Grouping `MemorySkill`s into a `MemoryRuntime` must not change behavior** —
  only declaration order may. A multi-skill `MemoryRuntime` therefore discovers
  its skills in declaration order with last-wins on intra-runtime name clash,
  mirroring the inter-runtime rule, so the three composition forms above stay
  interchangeable.

- **`MemoryRuntime` raising on `install`/`remove`/`lock_dir` is intentional**, not
  an unfinished stub. In-memory skills are defined in code; there is nothing to
  install or lock. The protocol is satisfied structurally; the read-only members
  fail loudly rather than pretend.
