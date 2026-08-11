---
status: accepted
date: 2026-06-18
---

# Skill runtimes own read & execute; Plugin/Toolkit compose multiple runtimes via last-to-first chain delegation

Surfaced while introducing the `Skill` aggregate and a forthcoming
`MemoryRuntime` whose skills are read and executed straight from memory (no
filesystem). An earlier draft of this ADR proposed the opposite — content reads
done locally on self-contained `Skill` objects, execution alone abstracted. That
draft was never implemented; this records the decision that replaced it.

## Context

`SkillRuntime` (`ag2/tools/skills/runtime/protocol.py`) owns storage,
discovery, and execution. The open question was where **reading** content
(`SKILL.md`, resources, script source) and **running** scripts should live now
that more than one backend exists:

- A `LocalRuntime` reads from the filesystem and runs scripts in a subprocess /
  sandbox.
- A `MemoryRuntime` holds skill content in RAM and must read and execute it
  there — a local-filesystem read would have nothing to read.

Making `Skill` self-contained (carrying its own content + executor) was
considered and rejected: it pushes a runtime/executor reference onto every
`Skill`, and it does not compose when you want **two sources at once** — global
(`~/.agents/skills`) and project (`.agents/skills`) — each with its **own**
sandbox/timeout/storage config. Today a single runtime forces one config for
both.

## Decision

**Read and execute are polymorphic operations on the *runtime*. `Skill`,
`Script`, and `Resource` are pure descriptors. The `Toolkit`/`Plugin` hold
*multiple* runtimes and dispatch by last-to-first chain delegation.**

- **Runtime IO is name-keyed and raises on miss.** `runtime.read(name)`,
  `runtime.read_resource(name, resource)`, `runtime.execute(name, script, args)`.
  Each raises `SkillNotFoundError` when *this* runtime does not own `name`.
  Execution mechanics (shebang detection, chmod, `python3`/`sh`/`./` command
  building, sandbox dispatch) live **inside** the runtime, so `MemoryRuntime` can
  run from RAM and `LocalRuntime` via its sandbox.
- **`runtime.skills`** exposes that runtime's descriptors (cached discovery).
  `Skill` = `name` + `metadata` (pure frontmatter) + `scripts: list[Script]` +
  `resources: list[Resource]` + `resources_truncated`. `Script`/`Resource` carry
  only `name`. No `read()`/`run()` on descriptors, no runtime back-reference.
- **`Toolkit`/`Plugin` accept `*runtimes`.** Example:
  `SkillPlugin(LocalRuntime("~/.agents/skills"), LocalRuntime(".agents/skills"))`.
- **Chain delegation, last-to-first.** A tool call walks the runtimes from last
  to first, catching `SkillNotFoundError` to try the next. The first runtime that
  owns the skill handles it; its result — success **or** genuine error — ends the
  walk.
- **One precedence rule, applied everywhere.** Last-to-first / dedup-by-name
  governs execution routing, reads, the injected `<available_skills>` catalog,
  the `Literal[name]` constraint, **and** the gating decision. So a `foo` defined
  in both project and global resolves to the project one consistently across
  display, routing, and reads.

## Consequences / things that look wrong but are deliberate

- **`SkillNotFoundError` is load-bearing — it is the *only* fall-through.** A
  script that exists but exits non-zero, or a missing script *inside* an owning
  runtime, must propagate, **not** advance the chain. If any error fell through,
  a failing project skill would silently re-run a **different same-named** global
  skill — wrong result, and a side-effecting script could execute twice. Anything
  caught around the chain other than `SkillNotFoundError` is a bug.

- **Last-to-first looks backwards next to the constructor argument order.**
  `SkillPlugin(global, project)` lists global first, yet project wins — because
  the *last* runtime has priority (project overrides global, matching the
  existing path-priority intuition). Reversing the walk to first-to-last would
  invert that and is wrong.

- **`Skill` carries no runtime reference, yet `run_skill_script` reaches the
  right runtime.** Routing is the toolkit's chain walk, not a property of the
  descriptor. Descriptors stay inert and shareable; the toolkit owns the
  `*runtimes` tuple.

- **A `MemoryRuntime` (skills not on the host filesystem) is now a first-class
  citizen.** Because reads go through `runtime.read`, not a local file open, an
  in-RAM runtime needs no special-casing — it implements the same three IO
  methods.

To read or run a skill: call the toolkit, which walks `*runtimes` last-to-first
and delegates to `runtime.read` / `runtime.read_resource` / `runtime.execute`.
Catch nothing but `SkillNotFoundError` around that walk.
