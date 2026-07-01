---
status: accepted
date: 2026-06-16
---

# Public APIs take ready-to-use objects and reuse existing seams

Two general design rules, surfaced while cleaning up the eval entry points
(`run_agent` / `run_pairwise` / `run_variants` in `ag2/eval/runtime/`)
but meant to apply framework-wide. The eval change is the worked example.

## Context

The eval runners each take a *target* to run over a *dataset*. Two earlier
conveniences made both inputs polymorphic:

- **Suite as anything** — `suite=` accepted a `Suite`, a `Path`/path-string, or
  an inline `list[dict]`, and the runner dispatched to `Suite.from_jsonl` /
  `Suite.from_list` internally.
- **Target as a factory** — `agent=` accepted an `Agent` instance *or* a callable
  that built a fresh `Agent` per task (optionally taking a keyword-only `config`
  the runner injected). `Variants` was built entirely on this:
  `Variants.from_configs(factory, {...})` produced one `partial` per variant.

Both turned out to be the wrong kind of convenience, each for a reason that
generalizes past eval.

## Decision

### 1. A parameter's type should be a ready-to-use object, not a side-effecting recipe

Accept what the caller already holds, fully constructed. Reject union members
whose meaning is "go do I/O or construction *inside* the function": a `Path`
silently implies *reading happens in here*, a `list[dict]` implies *parsing and
validation happen in here*. Push that work to the call site so the side effect
is explicit and lives outside the callee — the function just consumes the
result. (This is the runtime-side-effect rule from `AGENTS.md` applied to
signatures: the I/O belongs in the constructor seam — `Suite.from_jsonl(path)`,
`Suite.from_list([...])` — not in every consumer of a `Suite`.)

Allow exactly one deliberate exception: the single most common, lowest-friction
shorthand may stay as sugar. Here that is `str` — a bare string is the easiest,
most-reached-for input, so `run_*` keeps it as "a one-prompt suite". Everything
heavier (files, task lists) is constructed explicitly.

> **Eval application:** `suite: Suite | str | Path | list[dict]` → `suite: str | Suite`.
> Callers pass `Suite.from_list([...])` / `Suite.from_jsonl(path)`; a bare `str`
> is one-prompt sugar, which keeps the simplest run a one-liner:
> `await run_agent("Hi, agent!", agent=agent)`.

### 2. Don't add a mechanism for behavior an existing seam already covers

Before introducing indirection, check whether existing composition already does
the job. The factory target existed only to vary an agent's config per task —
but `Agent.ask(config=…)` *already* overrides the agent's own config per call.
So on-the-fly agent construction is redundant: pass the agent you have (with or
without a baked-in config) and let the per-call override carry the rest. The
parallel machinery the factory required — signature introspection ("does it take
`config`?"), a `RuntimeWarning` when it doesn't, build-vs-reuse branching — is
deleted rather than maintained.

> **Eval application:** `run_*` consume a prebuilt `Agent` instance; per-task
> config flows via `model_config` → `ask(config=…)`. `Variants` is a mapping of
> name → `Agent` instance (`agents=`, with an `axis` label defaulting to
> `"variant"`); the `from_*` factory constructors are removed. `store_dir` is
> optional across all three runners.

## Consequences

- The runner no longer introspects callables or branches on build-vs-reuse — a
  target is just an object with `.name` and `.ask`. A construction that can fail
  now fails at the call site, where the caller sees it; a failing `ask` is still
  captured on the per-task `Trace` (never raised out of `run_*`), so a bad
  variant is recorded rather than aborting a sweep.
- One shared `Agent` instance is reused across a suite's tasks (and repeats).
  Agents are effectively stateless across `ask` calls — history lives on the
  per-call stream — so reuse is safe; do not rely on per-task instance identity.
- Applying rule 1 elsewhere: a new public function that needs a built thing
  should take the built thing, and offer a separate explicit constructor for the
  path/dict/recipe form — not fold the I/O into itself. Reserve the `str`-style
  sugar for the one input that earns it.
