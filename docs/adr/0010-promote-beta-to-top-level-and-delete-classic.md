---
status: accepted
date: 2026-06-26
---

# Promote `autogen.beta.*` to `autogen.*` and delete the classic framework

## Context

The repository carried two stacked frameworks under one import name:

- the classic AG2 framework at the top level (`autogen.ConversableAgent`,
  `GroupChat`, `oai`, `agentchat`, `coding`, `interop`, the vendored
  `fast_depends`, …), and
- the protocol-driven framework under `autogen.beta.*`.

`autogen.beta` was the product direction and had become nearly self-contained:
the classic code never imported it, and it reached into the classic code in only
four places (`import_utils`, and a `ConversableAgent`/`OpenAIWrapper` interop
adapter). Keeping both shipped a confusing dual surface, doubled the test/CI
matrix, and forced every new feature to choose a namespace.

## Decision

Promote the beta framework to the top level and remove the classic one entirely.

- **Delete all classic code** (~395 modules) and its tests. No code is retained.
- **`autogen.beta.* → autogen.*`** — the beta package *is* the top-level package.
  There is **no `autogen.beta` compatibility alias**; all imports move to
  `autogen.*`.
- **Sever the interop bridge**: `ConversableAdapter` / `Agent.as_conversable()`
  are removed (they existed only to talk to the deleted classic framework).
- **Relocate shared infra** into the package: `import_utils.py` →
  `autogen/_import_utils.py`; depend on the real `fast_depends` package rather
  than the vendored copy.
- **Distribution: `ag2` only.** The PyPI distribution is `ag2` (import name
  `autogen`); the separate `autogen` alias distribution is dropped. The build
  backend is `uv_build` (static version, `module-name = "autogen"`), which is
  why the legacy `setup.py`/`setup_autogen.py` generation machinery was removed.
- **Extras = the `missing_optional_dependency` set.** `[project.optional-dependencies]`
  contains exactly the extras declared via `missing_optional_dependency(...)` in
  the codebase. Packages registered via `missing_additional_dependency(...)` —
  extensions and optional infra (`docker`, `daytona`, `exa-py`, `tinyfish`,
  `sounddevice`, `websockets`, `watchdog`, …) — are **not** extras; they are
  installed directly where needed.
- **CI**: classic-only workflows (core/contrib/integration/optional-deps) are
  deleted; the beta workflows become the canonical test/release pipeline.

## Consequences

- This is a 1.0-scale breaking change: every `from autogen import ConversableAgent`
  (and the classic API generally) breaks by design. Existing `autogen.beta.*`
  imports must move to `autogen.*`.
- The tree, test suite, CI matrix, and packaging are substantially simpler; there
  is a single agent framework and a single published distribution.
- Adding a provider/feature no longer involves a namespace choice.
- Docs and notebooks still referencing the classic API or the `pyautogen`/`autogen`
  distribution names are migrated separately.
