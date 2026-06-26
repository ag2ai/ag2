---
status: accepted
date: 2026-06-26
---

# Rename the import package from `autogen` to `ag2`

## Context

The PyPI distribution has long been `ag2`, but the importable module was
`autogen` (`from autogen import Agent`) — the split [ADR 0010](0010-promote-beta-to-top-level-and-delete-classic.md)
left in place while it focused on promoting the protocol framework and deleting
the classic one.

The split is now pure friction: the project, docs, URLs, and CLI are all `ag2`,
yet the import name was `autogen` — a name that also belongs to the upstream
Microsoft AutoGen project this repo forked from, which is a constant source of
confusion. With the classic API already gone, there is no compatibility reason
left to keep the `autogen` import name.

## Decision

Rename the import package `autogen` → `ag2`. `import ag2` / `from ag2 import …`
is the only supported import surface.

- **Directory `autogen/` → `ag2/`**; `module-name = "ag2"` in `[tool.uv.build-backend]`.
  The distribution name (`ag2`) is unchanged.
- **Hard break, no shim.** There is no `autogen` compatibility package or alias;
  `import autogen` raises `ModuleNotFoundError`. This is consistent with the
  1.0-scale breaking changes made in ADR 0010.

## Consequences

- Every `import autogen` / `from autogen…` (including the exception qualnames
  such as `ag2.exceptions.ToolNotFoundError` and the root logger name `ag2`)
  changes by design.
- Heritage references to the upstream `microsoft/autogen` project and the
  historical `pyautogen` distribution are intentionally preserved; only the
  *import package* was renamed.
- User-facing docs that still teach the removed classic API (README examples,
  the website docs tree, and the CLI skill content) are stale beyond a
  mechanical rename and are migrated separately, not in this change.
