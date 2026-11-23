---
status: accepted
date: 2026-11-22
---

# 0040. `ModelConfig.model` may be `None`

## Context

`ModelConfig.model` was typed `-> str`. Every provider config names a model, so
the type held for them. `ACPConfig` does not fit: its `model` field is
`str | None`, where `None` means "run on the agent's own default". That default
is chosen by the CLI agent at session start, so the config cannot name it.

The mismatch, plus `ACPConfig` lacking `provider` and `create_files_client`,
meant `ACPConfig` did not satisfy `ModelConfig`. The documented usage
`Agent("coder", config=ACPConfig([...]))` failed mypy for users, and the
repository's own tests hid it behind `cast(ModelConfig, cfg)`.

Two fixes were weighed:

- Change `ACPConfig.model` to `str` with `""` meaning the default. The protocol
  stays as it was, but a public field breaks: `ACPConfig(model=None)` stops
  type-checking and `cfg.model is None` checks silently turn false. It also
  swaps a clear `None` for a magic string.
- Widen the protocol's `model` to `str | None`.

## Decision

`ModelConfig.model` returns `str | None`. `None` means the config asks for no
particular model and the backend picks its own. `ACPConfig` keeps its
`model: str | None` field unchanged.

`ACPConfig` also gains `provider`, which raises `NotImplementedError` (an ACP
agent has no AG2 provider, as with the NLIP and MCP-sampling configs), and
`create_files_client`, which raises `FilesAPIUnsupportedError`.

## Consequences

- `ACPConfig` and its presets are `ModelConfig`s for type checkers. Tests and
  user code pass them to `Agent(config=...)` without a cast.
- Code that reads `config.model` must handle `None`. Inside AG2 that was one
  place (the MCP-server answerer, which now reports `"unknown"`); metrics
  already mapped a missing name to `"unknown"`. User code that assigns
  `config.model` to a `str` gets a new mypy error.
- A config can now say "no model" in two ways: `None` (asks for none, the
  backend picks) and `raise NotImplementedError` (no notion of a model at all,
  e.g. NLIP). Readers that report a name must handle both.
