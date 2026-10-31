---
status: accepted
date: 2026-10-26
---

# 0025. Removed provider parameters get one minor release of deprecation

## Context

`anthropic` 2.x sends Messages API version `2026-10-21`, which answers HTTP 400 to
`temperature`, `top_p` and `top_k` on every model. `AnthropicConfig` exposed all three, so
code that sets `temperature=0` for reproducibility would start failing at request time, or at
construction if the fields were deleted outright. The repository had no deprecation policy.

## Decision

A config field the provider no longer accepts stays settable for one minor release, then is
removed in the next. `AnthropicConfig` keeps `temperature`, `top_p` and `top_k` in 1.3.0 and
drops them in 1.4.0.

While deprecated, `create()` emits a `DeprecationWarning` naming the fields, and what happens
to the value follows the installed SDK. With `anthropic` 2.x it is **dropped, not sent**: that
SDK sends the API version that answers 400, so sending would turn a warning into a failure.
With `anthropic` 1.x it is still sent, in the request's extra body, because its API version
accepts it and those users lose nothing. A direct `AnthropicClient` caller passing the keys in
`create_options` gets the same treatment. The version switch lives in `ag2/config/anthropic/_compat.py`. The type-level mark is `@deprecated` on the `__init__`
and `copy()` overloads, since the decorator cannot be applied to a dataclass field.

The warning is raised in `create()`, not in `__post_init__`, so `copy()` does not repeat it
and construction stays free of side effects.

## Consequences

- Existing callers keep working. With `anthropic` 2.x they lose the determinism they asked
  for, so the warning says the value is ignored there; the behavior depends on the installed SDK.
- `DeprecationWarning` is hidden by default outside `__main__` and test runners, so many
  users will first learn of it from the 1.4.0 release notes.
- A parameter the provider rejects is not a precedent for keeping unsupported values
  forwardable; `extra_body` remains an uninterpreted escape hatch and still reaches the API.
