---
status: accepted
date: 2026-11-06
---

# 0036. Provider exports load on first use

## Context

`ag2/__init__.py` imports `ag2.agent`, which imports `ag2.config`. That
package eagerly imported every installed provider SDK (anthropic, openai,
google-genai, mistralai, ollama, xai-sdk, dashscope, …) and built their
pydantic models, so any `import ag2.*` took about 1.3 s of SDK imports before
user code ran. That cost is paid even by processes that use one provider, or
none (cold starts, CLIs, workers that wake on demand).

## Decision

`ag2.config` keeps `LLMClient`, `ModelConfig` and `ModelProvider` eager. It
exports every provider class through a PEP 562 module `__getattr__` built by
`ag2._lazy.lazy_exports`: the provider module is imported on first attribute
access and the value is cached in the package namespace. An `ImportError`
still yields the `missing_optional_dependency` stand-in with the same install
hint, now created on first access instead of at import. A `TYPE_CHECKING`
block keeps the real imports visible to type checkers, and `__dir__` lists
every export.

Core modules must not import a provider module, so `import ag2` loads no
provider SDK; `test/config/test_lazy_exports.py` enforces this.

## Consequences

- `import ag2` no longer pays for provider SDKs it does not use. A process that
  wants them preloaded (pre-fork servers, warm-up hooks) must import the
  provider explicitly, for example `from ag2.config import OpenAIConfig`.
- `vars(ag2.config)` shows a provider only after it is first accessed; use
  `dir()` or `__all__` to list exports.
- Tooling that treats any module with `__getattr__` specially (freezegun's
  `ignore=` lists, some auto-instrumentation) sees `ag2.config` as such a module.
- New provider exports go into the lazy map, not a `try/except ImportError`
  import; AGENTS.md's re-export rules describe both patterns.
