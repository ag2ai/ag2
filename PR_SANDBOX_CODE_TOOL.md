# PR

## Title

```
feat(beta): add SandboxCodeTool with Daytona and Docker code-execution extensions
```

## Description

### Why

`autogen.beta` only supported code execution via LLM-provider built-ins (`CodeExecutionTool` → Anthropic / OpenAI / Gemini server-side sandboxes). This adds a client-side option so you can run code on infrastructure you control — Daytona (hosted) or Docker (local) today, and any custom backend via the `CodeEnvironment` protocol. Works on every provider, supports custom packages and persistent state across calls.

`SandboxCodeTool` requires an explicit `environment` argument — there is **no unsandboxed local default**. The class name is a contract: it executes whatever the model writes, and that should only ever happen inside something that genuinely sandboxes execution. For unsandboxed local shell access, `LocalShellTool` already exists with `allowed`/`blocked`/`readonly` guardrails.

### What

**Core** — `autogen/beta/tools/code/`

- `CodeEnvironment` protocol — `async run(code, language, *, context=None) -> CodeRunResult`, `supported_languages`. The optional `context` lets backends resolve `Variable` markers from `context.variables`.
- `SandboxCodeTool` — wraps any `CodeEnvironment` and exposes `run_code(code, language)` as a function tool. No provider mapper changes — works on every provider.

**Extensions** — `autogen/beta/extensions/` (per the [Beta Contribution Policy](website/docs/beta/contribution_policy.mdx))

- `daytona/DaytonaCodeEnvironment` — built on `AsyncDaytona`. Lazy sandbox creation, `atexit` + `async with` cleanup, supports `image` or `snapshot`, `env_vars`, `resources`, `timeout`. Reads `DAYTONA_API_KEY` / `_API_URL` / `_TARGET` from the environment by default.
- `docker/DockerCodeEnvironment` — local Docker container. Strict safety defaults: `network_mode="none"`, `mem_limit="512m"`, `auto_remove=True`. Runs `docker exec` against a long-lived container so files and installed packages persist between calls. Sync `docker` SDK driven via `asyncio.to_thread`.
- Both backends accept `Variable` for runtime-resolved per-tenant config (`image`, `env_vars`, etc.) — same DI pattern as core builtin tools.

### Usage

```python
from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.tools import SandboxCodeTool
from autogen.beta.extensions.daytona import DaytonaCodeEnvironment
from autogen.beta.extensions.docker import DockerCodeEnvironment

# Hosted sandbox (Daytona)
agent = Agent(
    "analyst",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
    tools=[SandboxCodeTool(DaytonaCodeEnvironment())],
)

# Local container (Docker)
agent = Agent(
    "analyst",
    config=AnthropicConfig(model="claude-sonnet-4-6"),
    tools=[SandboxCodeTool(DockerCodeEnvironment(image="python:3.12-slim"))],
)
```

Install the extras you need: `pip install "ag2[daytona]"` and/or `pip install "ag2[docker]"`. Daytona extra bumped to `daytona>=0.171.0` (for `AsyncDaytona`); Docker extra is the existing `docker>=6.0.0`.

### Validation

- 38 new unit tests added (4 SandboxCodeTool / wiring, 5 custom-backend coverage, 18 Daytona with mocked SDK incl. Variable resolution, 17 Docker with mocked SDK incl. Variable resolution); full `test/beta/tools/` suite still green (no regressions).
- `python -c "from autogen.beta.tools import SandboxCodeTool; SandboxCodeTool()"` raises `TypeError` — explicit environment is enforced.
- Smoke-tested end-to-end against real Daytona: `python` and `bash`, sandbox reused across calls, clean async teardown, full agent flow (model → `run_code` → Daytona → result back to model).
- Docker smoke test (`ms_code/smoke_docker.py`) auto-skips when no daemon is running; covers container lifecycle, `python` + `bash`, file persistence, and verifies the default `network_mode="none"` blocks egress.

### Docs

New page: `website/docs/beta/tools/code_execution.mdx`. Contrasts built-in vs remote code execution, with a Tabs block covering Daytona / Docker / Custom backends, a `!!! warning` callout explaining why `environment` is required, and a state-persistence reference table.
