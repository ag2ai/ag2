---
name: repo-structure
description: AG2 repository layout — where to add new code, tests, and docs; module conventions; PR workflow.
version: 1.0.0
license: Apache-2.0
---

# Repo Structure

Reference guide for navigating and extending the `ag2ai/ag2` monorepo.

## Top-level layout

```
ag2/
├── autogen/beta/          # New beta package (this is the primary development surface)
├── autogen/               # Legacy autogen package (not covered here)
├── test/beta/             # Tests for autogen.beta
├── website/docs/beta/     # Documentation for autogen.beta (MDX + Mintlify)
├── skills/                # Bundled repository-development skills (this directory)
├── pyproject.toml         # Project metadata and dependencies
└── .pre-commit-config.yaml
```

## autogen/beta package layout

```
autogen/beta/
├── __init__.py            # Top-level exports: Agent, Context, fanout, ...
├── agent.py               # Agent class + AgentReply
├── config.py              # LLM provider configs (OpenAIConfig, AnthropicConfig, ...)
├── exceptions.py          # Custom exception hierarchy
├── fanout.py              # fanout() for parallel agent invocations
├── middleware/            # Middleware system
│   ├── __init__.py
│   ├── builtin/           # Built-in middlewares (Telemetry, Prometheus, Conditional, ...)
│   └── ...
├── scheduler.py           # AgentScheduler (cron/interval/delay)
├── tools/                 # Tools and toolkits
│   ├── __init__.py        # Public tool exports
│   ├── builtin/           # Built-in (provider-side) tools
│   ├── final/             # @tool decorator, Toolkit base, FunctionTool
│   ├── skills/            # SkillsToolkit, SkillSearchToolkit, LocalRuntime
│   └── toolkits/          # FilesystemToolkit, MemoryToolkit, DeferredToolkit, MCPServer
└── ...
```

## Where to add new code

| What you're adding | Where it goes |
| :--- | :--- |
| New LLM provider config | `autogen/beta/config.py` or a new config module |
| New middleware | `autogen/beta/middleware/builtin/<name>.py`; export from `middleware/builtin/__init__.py` and `middleware/__init__.py` |
| New toolkit | `autogen/beta/tools/toolkits/<name>.py`; export from `toolkits/__init__.py` and `tools/__init__.py` |
| New built-in provider tool | `autogen/beta/tools/builtin/<name>.py`; export from `tools/builtin/__init__.py` |
| New top-level feature | `autogen/beta/<name>.py`; export from `autogen/beta/__init__.py` |

## Tests

Test files mirror the source layout under `test/beta/`:

```
test/beta/
├── conftest.py            # Shared fixtures: context, async_mock
├── test_agent.py
├── test_fanout.py
├── tools/
│   ├── test_local_skills.py
│   └── test_skill_search.py
└── ...
```

Rules:
- One test file per source module.
- File name: `test_<source_module_name>.py`.
- Fixtures defined in `conftest.py` are available everywhere.

## Documentation

Docs live at `website/docs/beta/`. See the `docs-writer` skill for MDX conventions.

```
website/docs/beta/
├── roadmap.mdx            # Completed / In Progress / Future Priorities
├── agents.mdx             # Agent, AgentReply
├── tools/
│   ├── common_toolkits.mdx
│   └── ...
└── advanced/
    ├── fanout.mdx
    ├── scheduler.mdx
    └── ...
```

Nav is declared in `website/mint-json-template.json.jinja`. Add a new page to the appropriate `group.pages` array there.

## PR workflow

1. Branch from `main` with a descriptive name: `feat/beta-<feature>`, `fix/beta-<bug>`, `docs/beta-<topic>`.
2. Commit with the prefix `feat(beta):`, `fix(beta):`, `docs(beta):`, etc.
3. Pre-commit hooks run automatically: `ruff`, `mypy`, `typos`, `codespell`, license headers.
4. Open PR against `main`; the CI matrix tests Python 3.10 / 3.11 / 3.12 on Ubuntu and macOS.
5. Never force-push to `main`. Never merge your own PR.

## Roadmap

`website/docs/beta/roadmap.mdx` tracks what's done, in progress, and planned. When shipping a feature:
1. Move it from **In Progress** or the relevant **P-level** block into **Completed**.
2. If it was a P2 item listed as "(done — ...)", update the note with the real PR number.
