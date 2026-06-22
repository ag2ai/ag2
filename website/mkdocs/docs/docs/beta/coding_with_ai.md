---
title: "Coding with AI Assistants"
sidebarTitle: "AI Coding Assistants"
---

Set up your AI coding assistant (Claude Code, Cursor, Copilot, Codex, Windsurf, or any agent) so it can build **AG2 Beta** apps with you using current, accurate APIs and examples.

!!! tip "Point your agent at this page"
    Paste this page's link into your assistant and ask it to follow the setup. It can install the AG2 skills and configure itself. Everything below is written to be runnable copy-paste.

## Why this matters

AG2 Beta (`autogen.beta`) is a new, async, protocol-driven API. Models were largely trained on the older `autogen` / `pyautogen` surface, so out of the box an assistant will reach for stale patterns such as synchronous `ConversableAgent`, `initiate_chat`, and the like. The setup on this page gives your assistant three things it otherwise lacks: **AG2-specific skills**, a **Beta-only docs reference**, and **project rules** that keep it on the Beta API.

!!! warning "Beta only"
    This page targets `autogen.beta`. The pre-Beta API (known as the classic API) is being retired at v1.0, so point your assistant at the Beta docs and skills, not legacy `autogen` examples it may have memorized.

## Step 1: Install the AG2 Skills

The most critical and dev-accelerating step.

[ag2-skills](https://github.com/ag2ai/ag2-skills){.external-link target="_blank"} is a catalog of [Agent Skills](https://agentskills.io/){.external-link target="_blank"}, on-demand instruction packs that teach an assistant how to build with AG2 Beta. Each skill loads only its name and description until it's relevant, then pulls in the full recipe. Skills cover quickstart, custom tools, the multi-agent network, middleware, memory, structured output, evaluation, and more.

The fastest path uses the [`skills` CLI](https://skills.sh){.external-link target="_blank"}:

=== "All skills"
    ```bash
    # Install the full AG2 Beta skill catalog
    npx skills add ag2ai/ag2-skills
    ```

=== "One skill"
    ```bash
    # Install just the quickstart (good first taste)
    npx skills add ag2ai/ag2-skills@ag2-quickstart
    ```

=== "Manual (Claude Code)"
    ```bash
    # Clone and copy individual skills into your user skills directory
    git clone https://github.com/ag2ai/ag2-skills.git
    cp -r ag2-skills/skills/ag2-overview   ~/.claude/skills/
    cp -r ag2-skills/skills/ag2-quickstart ~/.claude/skills/
    ```

!!! tip "Where to start"
    After installing, tell your assistant to load **`ag2-overview`** (a map of AG2 Beta capabilities) and **`ag2-quickstart`** (a minimal working agent). From there it can pull in the specific skill it needs, such as `ag2-add-custom-tool` or `ag2-network-quickstart`.

## Step 2: Point your assistant at the Beta docs

Skills teach patterns; the docs keep your assistant honest about the *exact* current signatures. The biggest risk is your assistant falling back on the **classic** `autogen` API it was trained on (`ConversableAgent`, `initiate_chat`, `GroupChat`) — all retiring at v1.0. Give it Beta-only ground truth:

- **Live Beta docs:** [`https://docs.ag2.ai/latest/docs/beta/agents/`](https://docs.ag2.ai/latest/docs/beta/agents/){.external-link target="_blank"} — point your agent at this section (most assistants can fetch a URL).
- **Beta docs source (Markdown):** [`ag2ai/ag2/website/docs/beta`](https://github.com/ag2ai/ag2/tree/main/website/docs/beta){.external-link target="_blank"} — the raw `.mdx` your agent can read directly from GitHub.
- **Beta `llms.txt`:** [`https://docs.ag2.ai/latest/llms.txt`](https://docs.ag2.ai/latest/llms.txt){.external-link target="_blank"} — a machine-readable index of the Beta docs following the [llms.txt standard](https://llmstxt.org/){.external-link target="_blank"}, plus [`llms-full.txt`](https://docs.ag2.ai/latest/llms-full.txt){.external-link target="_blank"} for the entire Beta docs in one file. Both are Beta-scoped, so they never point your agent at the classic API.

Then anchor your prompt: *"Build with `autogen.beta` only. If a signature is unfamiliar, check the Beta docs before writing code — do not use the classic `autogen` API."*

!!! warning "Be careful with generic docs indexers for AG2 right now"
    Third-party docs-MCP servers and code indexers typically ingest the **entire** `ag2` repository, which still has the classic API. Pointed at AG2 today they surface `ConversableAgent` / `initiate_chat` examples — the opposite of what you want for Beta. Until a Beta-scoped index exists, rely on the AG2 Skills and the Beta docs links above, which are Beta-only by construction.

## Step 3: Add project rules to your repo

A rules file pins AG2 Beta conventions for every session in your project. The open [AGENTS.md](https://agents.md/){.external-link target="_blank"} standard is read by Cursor, Copilot, Codex, Gemini CLI, Windsurf, and others; Claude Code reads `CLAUDE.md`. Drop one (or both, a symlink works) at your repo root:

```markdown
# AGENTS.md

This project is built on **AG2 Beta** (`autogen.beta`). Follow these rules.

## API surface
- Import only from `autogen.beta` and its submodules (`autogen.beta.config`,
  `autogen.beta.tools`, ...). Do NOT use the legacy `autogen` / `pyautogen`
  API (`ConversableAgent`, `initiate_chat`) — it is retired at v1.0.
- Agents are async. Use `await agent.ask(...)` and `await reply.ask(...)`.

## Conventions
- Do not use `from __future__ import annotations`.
- Public signatures accept `str | os.PathLike[str]`; use `pathlib.Path` internally.
- Prefer top-level imports; no imports inside functions.

## Docs & skills
- Install and use the AG2 skills: `npx skills add ag2ai/ag2-skills`.
- Beta docs: https://docs.ag2.ai/latest/docs/beta/agents/ (machine-readable index at /latest/llms.txt).
- When unsure of a signature, check the Beta docs before writing code.
- Do NOT trust generic docs indexers — they surface the retired classic API.
```

=== "Claude Code"
    ```bash
    # Claude Code reads CLAUDE.md — symlink it to a single source of truth
    ln -s AGENTS.md CLAUDE.md
    ```

=== "Cursor"
    ```bash
    # Cursor reads AGENTS.md, or project rules under .cursor/rules/
    mkdir -p .cursor/rules
    cp AGENTS.md .cursor/rules/ag2-beta.md
    ```

=== "Copilot"
    ```bash
    # GitHub Copilot reads .github/copilot-instructions.md
    mkdir -p .github
    cp AGENTS.md .github/copilot-instructions.md
    ```

## Per-tool summary

| Assistant | Skills | Beta docs | Project rules |
|---|---|---|---|
| **Claude Code** | `npx skills add ag2ai/ag2-skills` or copy to `~/.claude/skills/` | point at the docs URL | `CLAUDE.md` (symlink to `AGENTS.md`) |
| **Cursor** | `npx skills add ag2ai/ag2-skills` | point at the docs URL | `AGENTS.md` or `.cursor/rules/` |
| **Copilot / Codex / Windsurf** | `npx skills add ag2ai/ag2-skills` | point at the docs URL | `AGENTS.md` (Copilot: `.github/copilot-instructions.md`) |
| **Any agent** | paste `SKILL.md` contents into context | paste the docs (or `llms-full.txt`) | paste the rules above into your prompt |

## Tips & caveats

- **Start from the quickstart.** Have your assistant scaffold from `ag2-quickstart` rather than inventing structure — see [Agent Communication](agents.md) and [Model Configuration](model_configuration.md).
- **Prefer the documented high-level API** (`autogen.beta`, `autogen.beta.tools`, `autogen.beta.config`) over reaching into internals.
- **Always review generated code.** Models still drift toward legacy `autogen` patterns; verify imports come from `autogen.beta` and that calls are `await`ed.
- **Run the tests.** AG2 is async throughout — encourage your assistant to write and run [tests](testing.md) with `TestConfig` instead of hitting a live model.

!!! note "Going further"
    The skill catalog mirrors these docs section-for-section — `ag2-network-quickstart` for [multi-agent networks](network/overview.md), `ag2-structured-output` for [typed responses](structured_output.md), `ag2-evaluation` for the [eval framework](evaluation/evaluation.md), and more. Install the set once and your assistant can pull in whichever it needs.
