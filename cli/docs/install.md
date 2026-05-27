# ag2 install

> Install skills, templates, tools, datasets, agents, and bundles into your project — the AI-native package manager.

## Problem

AI coding agents (Claude Code, Cursor, Copilot, etc.) are powerful but start
every project with zero context. Developers manually write rules, copy patterns
from docs, and configure tools by hand. There's no way to bootstrap an AI
agent's knowledge about a framework, project, or domain.

Meanwhile, project scaffolding tools (Cookiecutter, Copier, create-next-app)
generate code but don't give AI agents any understanding of what they generated.
And MCP servers require manual configuration per IDE.

`ag2 install` solves both: every artifact ships code *and* skills that make AI
agents understand the code. One command bootstraps both your project and your
AI assistant.

## Core Concept

**Every artifact bundles skills.** This is what makes `ag2 install` different
from npm, pip, or any existing package manager:

- Install a **template** → get a project scaffold + skills that teach your AI
  agent the architecture, conventions, and extension patterns
- Install a **tool** → get runtime code + skills that teach your AI agent how
  to register, test, and debug it
- Install a **dataset** → get data + skills that teach your AI agent the schema
  and how to build evals
- Install an **agent** → get a pre-built Claude Code subagent with its own
  tools, skills, and memory

## Artifact Types

| Type | What it is | What gets installed |
|------|-----------|---------------------|
| **Skills** | Knowledge for AI agents (any framework) | IDE rules/skills + AGENTS.md |
| **Templates** | Project scaffolds + skills | Files + skills + dependencies |
| **Tools** | AG2 functions or MCP servers + skills | Source + IDE config + skills |
| **Datasets** | Data references + skills | Metadata + data + eval registration |
| **Agents** | Pre-built Claude Code subagents | Agent def + MCP servers + skills |
| **Bundles** | Curated collections of artifacts | All of the above, composed |

## Commands

```bash
# Skills — install AI agent knowledge (universal, not AG2-only)
ag2 install skills                          # AG2 skills (default)
ag2 install skills ag2 fastapi react        # Multiple packs
ag2 install skills --name build-group-chat  # Single skill

# Templates — scaffold a project with full AI context
ag2 install template fullstack-agentic-app
ag2 install template mcp-server-python --var project_name=my-server
ag2 install template research-pipeline --preview

# Tools — add capabilities (AG2 functions or MCP servers)
ag2 install tool web-search                 # AG2 tool function
ag2 install tool github-mcp                 # MCP server (auto-configures IDE)

# Datasets — add evaluation and training data
ag2 install dataset agent-eval-bench        # Inline sample only
ag2 install dataset agent-eval-bench --full # Download remote files too

# Agents — install pre-built Claude Code subagents
ag2 install agent research-analyst
ag2 install agent code-reviewer

# Bundles — curated collections
ag2 install bundle customer-service-starter

# Install from a local path or GitHub URL
ag2 install from ./local-artifact
ag2 install from github.com/someuser/my-skills

# Discovery and management
ag2 install search "web scraping"           # Search across all types
ag2 install list [type]                     # List available artifacts
ag2 install list installed                  # Show what's installed
ag2 install update                          # Update installed artifacts
ag2 install uninstall <name>               # Remove an artifact

# Options available on all subcommands
--target claude,cursor                      # Specific IDE targets
--target all                                # All 18+ IDE targets
--project-dir ./my-project                  # Target directory
```

## Artifact Repository

All artifacts are hosted at `github.com/ag2ai/resource-hub`:

```
ag2ai/resource-hub/
├── registry.json                    # Auto-generated global index
├── schema/
│   └── v1.json                      # JSON Schema for artifact.json
│
├── skills/
│   ├── ag2/                         # AG2 framework
│   │   ├── artifact.json
│   │   ├── rules/
│   │   │   └── imports/SKILL.md
│   │   ├── skills/
│   │   │   └── build-group-chat/SKILL.md
│   │   ├── agents/
│   │   │   └── ag2-architect/SKILL.md
│   │   └── commands/
│   │       └── explain-flow/SKILL.md
│   ├── crewai/                      # Other frameworks
│   ├── langchain/
│   ├── fastapi/
│   ├── react/
│   └── docker/
│
├── templates/
│   ├── fullstack-agentic-app/
│   │   ├── artifact.json
│   │   ├── scaffold/                # Project files (any language)
│   │   └── skills/                  # Template-specific skills
│   ├── mcp-server-python/
│   ├── mcp-server-typescript/
│   └── research-pipeline/
│
├── tools/
│   ├── web-search/                  # AG2 tool function
│   │   ├── artifact.json
│   │   ├── src/
│   │   └── skills/
│   ├── github-mcp/                  # MCP server
│   │   ├── artifact.json
│   │   ├── src/
│   │   └── skills/
│   └── slack-mcp/
│
├── datasets/
│   ├── agent-eval-bench/
│   │   ├── artifact.json
│   │   ├── data/                    # Small inline samples
│   │   └── skills/
│   └── tool-calling-bench/
│
├── agents/
│   ├── research-analyst/
│   │   ├── artifact.json
│   │   ├── agent.md                 # Claude Code agent definition
│   │   ├── mcp/                     # Bundled MCP servers
│   │   └── skills/
│   └── code-reviewer/
│
├── bundles/
│   ├── customer-service-starter/
│   │   └── artifact.json
│   └── research-assistant/
│
└── LICENSE
```

## Artifact Specification

### Universal Manifest: `artifact.json`

Every artifact has this core schema:

```json
{
  "$schema": "https://ag2ai.github.io/resource-hub/schema/v1.json",
  "name": "my-artifact",
  "type": "skills | template | tool | dataset | agent | bundle",
  "display_name": "My Artifact",
  "description": "What this artifact provides",
  "version": "1.0.0",
  "authors": ["ag2ai"],
  "license": "Apache-2.0",
  "tags": ["tag1", "tag2"],
  "requires": {
    "python": ">=3.10"
  },
  "skills": {
    "dir": "skills/",
    "auto_install": true
  },
  "depends": ["skills/ag2", "tools/web-search"]
}
```

### Skills Format

Skills follow the **Agent Skills Open Standard** (agentskills.io). Each skill
is a directory with a `SKILL.md` file:

```
build-group-chat/
├── SKILL.md              # Required: frontmatter + instructions
├── references/           # Optional: supporting docs
└── assets/               # Optional: templates, examples
```

SKILL.md frontmatter (Agent Skills standard + extensions):

```yaml
---
name: build-group-chat
description: Build an AG2 handoff-driven workflow with DefaultPattern
license: Apache-2.0
compatibility: python>=3.10, ag2>=0.6
metadata:
  category: skill
  framework: ag2
---
```

Skills are not limited to AG2 — any framework or technology can have a skill
pack. The install targets transform each SKILL.md into the native format for
18+ IDEs:

| Target | Format | Key Fields |
|--------|--------|-----------|
| Claude Code | `.claude/skills/{name}/SKILL.md` | `name`, `description`, `user-invocable` |
| Cursor | `.cursor/rules/{name}/RULE.md` | `description`, `globs`, `alwaysApply` |
| Copilot | `.github/instructions/{name}.instructions.md` | `applyTo` |
| Windsurf | `.windsurf/rules/{name}.md` | `trigger`, `globs`, `description` |
| Cline | `.clinerules/{name}.md` | `paths` |
| Continue | `.continue/rules/{name}.md` | `name`, `globs`, `description` |
| AGENTS.md | `AGENTS.md` (always generated) | Universal cross-tool format |
| (13 more) | See `ag2 install list targets` | Tool-specific transforms |

### Template Manifest Extension

Templates are language-agnostic project scaffolds. The scaffold directory is
copied as-is, with minimal variable substitution (`{{ var }}` syntax).
Heavy customization is left to the AI agent using the bundled skills.

```json
{
  "template": {
    "scaffold": "scaffold/",
    "variables": {
      "project_name": {
        "prompt": "Project name",
        "default": "my-app",
        "transform": "slug"
      },
      "description": {
        "prompt": "Project description",
        "default": "An agentic application"
      }
    },
    "ignore": ["__pycache__", "*.pyc", ".git", "node_modules"],
    "post_install": [
      "uv sync",
      "npm install --prefix frontend"
    ]
  }
}
```

Design philosophy: **scaffold a working project, let the AI agent customize
it.** The bundled skills teach the AI the architecture, conventions, and
extension patterns — more powerful than any template variable system.

### Tool Manifest Extension

Tools come in two variants:

**AG2 Tool Functions** — Python functions with type annotations:

```json
{
  "tool": {
    "kind": "ag2",
    "source": "src/",
    "functions": [
      {"name": "web_search", "description": "Search the web"},
      {"name": "fetch_page", "description": "Extract text from a URL"}
    ],
    "requires": ["httpx>=0.27", "beautifulsoup4>=4.12"],
    "install_to": "tools/"
  }
}
```

**MCP Servers** — Model Context Protocol servers (Python or TypeScript):

```json
{
  "tool": {
    "kind": "mcp",
    "runtime": "python",
    "source": "src/",
    "entry_point": "server.py",
    "transport": "stdio",
    "mcp_config": {
      "command": "uv",
      "args": ["run", "--directory", "${toolDir}", "server.py"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "tools_provided": [
      {"name": "list_repos", "description": "List GitHub repositories"},
      {"name": "create_issue", "description": "Create a GitHub issue"}
    ],
    "requires": ["mcp[cli]>=1.0", "pygithub>=2.0"]
  }
}
```

MCP tool installation auto-configures the IDE:

| IDE | Config File | Key Format |
|-----|-------------|------------|
| Claude Code | `.mcp.json` | `mcpServers` |
| Cursor | `.cursor/mcp.json` | `mcpServers` |
| VS Code / Copilot | `.vscode/mcp.json` | `servers` |
| Claude Desktop | `~/Library/.../claude_desktop_config.json` | `mcpServers` |

### Dataset Manifest Extension

Datasets support inline small files and remote large files:

```json
{
  "dataset": {
    "inline": "data/",
    "remote": [
      {
        "name": "full-bench.jsonl",
        "url": "https://huggingface.co/datasets/ag2ai/eval-bench/resolve/main/full-bench.jsonl",
        "size": "150MB",
        "sha256": "a1b2c3..."
      }
    ],
    "format": "jsonl",
    "schema": {
      "fields": [
        {"name": "input", "type": "string", "description": "User input prompt"},
        {"name": "expected", "type": "string", "description": "Expected output"},
        {"name": "category", "type": "string", "description": "Task category"}
      ]
    },
    "splits": {
      "sample": "data/sample.yaml",
      "full": "full-bench.jsonl"
    },
    "eval_compatible": true
  }
}
```

### Agent Manifest Extension

Agents are pre-built Claude Code subagents:

```json
{
  "agent": {
    "source": "agent.md",
    "model": "sonnet",
    "tools": ["Read", "Grep", "Glob", "Bash", "WebSearch", "WebFetch"],
    "max_turns": 50,
    "memory": "project",
    "mcp_servers": {
      "bundled": ["mcp/web-crawler"],
      "external": []
    },
    "preload_skills": ["research-methodology"]
  }
}
```

The `agent.md` file uses Claude Code's native agent format:

```yaml
---
name: ag2-research-analyst
description: Autonomous research agent for investigating topics
model: sonnet
tools: Read, Grep, Glob, Bash, WebSearch, WebFetch
maxTurns: 50
memory: project
skills:
  - ag2-research-methodology
mcpServers:
  - web-crawler:
      type: stdio
      command: uv
      args: ["run", "--directory", "${CLAUDE_SKILL_DIR}/../mcp/web-crawler", "server.py"]
---

You are a research analyst. Your job is to thoroughly investigate topics
and produce well-sourced reports.

[full system prompt here...]
```

For v1, agents install only to Claude Code (`.claude/agents/`). Other IDEs
will be supported as they add custom agent features.

### Bundle Manifest Extension

Bundles compose multiple artifacts:

```json
{
  "bundle": {
    "artifacts": [
      {"ref": "templates/customer-service-app", "required": true},
      {"ref": "tools/slack-mcp", "required": false},
      {"ref": "datasets/customer-service-eval", "required": false},
      {"ref": "skills/customer-service-patterns", "required": true}
    ],
    "install_order": ["skills", "tools", "templates", "datasets"]
  }
}
```

Optional artifacts prompt the user during installation.

## Terminal Output

### Installing Skills

```bash
ag2 install skills ag2 fastapi
```

```
╭─ AG2 Install ─────────────────────────────────────────╮
│ Installing 2 skill packs                               │
╰────────────────────────────────────────────────────────╯

  Fetching ag2 v0.3.0 (22 skills)...
  Fetching fastapi v1.0.0 (8 skills)...

Installing AG2 Skills (22 items)

  ✓ Claude Code               22 files → .claude/skills/
  ✓ Cursor                    22 files → .cursor/rules/
  ✓ AGENTS.md                  1 file  → AGENTS.md

Installing FastAPI Skills (8 items)

  ✓ Claude Code                8 files → .claude/skills/
  ✓ Cursor                     8 files → .cursor/rules/
  ✓ AGENTS.md                  1 file  → AGENTS.md (appended)

Done. 30 skills installed for 2 target(s).
Your AI assistant now understands AG2 and FastAPI.
```

### Installing a Template

```bash
ag2 install template fullstack-agentic-app
```

```
╭─ AG2 Install ─────────────────────────────────────────╮
│ Template: Fullstack Agentic Application v1.0.0         │
│ A complete fullstack app with AG2 agents               │
╰────────────────────────────────────────────────────────╯

? Project name: my-app
? Project description: My agentic application

  ✓ Scaffolded project (24 files)
    ├── src/api/main.py
    ├── src/agents/assistant.py
    ├── src/frontend/package.json
    ├── pyproject.toml
    ├── Dockerfile
    └── ... (19 more)
  ✓ Installed 5 template skills → .claude/skills/
  ✓ Installed dependency: skills/ag2 (22 skills)

Running post-install...
  ✓ uv sync
  ✓ npm install --prefix src/frontend

Done. Your AI assistant understands this project's architecture.
Try: "Add a new agent endpoint for data analysis"
```

### Installing a Tool (MCP)

```bash
ag2 install tool github-mcp
```

```
╭─ AG2 Install ─────────────────────────────────────────╮
│ Tool: GitHub MCP Server v1.0.0                         │
│ GitHub operations via Model Context Protocol           │
╰────────────────────────────────────────────────────────╯

  ✓ Installed server → ./tools/github-mcp/
  ✓ Installed dependencies: mcp[cli]>=1.0, pygithub>=2.0
  ✓ Configured .mcp.json (Claude Code)
  ✓ Configured .cursor/mcp.json (Cursor)
  ✓ Installed 2 skills → .claude/skills/

MCP tools now available:
  list_repos        List GitHub repositories
  create_issue      Create a GitHub issue
  search_code       Search code across repos

Requires: GITHUB_TOKEN environment variable
```

### Installing an Agent

```bash
ag2 install agent research-analyst
```

```
╭─ AG2 Install ─────────────────────────────────────────╮
│ Agent: Research Analyst v1.0.0                         │
│ Autonomous research with web search and synthesis      │
╰────────────────────────────────────────────────────────╯

  ✓ Installed agent → .claude/agents/ag2-research-analyst.md
  ✓ Installed MCP server: web-crawler → .claude/agents/ag2-research-analyst/mcp/
  ✓ Installed 3 skills → .claude/skills/
  ✓ Installed dependency: skills/ag2 (22 skills)

Usage:
  Natural language:   "Use the research-analyst to investigate X"
  @-mention:          @ag2-research-analyst investigate X
  Session-wide:       claude --agent ag2-research-analyst

Model: sonnet | Memory: project | Max turns: 50
```

### Searching Artifacts

```bash
ag2 install search "web scraping"
```

```
╭─ AG2 Artifacts ─ Results for "web scraping" ──────────╮
│                                                        │
│  skills  web-scraping-patterns v1.0.0                  │
│          Web scraping best practices and patterns      │
│                                                        │
│  tool    web-search v1.2.0                             │
│          Search the web and extract page content       │
│                                                        │
│  tool    browser-mcp v0.5.0                            │
│          Browser automation via Playwright MCP          │
│                                                        │
│  agent   web-researcher v1.0.0                         │
│          Autonomous web research agent                  │
│                                                        │
│  bundle  web-research-starter v1.0.0                   │
│          Complete web research toolkit                  │
│                                                        │
╰────────────────────────────────────────────────────────╯

Install with: ag2 install <type> <name>
```

## Installation Semantics by Type

### Skills

1. Fetch skill pack from resource hub (or use bundled cache for AG2)
2. Transform each `SKILL.md` to target IDE's native format
3. Install to detected (or specified) IDE targets
4. Always generate/append AGENTS.md for universal compatibility

### Templates

1. Fetch template from resource hub
2. Prompt for variables (interactive or `--var key=value`)
3. Copy scaffold to current directory with variable substitution
4. Install bundled skills to IDE targets
5. Resolve and install dependency artifacts (`depends`)
6. Run post-install commands

### Tools (AG2)

1. Copy source to `./tools/<name>/`
2. Install Python dependencies
3. Install bundled skills to IDE targets
4. Print registration pattern

### Tools (MCP)

1. Copy server source to `./tools/<name>/`
2. Install dependencies
3. Auto-configure MCP in all detected IDEs (`.mcp.json`, `.cursor/mcp.json`, etc.)
4. Install bundled skills
5. Print available tool list

### Datasets

1. Download metadata and inline data to `./data/<name>/`
2. For `--full`: download remote files with progress bar and checksum verification
3. Install bundled skills
4. Register for `ag2 test eval` if `eval_compatible: true`

### Agents

1. Install `agent.md` → `.claude/agents/ag2-<name>.md`
2. If bundled MCP servers: install to `.claude/agents/ag2-<name>/mcp/`
3. Install bundled skills to `.claude/skills/`
4. Resolve and install dependencies

### Bundles

1. Fetch bundle manifest
2. Prompt for optional artifacts
3. Install each artifact in dependency order
4. Install bundle-level skills (if any)

### From (local path or URL)

`ag2 install from <source>` installs an artifact from outside the official
registry:

```bash
# Install from a local directory
ag2 install from ./my-local-artifact

# Install from a GitHub URL (coming soon)
ag2 install from github.com/someuser/my-artifact
```

**Local path**: the directory must contain an `artifact.json`. The command
reads the manifest, determines the artifact type, and delegates to the
appropriate type-specific installer. The local artifact is staged into the
client cache and then installed normally.

Supported types for local install: `skills`, `template`, `tool`, `agent`,
`dataset`.

**Options:**
- `--target` / `-t` — specific IDE target(s)
- `--project-dir` / `-d` — target project directory

## Registry Architecture

### v1: Git-Based (current)

- `registry.json` at repo root = full artifact index
- CLI fetches `registry.json` first (single HTTP request), caches locally
- Individual artifacts fetched on demand via GitHub API (sparse)
- AG2 skills bundled in CLI for offline fallback
- Cache at `~/.ag2/cache/artifacts/`

### Lockfile: `.ag2-artifacts.lock`

Tracks what's installed and at what version:

```json
{
  "installed": {
    "skills/ag2": {"version": "0.3.0", "installed_at": "2026-03-18T00:00:00Z"},
    "tools/github-mcp": {"version": "1.0.0", "installed_at": "2026-03-18T00:00:00Z"},
    "agents/research-analyst": {"version": "1.0.0", "installed_at": "2026-03-18T00:00:00Z"}
  }
}
```

Enables `ag2 install update` to check for newer versions.

### v2: API Registry (future)

- REST API for search, install metrics, ratings
- Automated security scanning
- Community submissions beyond PR-based workflow

## Implementation Notes

### Remote Fetching

Use `httpx` (already a dependency) to fetch from GitHub:

1. `registry.json` → `https://raw.githubusercontent.com/ag2ai/resource-hub/main/registry.json`
2. Individual files → GitHub Contents API or raw URLs
3. Full artifact directories → GitHub API tree endpoint or tarball

Cache strategy:
- `registry.json`: cache for 1 hour, force refresh with `--refresh`
- Artifact content: cache by version (immutable once cached)
- Bundled AG2 skills: always available offline

### Dependency Resolution

Simple DAG resolution — no complex version solving needed for v1:

1. Build dependency graph from `depends` fields
2. Topological sort
3. Install in order, skip already-installed at same version
4. Circular dependencies → error

### MCP Auto-Configuration

When installing an MCP tool, detect which IDEs are present and write
config to each:

```python
MCP_CONFIG_PATHS = {
    "claude": Path(".mcp.json"),
    "cursor": Path(".cursor/mcp.json"),
    "vscode": Path(".vscode/mcp.json"),
}
```

Each has slightly different format (VS Code uses `"servers"` not
`"mcpServers"`, supports `"inputs"` for secrets). The installer handles
the translation.

### Template Variable Substitution

Minimal engine — only `{{ var }}` replacement in files with `.tmpl`
extension (or explicitly listed). Processed files have `.tmpl` stripped.
Non-`.tmpl` files are copied verbatim.

This is intentionally simple. Complex customization is handled by the AI
agent using the bundled skills, not by a template engine.

### Upgrade from Bundled Skills

Current: AG2 skills bundled in `cli/src/ag2_cli/content/skills/`.
Future: skills live in `ag2ai/resource-hub` repo, CLI fetches remotely.

Transition path:
1. Keep bundled skills as offline fallback
2. Add remote fetching with cache
3. Remote takes priority when available
4. Bundled version updates with CLI releases

## Dependencies

- `httpx` ≥0.27 — already in CLI deps, used for artifact fetching
- `rich` ≥13.0 — already in CLI deps, used for progress and output
- `questionary` ≥2.0 — already in CLI deps, used for interactive prompts
- No new dependencies required
