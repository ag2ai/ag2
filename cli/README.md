# AG2 CLI

Build, run, test, and deploy multi-agent applications from the terminal.

```
pip install ag2-cli
```

```
    ___   ______ ___
   /   | / ____/|__ \
  / /| |/ / __  __/ /
 / ___ / /_/ / / __/
/_/  |_\____/ /____/

  Build, run, test, and deploy multi-agent applications
```

## Commands

| Command | Description | Status |
|---------|-------------|--------|
| `ag2 install skills` | Install AG2 skills into your IDE | ✅ Ready |
| `ag2 install templates` | Install project templates from artifacts repo | 🔜 In Progress |
| `ag2 install list` | List available skills, templates, targets | ✅ Ready |
| `ag2 install uninstall` | Remove installed skill files | ✅ Ready |
| `ag2 run` | Run an agent or team from a file | 📋 [Design](docs/run.md) |
| `ag2 chat` | Interactive terminal chat with agents | 📋 [Design](docs/run.md) |
| `ag2 serve` | Expose agents as REST/MCP/A2A endpoints | 📋 [Design](docs/serve.md) |
| `ag2 create` | Scaffold projects, agents, tools, teams | 📋 [Design](docs/create.md) |
| `ag2 test eval` | Run evaluation suites against agents | 📋 [Design](docs/test.md) |
| `ag2 test bench` | Standardized benchmarks | 📋 [Design](docs/test.md) |
| `ag2 doctor` | AI-powered diagnostics and profiling | 📋 [Design](docs/doctor.md) |
| `ag2 replay` | Replay, debug, and branch conversations | 📋 [Design](docs/replay.md) |
| `ag2 explore` | AI-powered codebase/API analysis | 📋 [Design](docs/explore.md) |
| `ag2 arena` | A/B test agent implementations | 📋 [Design](docs/arena.md) |
| `ag2 convert` | Migrate from CrewAI, LangChain, etc. | 📋 [Design](docs/convert.md) |
| `ag2 audit` | Security and safety scanning | 📋 [Design](docs/audit.md) |
| `ag2 watch` | Live monitoring dashboard | 📋 [Design](docs/watch.md) |
| `ag2 proxy` | Wrap CLIs/APIs/modules as AG2 tools | 📋 [Design](docs/proxy.md) |
| `ag2 market` | Community agent/tool marketplace | 📋 [Design](docs/market.md) |

## Quick Start

```bash
# Install skills into your IDE (auto-detects Cursor, Claude Code, etc.)
ag2 install skills

# Install for a specific target
ag2 install skills --target cursor

# List what's available
ag2 install list skills
ag2 install list targets
```

## Architecture

```
cli/
├── src/ag2_cli/
│   ├── app.py              # Main Typer application
│   ├── commands/            # Command implementations
│   │   ├── install.py       # ag2 install (skills, templates, list, uninstall)
│   │   ├── run.py           # ag2 run, ag2 chat
│   │   ├── create.py        # ag2 create (project, agent, tool, team)
│   │   ├── serve.py         # ag2 serve
│   │   └── test.py          # ag2 test (eval, bench)
│   ├── install/             # Install subsystem
│   │   ├── registry.py      # Content pack loading
│   │   └── targets/         # IDE target implementations
│   │       ├── base.py      # DirectoryTarget, SingleFileTarget
│   │       ├── claude.py    # Claude Code target
│   │       └── copilot.py   # GitHub Copilot target
│   ├── content/             # Bundled content packs
│   │   └── skills/          # Skills pack (rules, skills, agents, commands)
│   └── ui/                  # Rich UI components
│       ├── logo.py          # AG2 banner
│       ├── console.py       # Shared console instances
│       └── theme.py         # Color theme
├── docs/                    # Use case design documents
└── tests/
```

## Tech Stack

- **[Typer](https://typer.tiangolo.com/)** — CLI framework (type-hint driven, built on Click)
- **[Rich](https://rich.readthedocs.io/)** — Terminal formatting (tables, panels, progress bars, syntax highlighting)
- **[questionary](https://github.com/tmbo/questionary)** — Interactive prompts (multi-select, fuzzy search)

## Development

```bash
cd cli
pip install -e ".[dev]"
ag2 --version
```

## Artifacts Repository

Skills, templates, and marketplace packages are hosted at
[github.com/ag2ai/artifacts](https://github.com/ag2ai/artifacts).

## License

Apache-2.0
