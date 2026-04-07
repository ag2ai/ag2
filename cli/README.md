# AG2 CLI

Build, run, test, and deploy multi-agent applications from the terminal.

```
pip install ag2-cli
```

```
██                          ██
  ██                      ██
    ██    ██████████    ██
      ████          ████
    ██                  ██
  ██    ██████████████    ██
  ██  ████  ██████  ████  ██
      ██████████████████

  ████      ██████  ██████
██    ██  ██              ██
████████  ██  ████    ████
██    ██  ██    ██  ██
██    ██    ████    ████████

  Build, run, test, and deploy multi-agent applications
```

## Commands

| Command | Description | Status |
|---------|-------------|--------|
| `ag2 install skills` | Install AG2 skills into your IDE | ✅ Ready |
| `ag2 install template` | Install project templates from resource hub | ✅ Ready |
| `ag2 install tool` | Install AG2 tools and MCP servers | ✅ Ready |
| `ag2 install dataset` | Install datasets for evaluation | ✅ Ready |
| `ag2 install agent` | Install pre-built Claude Code subagents | ✅ Ready |
| `ag2 install bundle` | Install curated artifact collections | ✅ Ready |
| `ag2 install list` | List available skills, templates, targets | ✅ Ready |
| `ag2 install search` | Search for artifacts across all types | ✅ Ready |
| `ag2 install uninstall` | Remove installed artifacts | ✅ Ready |
| `ag2 run` | Run an agent or team from a file | ✅ Ready |
| `ag2 chat` | Interactive terminal chat with agents | ✅ Ready |
| `ag2 serve` | Expose agents as REST/MCP/A2A endpoints | ✅ Ready |
| `ag2 create` | Scaffold projects, agents, tools, teams | ✅ Ready |
| `ag2 test eval` | Run evaluation suites against agents | ✅ Ready |
| `ag2 test bench` | Standardized benchmarks | 🔜 Coming Soon |
| `ag2 replay` | Replay, debug, and branch conversations | ✅ Ready |
| `ag2 arena` | A/B test agent implementations | ✅ Ready |
| `ag2 proxy` | Wrap CLIs/APIs/modules as AG2 tools | ✅ Ready |
| `ag2 publish` | Publish artifacts to the registry | ✅ Ready |

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
[github.com/ag2ai/resource-hub](https://github.com/ag2ai/resource-hub).

## License

Apache-2.0
