# ag2 market

> Discover, install, and publish community agents, tools, and templates.

## Problem

As the AG2 ecosystem grows, developers need a way to discover and share
reusable agents, tools, and templates. Today, everything is copy-pasted
from docs and examples. There's no npm/pip for agents.

## Commands

```bash
# Search for community tools
ag2 market search "slack integration"
ag2 market search "web scraping" --type tool

# Browse categories
ag2 market browse
ag2 market browse --category "devops"

# Install a community package
ag2 market install ag2-community/slack-notifier
ag2 market install ag2-community/github-pr-reviewer

# Show package details
ag2 market info ag2-community/slack-notifier

# Publish your own
ag2 market publish ./my_tool --description "Real-time stock price fetcher"

# List installed community packages
ag2 market list

# Update all installed packages
ag2 market update
```

## Package Structure

A market package is a directory with:

```
my-slack-tool/
├── manifest.yaml       # Package metadata
├── tools/
│   └── slack.py        # AG2 tool definitions
├── agents/
│   └── notifier.py     # Optional: pre-built agent using the tools
├── tests/
│   └── test_slack.py   # Tests
├── README.md           # Documentation
└── requirements.txt    # Extra dependencies
```

```yaml
# manifest.yaml
name: slack-notifier
version: 1.0.0
description: Send Slack messages and monitor channels with AG2 agents
author: ag2-community
license: Apache-2.0
type: tool
tags: [slack, notifications, messaging]
requires:
  - slack-sdk>=3.0
tools:
  - send_message
  - list_channels
  - read_channel_history
agents:
  - channel_monitor
```

## Registry

The marketplace is backed by the `ag2ai/artifacts` repository:

```
ag2ai/artifacts/
├── skills/          # IDE skills (existing)
├── templates/       # Project templates (existing)
└── market/          # Community packages
    ├── index.json   # Package index
    └── packages/
        ├── slack-notifier/
        ├── github-tools/
        ├── web-scraper/
        └── ...
```

For v1, packages are submitted via PR to the artifacts repo.
Future: dedicated registry service with API.

## Search & Discovery

```bash
ag2 market search "database"
```

```
╭─ AG2 Market ─ Results for "database" ──────────────╮
│                                                     │
│  📦 pg-query-tools (v1.2.0)                        │
│     PostgreSQL query and schema inspection tools    │
│     ⭐ 142 installs | by: ag2-community             │
│                                                     │
│  📦 mongodb-agent (v0.8.0)                         │
│     MongoDB CRUD operations with natural language   │
│     ⭐ 89 installs | by: datatools-org              │
│                                                     │
│  📦 redis-cache-tool (v1.0.0)                      │
│     Redis caching tools for agent memory            │
│     ⭐ 67 installs | by: ag2-community              │
│                                                     │
╰─────────────────────────────────────────────────────╯
```

## Installation

```bash
ag2 market install ag2-community/slack-notifier
```

```
Installing slack-notifier v1.0.0...
  ✓ Downloaded from ag2ai/artifacts
  ✓ Installed extra dependencies: slack-sdk>=3.0
  ✓ Tools available: send_message, list_channels, read_channel_history
  ✓ Agent available: channel_monitor

Usage:
  from ag2_market.slack_notifier import send_message_tool
  send_message_tool.register_tool(my_agent)
```

Packages install to `~/.ag2/market/` and are importable from `ag2_market.*`.

## Implementation Notes

### v1: Git-Based Registry
- Index file in `ag2ai/artifacts/market/index.json`
- Packages stored as directories in the same repo
- Install = clone/download specific directory
- Publish = submit PR

### v2 (future): API Registry
- REST API for search, install, publish
- Usage metrics (install counts, ratings)
- Automated security scanning of published packages
- Version management with semver

### Dependency Resolution
When a market package has `requirements.txt`, the install command:
1. Creates a virtual environment or installs to the current one
2. Runs `pip install -r requirements.txt`
3. Validates that tools are importable
