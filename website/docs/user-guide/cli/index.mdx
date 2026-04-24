---
sidebarTitle: CLI Overview
title: "AG2 CLI"
description: "A command-line interface for building, running, testing, and deploying multi-agent applications with AG2."
---

The AG2 CLI (`ag2`) is a terminal tool for the full agent development lifecycle — from scaffolding projects to running, testing, serving, and publishing agents.

## Installation

```bash
pip install ag2-cli
```

Verify it's installed:

```bash
ag2 --help
```

## Commands at a Glance

| Command | Description |
|---------|-------------|
| [`ag2 run`](run) | Execute an agent with a message |
| [`ag2 chat`](run#ag2-chat) | Interactive terminal chat session |
| [`ag2 create`](create) | Scaffold projects, agents, tools, and teams |
| [`ag2 serve`](serve) | Expose agents as REST, MCP, or A2A endpoints |
| [`ag2 test`](test) | Run evaluation suites and benchmarks |
| [`ag2 install`](install) | Install skills, templates, tools, and more |
| [`ag2 replay`](replay) | Replay and debug past sessions |
| [`ag2 arena`](arena) | A/B test agent implementations |
| [`ag2 proxy`](proxy) | Wrap CLIs, APIs, or modules as AG2 tools |
| [`ag2 publish`](publish) | Publish artifacts to the AG2 registry |

## Quick Start

Create a new project and run it:

```bash
ag2 create project my-app --template blank
cd my-app
ag2 run main.py --message "Hello, agent!"
```

Or run an existing agent file:

```bash
ag2 run my_agent.py --message "Summarize this document"
```

Start an interactive chat session:

```bash
ag2 chat my_agent.py
```

## Agent Discovery

Most commands (`run`, `chat`, `serve`, `test`) need to find an agent in your Python file. The CLI looks for these in order:

1. A `main()` function
2. A module-level variable named `agent` or `team`
3. A module-level list named `agents`
4. Any single `ConversableAgent` instance

You can also use YAML config files instead of Python:

```yaml title="agent.yaml"
llm:
  model: gpt-4o-mini

agents:
  - name: assistant
    system_message: "You are a helpful assistant."
```

```bash
ag2 run agent.yaml --message "Hello"
```
