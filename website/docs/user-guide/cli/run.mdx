---
sidebarTitle: Run & Chat
title: "ag2 run & ag2 chat"
description: "Execute agents from the terminal or start interactive chat sessions."
---

## ag2 run

Execute an agent from a Python file or YAML config with a single message.

```bash
ag2 run <agent_file> --message "Your prompt here"
```

### Options

| Flag | Description |
|------|-------------|
| `--message`, `-m` | The message to send to the agent |
| `--max-turns` | Maximum conversation turns (default: 10) |
| `--verbose`, `-v` | Show detailed agent output |
| `--json` | Output structured JSON result |

### Examples

Run an agent with a message:

```bash
ag2 run my_agent.py --message "What is the weather in SF?"
```

Pipe input from stdin:

```bash
echo "Summarize this" | ag2 run my_agent.py
```

Get structured JSON output:

```bash
ag2 run my_agent.py --message "Hello" --json
```

### YAML Config

You can define agents in YAML instead of Python:

```yaml title="config.yaml"
llm:
  model: gpt-4o-mini

agents:
  - name: assistant
    system_message: "You are a helpful research assistant."
```

```bash
ag2 run config.yaml --message "Research quantum computing"
```

---

## ag2 chat

Start an interactive terminal chat session with an agent or team.

```bash
ag2 chat <agent_file>
```

### Options

| Flag | Description |
|------|-------------|
| `--model` | Create an ad-hoc agent with this model |
| `--system` | System message for ad-hoc agent |
| `--verbose`, `-v` | Show detailed output |
| `--max-turns` | Maximum turns per exchange (default: 10) |

### Examples

Chat with an agent from a file:

```bash
ag2 chat my_agent.py
```

Create a quick ad-hoc agent:

```bash
ag2 chat --model gpt-4o-mini --system "You are a Python expert"
```

### Session Commands

During a chat session, you can use these commands:

| Command | Description |
|---------|-------------|
| `/quit` | End the session |
| `/cost` | Show token usage and cost |
| `/history` | Show conversation history |

### Session Recording

All chat sessions are automatically saved to `~/.ag2/sessions/` for later replay and debugging. Use [`ag2 replay`](replay) to review past sessions.
