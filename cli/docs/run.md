# ag2 run / ag2 chat

> Run agents from the terminal. Chat interactively with multi-agent teams.

## Problem

Today, running an AG2 agent requires writing a Python script with boilerplate:
import the agent classes, configure LLMs, create instances, call `initiate_chat()`,
handle output. There's no way to just *run* an agent quickly.

Every competing framework has this: CrewAI has `crewai run`, LangGraph has
`langgraph dev`. AG2 has nothing.

## Commands

### `ag2 run` — Execute an agent or team

```bash
# Run a Python file that exports agents
ag2 run my_team.py

# Pass an input message
ag2 run my_team.py --message "Research quantum computing trends"

# Run from a declarative YAML config
ag2 run team.yaml

# Pipe input from stdin
echo "Summarize this document" | ag2 run my_agent.py

# Verbose mode — show tool calls, token counts, speaker transitions
ag2 run my_team.py -m "Analyze Q4 earnings" --verbose

# JSON output for scripting and pipelines
ag2 run my_team.py -m "Summarize trends" --json

# Limit conversation turns (default: 10)
ag2 run my_team.py -m "Research topic" --max-turns 5
```

**Options:**
- `--message` / `-m` — input message (required unless piped via stdin)
- `--verbose` / `-V` — show detailed agent activity
- `--json` — output result as structured JSON (suppresses live rendering)
- `--max-turns` — maximum conversation turns (default: 10)

**JSON output format** (`--json`):
```json
{
  "output": "The agent's final response text",
  "turns": 3,
  "elapsed": 4.52,
  "agent_names": ["researcher", "writer"],
  "errors": [],
  "cost": { ... }
}
```

**Agent file convention**: `ag2 run` looks for one of these in the target file:
1. A function named `main()` — called directly
2. A variable named `agent` or `team` — starts a chat with it
3. A variable named `agents` (list) — creates a GroupChat and runs it

Example `my_team.py`:
```python
from autogen import AssistantAgent, UserProxyAgent, LLMConfig

config = LLMConfig(api_type="openai", model="gpt-4o")

with config:
    researcher = AssistantAgent("researcher", system_message="You research topics thoroughly.")
    critic = AssistantAgent("critic", system_message="You critically evaluate research.")

agent = researcher  # ag2 run picks this up
```

### `ag2 run` with YAML config

```yaml
# team.yaml
llm:
  model: gpt-4o

agents:
  - name: researcher
    system_message: "You research topics thoroughly using web search."
    tools: [web-search]
  - name: writer
    system_message: "You write clear, concise reports from research."

team:
  pattern: auto
  max_rounds: 10
```

```bash
ag2 run team.yaml --message "Write a report on AI agent frameworks in 2026"
```

### `ag2 chat` — Interactive terminal chat

```bash
# Chat with an agent from a file
ag2 chat my_agent.py

# Quick ad-hoc chat (no file needed)
ag2 chat --model gpt-4o --system "You are a Python expert"

# Chat with a multi-agent team — watch agents collaborate in real-time
ag2 chat my_team.py --verbose

# Limit turns per message (default: 10)
ag2 chat my_team.py --max-turns 5
```

**Options:**
- `--model` / `-M` — LLM model for ad-hoc chat (no agent file needed)
- `--system` / `-s` — system prompt for ad-hoc chat
- `--verbose` / `-V` — show detailed agent activity
- `--max-turns` — maximum turns per message (default: 10)

**Session commands** (type these during a chat session):
- `/quit`, `/exit`, `/q` — end the session
- `/cost` — show total accumulated token cost
- `/history` — show the number of turns so far

## Terminal UI Design

The chat interface uses Rich for beautiful output:

```
╭─ AG2 Chat ─────────────────────────────────────────╮
│ Team: research-crew (3 agents)                      │
│ Model: gpt-4o | Pattern: auto                       │
╰─────────────────────────────────────────────────────╯

You: Research the latest advances in protein folding

  ┌ researcher ──────────────────────────────────────┐
  │ I'll search for recent protein folding advances. │
  │                                                  │
  │ 🔧 web_search("protein folding 2026 advances")  │
  │ ✓ 8 results found                               │
  │                                                  │
  │ Based on my research, here are the key...        │
  └──────────────────────────────────────────────────┘

  ┌ critic ──────────────────────────────────────────┐
  │ Good overview, but missing the AlphaFold 3       │
  │ developments from DeepMind. Let me add...        │
  └──────────────────────────────────────────────────┘

 tokens: 2,847 | cost: $0.02 | turns: 3

You: █
```

Features:
- Agent messages in colored, labeled panels (Rich)
- Tool calls shown inline with status indicators
- Running token/cost counter
- Speaker transition indicators in group chats
- Session commands: `/cost`, `/history`, `/quit`
- Multi-turn conversation continuity (persistent user proxy)

## Implementation Notes

### Agent Discovery
The `ag2 run` command needs a convention for finding runnable agents in a file.
The discovery order:
1. `main()` function → call it
2. `agent` variable → `initiate_chat()` with the input message
3. `team` variable → same
4. `agents` list → wrap in GroupChat with sensible defaults
5. Any single `ConversableAgent` subclass instance → use it

### YAML Config Loader
The YAML config maps to AG2's Python API:
- `llm` → `LLMConfig(...)`
- `agents[].tools` → look up from built-in tool registry or local `tools/` dir
- `team.pattern` → `AutoPattern`, `RoundRobinPattern`, etc.

### Streaming
Use AG2's event system (`autogen.events`) and the beta streaming API
(`autogen.beta.stream`) for real-time output. The Rich console handles
progressive rendering.

## Dependencies
- `ag2` (the main framework) — required for agent execution
- `rich` — already in the CLI deps for terminal UI
- `prompt_toolkit` or `questionary` — for input history and completion in chat mode
