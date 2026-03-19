# ag2 create

> Scaffold AG2 projects, agents, tools, and teams from the command line.

## Problem

Getting started with AG2 means reading docs, copying examples, and wiring
things together manually. CrewAI has `crewai create crew`, LangGraph has
`langgraph new`. AG2 needs the same, but better — with AI-powered generation.

## Commands

### `ag2 create project` — Full project scaffold

```bash
ag2 create project my-research-bot
```

Creates:
```
my-research-bot/
├── pyproject.toml          # Dependencies with ag2 pre-configured
├── .env.example            # API key placeholders
├── .gitignore
├── agents/
│   ├── __init__.py
│   └── assistant.py        # Starter agent
├── tools/
│   ├── __init__.py
│   └── web_search.py       # Example tool
├── config/
│   └── llm.yaml            # LLM configuration
├── tests/
│   └── test_agents.py      # Starter test
└── main.py                 # Entry point (compatible with ag2 run)
```

Options:
```bash
# Use a specific template
ag2 create project my-app --template fullstack-agentic
ag2 create project my-app --template research-team
ag2 create project my-app --template rag-chatbot

# Generate from description (AI-powered, requires API key)
ag2 create project --from-description "A Slack bot that monitors channels
  and summarizes discussions daily"
```

### `ag2 create agent` — Single agent scaffold

```bash
ag2 create agent researcher --tools web-search,arxiv
```

Creates `agents/researcher.py`:
```python
from autogen import AssistantAgent, LLMConfig

config = LLMConfig(api_type="openai", model="gpt-4o")

with config:
    researcher = AssistantAgent(
        name="researcher",
        system_message="You are a thorough researcher...",
    )

# Tool registration
from tools.web_search import web_search_tool
from tools.arxiv import arxiv_tool
web_search_tool.register_tool(researcher)
arxiv_tool.register_tool(researcher)
```

**AI-powered generation:**
```bash
ag2 create agent --from-description "An agent that monitors Hacker News
  for AI papers and sends weekly Slack digests with summaries"
```

This uses an LLM to:
1. Determine what tools are needed (HN API, Slack webhook, summarization)
2. Generate the agent code with proper system message
3. Generate tool stubs for each required tool
4. Wire everything together

### `ag2 create tool` — Tool scaffold

```bash
ag2 create tool stock-price --description "Fetch real-time stock prices"
```

Creates `tools/stock_price.py`:
```python
from autogen.tools import tool

@tool(name="stock_price", description="Fetch real-time stock prices")
def stock_price(symbol: str) -> str:
    """Fetch the current stock price for a given ticker symbol.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "GOOGL")

    Returns:
        Current price and daily change as formatted string.
    """
    # TODO: Implement — connect to a stock price API
    raise NotImplementedError("Connect to your preferred stock API")
```

Options:
```bash
# Generate from an OpenAPI spec
ag2 create tool --from-openapi https://api.example.com/openapi.json

# Generate tools from a Python module's public functions
ag2 create tool --from-module pandas --functions read_csv,describe,to_json
```

### `ag2 create team` — Multi-agent team scaffold

```bash
ag2 create team code-review \
  --pattern round-robin \
  --agents reviewer,tester,merger
```

Creates `teams/code_review.py`:
```python
from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import run_group_chat, RoundRobinPattern

config = LLMConfig(api_type="openai", model="gpt-4o")

with config:
    reviewer = AssistantAgent(
        name="reviewer",
        system_message="You review code for correctness and style...",
    )
    tester = AssistantAgent(
        name="tester",
        system_message="You write tests for the reviewed code...",
    )
    merger = AssistantAgent(
        name="merger",
        system_message="You make final approval decisions...",
    )

async def main(message: str):
    result = await run_group_chat(
        pattern=RoundRobinPattern([reviewer, tester, merger]),
        messages=message,
        max_rounds=10,
    )
    return result
```

## AI-Powered Generation (`--from-description`)

This is the differentiator. When a user provides `--from-description`:

1. **Analyze the description** — extract required capabilities (web search,
   API access, file I/O, specific domains)
2. **Select appropriate agent pattern** — single agent, group chat, sequential,
   or swarm based on the task
3. **Generate tool stubs** — for each capability that doesn't have a built-in
4. **Generate agent code** — with proper system messages, tool registration,
   and orchestration
5. **Generate tests** — basic evaluation cases

The generation uses AG2's own agents internally:
- An architect agent that designs the system
- A coder agent that generates the Python code
- A reviewer agent that validates the output

This requires an API key (reads from env: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
etc.).

## Implementation Notes

### Template System
Project templates are Jinja2-based (AG2 already uses Jinja2 for its existing
template system in `/templates/`). Templates are stored in the artifacts repo
(`ag2ai/artifacts/templates/`) and cached locally.

### Built-in Tool Registry
`ag2 create agent --tools web-search` needs to know what tools exist. Maintain
a registry mapping tool names to:
- Built-in AG2 tools (web search, code execution, etc.)
- Common external tools (Slack, GitHub, HN, etc.)
- User-installed tools

### Interactive Mode
When run without arguments, `ag2 create` could launch an interactive wizard
using `questionary`:

```
? What do you want to create? (Use arrow keys)
  ❯ Project — full project scaffold
    Agent — single agent
    Tool — new tool
    Team — multi-agent team

? Project name: my-research-bot
? Select a template:
  ❯ blank — minimal starter
    research-team — web research + report writing
    rag-chatbot — RAG with vector database
    fullstack-agentic — full-stack app with agent backend
? Configure LLM provider:
  ❯ OpenAI (gpt-4o)
    Anthropic (claude-sonnet)
    Google (gemini-2.0-flash)
    Ollama (local)
```
