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
│   └── web_search.py       # Example tool stub
├── config/
│   └── llm.yaml            # LLM configuration
├── tests/
│   └── test_agents.py      # Starter test
└── main.py                 # Entry point (compatible with ag2 run)
```

Options:
```bash
# Use a specific template
ag2 create project my-app --template research-team
ag2 create project my-app --template rag-chatbot
ag2 create project my-app --template fullstack-agentic

# Generate from description (AI-powered, requires API key)
ag2 create project --from-description "A Slack bot that monitors channels
  and summarizes discussions daily"
```

**Templates:** `blank` (default), `research-team`, `rag-chatbot`, `fullstack-agentic`

### `ag2 create agent` — Single agent scaffold

```bash
ag2 create agent researcher --tools web-search,arxiv
```

Creates `agents/researcher.py` with agent boilerplate and commented tool
registration stubs.

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

Requires an API key (reads from env: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
or `GOOGLE_API_KEY`).

### `ag2 create tool` — Tool scaffold

```bash
ag2 create tool stock-price --description "Fetch real-time stock prices"
```

Creates `tools/stock_price.py` with a `@tool` decorated function stub.

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

Creates `teams/code_review.py` (or `code_review.py` if no `teams/` dir)
with group chat orchestration boilerplate.

**Patterns:** `auto` (default), `round-robin`, `random`, `manual`

### `ag2 create artifact` — Artifact scaffold for the AG2 registry

```bash
ag2 create artifact template my-template
ag2 create artifact tool web-scraper
ag2 create artifact dataset eval-bench
ag2 create artifact agent research-analyst
ag2 create artifact skills my-framework
ag2 create artifact bundle starter-kit
```

Scaffolds a complete artifact directory ready for authoring and publishing
to the AG2 registry (`ag2ai/resource-hub`).

**Arguments:**
- `artifact_type` — one of: `template`, `tool`, `dataset`, `agent`, `skills`, `bundle`
- `name` — artifact name (e.g. `my-template`)

**Options:**
- `--output` / `-o` — parent directory for output (default: current directory)

Each artifact type generates a different directory structure with an
`artifact.json` manifest and placeholder skills. Publish with
`ag2 publish artifact <path>`.

## AI-Powered Generation (`--from-description`)

Available on `create project` and `create agent`. When provided:

1. **Analyze the description** — extract required capabilities (web search,
   API access, file I/O, specific domains)
2. **Select appropriate agent pattern** — single agent or group chat based
   on the task complexity
3. **Generate tool stubs** — for each capability needed
4. **Generate agent code** — with proper system messages, tool registration,
   and orchestration
5. **Generate tests** — basic test file

The generation uses an LLM call (auto-detects from env: `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`).
