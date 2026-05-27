# ag2 serve

> Expose any AG2 agent as a REST API, MCP server, or A2A endpoint with one command.

## Problem

AG2 already has MCP server support (`python -m autogen.mcp`) and A2A integration,
but they're buried in the library. There's no unified way to take an agent and
make it available to external consumers.

LangGraph has `langgraph dev` for local serving. Haystack has `hayhooks run`.
AG2 agents are stuck in Python scripts.

## Commands

```bash
# Serve as a REST API (default)
ag2 serve my_team.py --port 8000

# Serve as an MCP server — instantly available to Claude Desktop, Cursor, etc.
ag2 serve my_agent.py --protocol mcp

# Serve as an A2A endpoint — other agents can discover and call yours
ag2 serve my_agent.py --protocol a2a --port 9000

# Serve with auto-reload during development
ag2 serve my_team.py --reload

# Expose to the internet via ngrok tunnel
ag2 serve my_team.py --ngrok

# Serve with a web playground for testing (coming soon)
ag2 serve my_team.py --playground
```

## Protocols

### REST API (default)

Generates a FastAPI app with these endpoints:

```
POST /chat          — send a message, get a response
POST /chat/stream   — SSE stream of agent activity
GET  /agents        — list available agents and their descriptions
GET  /health        — health check
GET  /docs          — auto-generated OpenAPI docs (Swagger UI)
```

**Request:**
```json
{
  "message": "Research quantum computing trends",
  "session_id": "optional-session-id",
  "max_turns": 10
}
```

**Streamed response** (SSE):
```
event: agent_message
data: {"agent": "researcher", "content": "I'll search for recent..."}

event: tool_call
data: {"agent": "researcher", "tool": "web_search", "args": {"query": "..."}}

event: tool_result
data: {"agent": "researcher", "tool": "web_search", "result": "..."}

event: agent_message
data: {"agent": "critic", "content": "Good overview, but..."}

event: done
data: {"turns": 4, "tokens": 3200, "cost": 0.03}
```

### MCP Server

Wraps the agent as a Model Context Protocol server using FastMCP. Exposes
two MCP tools:
- `chat` — send a message to the AG2 agent and get a response
- `list_agents` — list available agents and their orchestration type

```bash
ag2 serve my_agent.py --protocol mcp
# → MCP server running via SSE transport on http://localhost:8000
```

This means any AG2 agent can be instantly used from:
- Claude Desktop (add to `claude_desktop_config.json`)
- Cursor
- Windsurf
- Any MCP-compatible client

### A2A Endpoint

Wraps the agent using AG2's built-in `A2aAgentServer` from `autogen.a2a`.

```bash
ag2 serve my_agent.py --protocol a2a --port 9000
# → A2A server at http://localhost:9000/.well-known/agent.json
```

- **Single agent**: served at `/`
- **Multiple agents** (exported as `agents` list): each agent is mounted
  at `/<slug>/` where the slug is derived from the agent name

A2A protocol requires an exported `agent` or `agents` variable — a
`main()` function is not supported.

Other A2A-compatible agents (Google, Salesforce, etc.) can discover and
interact with your agent.

### --playground (coming soon)

Will launch a simple web UI alongside the API for testing agents in a
browser. For now, use `GET /docs` (Swagger UI) when serving with the
REST protocol.

### --ngrok

Expose your agent to the internet via an ngrok tunnel:

```bash
ag2 serve my_agent.py --ngrok --port 8000
# Local:  http://localhost:8000
# Public: https://abc123.ngrok.app
```

Requires:
- The `ngrok` pip package (`pip install ngrok`)
- An ngrok authtoken: set `NGROK_AUTHTOKEN` in your environment, or
  configure it via `ngrok config add-authtoken <token>`

The authtoken is resolved from (in order):
1. `NGROK_AUTHTOKEN` environment variable
2. ngrok config file (`~/Library/Application Support/ngrok/ngrok.yml`,
   `~/.config/ngrok/ngrok.yml`, or `~/.ngrok2/ngrok.yml`)

## Agent Discovery

Same convention as `ag2 run`:
1. `main()` function → used as the handler
2. `agent`/`team` variable → wrapped as chat endpoint
3. `agents` list → wrapped as GroupChat endpoint
4. Directory mode: each `.py` file in the directory becomes a separate endpoint

## Implementation Notes

### REST: Use FastAPI
AG2 already depends on a web stack for various features. FastAPI is the
natural choice (same author as Typer). The serve command:
1. Imports the agent file
2. Discovers agents using the convention
3. Generates a FastAPI app dynamically
4. Runs it with `uvicorn`

### MCP: Uses FastMCP
The serve command builds a FastMCP server that wraps the agent:
1. Import and discover the agent
2. Register a `chat` tool that sends messages and returns responses
3. Register a `list_agents` tool for introspection
4. Start the server with SSE transport

Requires the `mcp` pip package.

### A2A: Uses AG2's A2aAgentServer
Wraps `ConversableAgent` instances using `autogen.a2a.A2aAgentServer`:
1. Single agent: `A2aAgentServer(agent, url=...).build()` served directly
2. Multiple agents: each mounted as a Starlette `Mount` at `/<slug>/`

Requires the `ag2[a2a]` optional extra and `uvicorn`.

### Hot Reload
Use `watchfiles` (or uvicorn's built-in `--reload`) to watch the agent
file and restart on changes. Show a Rich notification on reload.

## Dependencies
- `ag2` — required
- `fastapi` + `uvicorn` — for REST serving
- `mcp` — for MCP serving (FastMCP)
- `ag2[a2a]` + `uvicorn` — for A2A serving
- `ngrok` — for `--ngrok` tunneling
- `watchfiles` — for `--reload` (or uvicorn's built-in reload)
