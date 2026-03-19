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

# Serve with a web playground for testing
ag2 serve my_team.py --playground

# Serve multiple agents from a directory
ag2 serve agents/ --port 8000
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

Wraps the agent as a Model Context Protocol server. Each agent tool becomes
an MCP tool. The agent itself becomes an MCP tool called `chat`.

```bash
ag2 serve my_agent.py --protocol mcp
# → MCP server running on stdio (or --transport sse for HTTP)
```

This means any AG2 agent can be instantly used from:
- Claude Desktop (add to `claude_desktop_config.json`)
- Cursor
- Windsurf
- Any MCP-compatible client

AG2 already has `autogen.mcp` — this command just wires it up automatically.

### A2A Endpoint

Wraps the agent using AG2's A2A integration (`autogen.a2a`). The agent
publishes an Agent Card for discovery.

```bash
ag2 serve my_agent.py --protocol a2a --port 9000
# → A2A server at http://localhost:9000/.well-known/agent.json
```

Other A2A-compatible agents (Google, Salesforce, etc.) can discover and
interact with your agent.

### --playground

Launches a simple web UI alongside the API:

```
┌─────────────────────────────────────────────────────┐
│  AG2 Playground — research-team                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  [Agent conversation rendered here]                 │
│                                                     │
│  ┌─────────────────────────────────┐  ┌──────────┐ │
│  │ Type your message...            │  │  Send  ▶ │ │
│  └─────────────────────────────────┘  └──────────┘ │
│                                                     │
│  Agents: researcher, critic, writer                 │
│  Pattern: auto | Model: gpt-4o                      │
│  Tokens: 0 | Cost: $0.00                            │
└─────────────────────────────────────────────────────┘
```

This is intentionally minimal — not AutoGen Studio. Just enough to test
your agents in a browser.

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

### MCP: Use existing autogen.mcp
AG2's MCP module already handles the protocol. The serve command just
needs to:
1. Import the agent
2. Create an MCP server from its tools
3. Add a `chat` tool that wraps `initiate_chat()`
4. Start the server

### A2A: Use existing autogen.a2a
Similar to MCP — wrap the agent using AG2's A2A SDK integration.

### Hot Reload
Use `watchfiles` (or uvicorn's built-in `--reload`) to watch the agent
file and restart on changes. Show a Rich notification on reload.

## Dependencies
- `ag2` — required
- `fastapi` + `uvicorn` — for REST serving
- `watchfiles` — for --reload
- (MCP and A2A deps already in ag2 optional extras)
