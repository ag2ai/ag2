# Serve an AG2 agent as an MCP server

These examples wrap an `autogen.beta` `Agent` with `autogen.beta.mcp.MCPServer` and
expose it over the Model Context Protocol so any MCP client can call it. The agent
appears as a single conversational tool: `ask(message, context?)`.

## Setup

```bash
# from the repo root
pip install -e ".[mcp,anthropic]"     # ag2 + the `mcp` extra + Anthropic provider
export ANTHROPIC_API_KEY=sk-ant-...    # the examples use claude-sonnet-4-6
```

## Run it locally (self-contained, no extra client needed)

One-shot — spawns the stdio server, lists the tool, and calls it once:

```bash
python -m examples.mcp.client_demo
```

Interactive multi-turn chat over stdio (`/reset`, `/tools`, `/quit`):

```bash
python -m examples.mcp.client_interactive
```

> The server is **stateless** (each `ask` call is a fresh conversation), so the
> interactive client keeps continuity by threading the running transcript back
> through the tool's `context` argument.

## Stdio server (Claude Desktop / Cursor / MCP Inspector)

```bash
python -m examples.mcp.server_stdio
# or inspect interactively:
npx @modelcontextprotocol/inspector python -m examples.mcp.server_stdio
```

Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ag2-claude": {
      "command": "python",
      "args": ["-m", "examples.mcp.server_stdio"],
      "env": { "ANTHROPIC_API_KEY": "sk-ant-..." }
    }
  }
}
```

## Streamable-HTTP server (remote / production)

```bash
uvicorn examples.mcp.server_http:app --host 127.0.0.1 --port 8000
```

Then connect a client to `http://127.0.0.1:8000/mcp` (in the MCP Inspector choose the
"Streamable HTTP" transport). `app` is a plain Starlette ASGI app — attach CORS / auth
middleware to it as needed.

## Files

| File | What it does |
|------|--------------|
| `server_stdio.py` | Wraps the agent and serves it over stdio (`MCPServer(agent).run_stdio()`). |
| `server_http.py`  | Exposes `app = MCPServer(agent).build_streamable_http()` for uvicorn. |
| `client_demo.py`  | Launches the stdio server and calls the `ask` tool end-to-end. |
