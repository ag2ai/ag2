# Serve an AG2 agent as an MCP server

These examples wrap an `autogen.beta` `Agent` with `autogen.beta.mcp.MCPServer` and
expose it over the Model Context Protocol so any MCP client can call it. The agent
appears as a single conversational tool: `ask(message, context?)`.

## Setup

```bash
# from the repo root
pip install -e ".[mcp,anthropic]"     # ag2 + the `mcp` extra + Anthropic provider
export ANTHROPIC_API_KEY=<your-anthropic-key>    # the examples use claude-sonnet-4-6
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
      "env": { "ANTHROPIC_API_KEY": "<your-anthropic-key>" }
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

## Authenticated server (OAuth 2.0 Resource Server)

Pass a `security=` requirement (built with `autogen.beta.mcp.security.require`) to
`build_streamable_http` to serve RFC 9728 protected-resource discovery and enforce
bearer tokens on `/mcp`. Token *issuance* stays with your authorization server; you
bring a `TokenVerifier` that validates the presented token.

```bash
uvicorn examples.mcp.server_http_auth:app --host 127.0.0.1 --port 8000

# discover where to authenticate
curl http://127.0.0.1:8000/.well-known/oauth-protected-resource/mcp

# call it (the demo accepts the static token "demo-secret-token")
python -m examples.mcp.client_http_auth
```

Without a token, `/mcp` returns `401` with a `WWW-Authenticate` header pointing at the
metadata; an insufficient scope returns `403`. The demo uses a static-token verifier —
replace it with one that checks your IdP's JWKS / introspection in production.

## Embed in an existing FastAPI / Starlette app

`build_streamable_http()` returns a Starlette app whose **lifespan** runs the MCP
session manager — and mounted sub-apps don't get their lifespan run by the parent, so a
plain `app.mount(...)` fails with *"Task group is not initialized"*. Use
`MCPServer(agent).mount_into(app, path="/mcp")` instead: it adds the MCP routes at the
app root (so `path` and the host-root `.well-known` land where clients expect) and
composes the lifespan into your app's. Any auth (`security=`) stays scoped to the MCP
route, so the rest of your app is unaffected.

```bash
uvicorn examples.mcp.server_embedded:app --host 127.0.0.1 --port 8000
#   GET  /        -> your API
#   POST /mcp     -> the AG2 agent over MCP
```

## Files

| File | What it does |
|------|--------------|
| `server_stdio.py`       | Wraps the agent and serves it over stdio (`MCPServer(agent).run_stdio()`). |
| `server_http.py`        | Exposes `app = MCPServer(agent).build_streamable_http()` for uvicorn. |
| `server_http_auth.py`   | Streamable-HTTP server with OAuth Resource Server auth + `.well-known` discovery. |
| `server_embedded.py`    | Mounts the agent into a FastAPI app via `MCPServer.mount_into(app)`. |
| `client_demo.py`        | Launches the stdio server and calls the `ask` tool once. |
| `client_interactive.py` | Interactive multi-turn stdio chat. |
| `client_http.py`        | Calls a running streamable-HTTP server. |
| `client_http_auth.py`   | Discovers metadata and calls the authenticated server with a bearer token. |
