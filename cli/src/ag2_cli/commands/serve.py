"""ag2 serve — expose agents as REST APIs, MCP servers, or A2A endpoints."""

from pathlib import Path
from typing import Any

import typer

from ..ui import console
from ._shared import require_ag2 as _require_ag2


def _require_fastapi() -> tuple[Any, Any]:
    """Import FastAPI + uvicorn or exit with a helpful error."""
    try:
        import fastapi
        import uvicorn

        return fastapi, uvicorn
    except ImportError:
        console.print("[error]FastAPI and uvicorn are required for REST serving.[/error]")
        console.print("Install with: [command]pip install fastapi 'uvicorn[standard]'[/command]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# REST protocol
# ---------------------------------------------------------------------------


def _build_rest_app(discovered: Any) -> Any:
    """Build a FastAPI app from a discovered agent."""
    from ..core.discovery import DiscoveredAgent
    from ..core.runner import execute

    _require_fastapi()
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    d: DiscoveredAgent = discovered

    app = FastAPI(
        title="AG2 Agent API",
        description=f"REST API for AG2 agent(s): {', '.join(d.agent_names)}",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class ChatRequest(BaseModel):
        message: str
        max_turns: int = 10

    class ChatResponse(BaseModel):
        output: str
        turns: int
        elapsed: float
        agent_names: list[str]

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/agents")
    async def list_agents() -> list[dict[str, str]]:
        return [{"name": n, "kind": d.kind} for n in d.agent_names]

    @app.post("/chat")
    async def chat(req: ChatRequest) -> ChatResponse:
        import asyncio

        result = await asyncio.to_thread(execute, d, req.message, max_turns=req.max_turns)
        return ChatResponse(
            output=result.output,
            turns=result.turns,
            elapsed=round(result.elapsed, 2),
            agent_names=result.agent_names,
        )

    return app


# ---------------------------------------------------------------------------
# MCP protocol
# ---------------------------------------------------------------------------


def _build_mcp_server(discovered: Any, host: str = "0.0.0.0", port: int = 8000) -> Any:
    """Build a FastMCP server that exposes the agent as an MCP tool.

    This lets any MCP client (Claude Desktop, Cursor, etc.) interact
    with the AG2 agent via the standard MCP tool-calling protocol.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        console.print("[error]MCP SDK is required for MCP serving.[/error]")
        console.print("Install with: [command]pip install mcp[/command]")
        raise typer.Exit(1)

    from ..core.discovery import DiscoveredAgent
    from ..core.runner import execute

    d: DiscoveredAgent = discovered
    agent_names = ", ".join(d.agent_names)

    mcp = FastMCP(f"AG2: {agent_names}", host=host, port=port)

    @mcp.tool()
    def chat(message: str) -> str:
        """Send a message to the AG2 agent and get a response."""
        result = execute(d, message, max_turns=1)
        if result.errors:
            return f"Error: {result.errors[0]}"
        return result.output

    @mcp.tool()
    def list_agents() -> str:
        """List the available agents and their orchestration type."""
        import json

        info = [{"name": n, "kind": d.kind} for n in d.agent_names]
        return json.dumps(info, indent=2)

    return mcp


# ---------------------------------------------------------------------------
# A2A protocol
# ---------------------------------------------------------------------------


def _build_a2a_app(discovered: Any, port: int) -> Any:
    """Build a Starlette ASGI app that exposes agents via A2A protocol.

    Uses AG2's built-in A2aAgentServer to wrap ConversableAgent instances
    as A2A-compatible HTTP endpoints.
    """
    try:
        from autogen.a2a import A2aAgentServer
    except ImportError:
        console.print("[error]A2A support is required.[/error]")
        console.print("Install with: [command]pip install 'ag2\\[a2a]'[/command]")
        raise typer.Exit(1)

    from ..core.discovery import DiscoveredAgent

    d: DiscoveredAgent = discovered

    if d.kind == "agent" and d.agent is not None:
        server = A2aAgentServer(
            d.agent,
            url=f"http://0.0.0.0:{port}",
        ).build()
        return server

    if d.kind == "agents" and d.agents:
        # Multi-agent: mount each agent at its own path
        from starlette.applications import Starlette
        from starlette.routing import Mount

        routes = []
        for agent in d.agents:
            slug = agent.name.lower().replace(" ", "-")
            a2a_server = A2aAgentServer(
                agent,
                url=f"http://0.0.0.0:{port}/{slug}/",
            ).build()
            routes.append(Mount(f"/{slug}/", a2a_server))

        return Starlette(routes=routes)

    if d.kind == "main":
        console.print("[error]A2A protocol requires an exported agent or agents list, not a main() function.[/error]")
        console.print("Export your agent as a module-level variable named 'agent' or 'agents'.")
        raise typer.Exit(1)

    console.print("[error]No agents found to serve via A2A.[/error]")
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# ngrok tunnel
# ---------------------------------------------------------------------------


def _start_ngrok(port: int) -> str:
    """Start an ngrok tunnel and return the public URL."""
    try:
        import ngrok as ngrok_sdk
    except ImportError:
        console.print("[error]ngrok SDK is required for tunneling.[/error]")
        console.print("Install with: [command]pip install ngrok[/command]")
        raise typer.Exit(1)

    import os

    # Try env var first, then fall back to ngrok config file
    authtoken = os.environ.get("NGROK_AUTHTOKEN")
    if not authtoken:
        try:
            import yaml

            for cfg_path in [
                Path.home() / "Library" / "Application Support" / "ngrok" / "ngrok.yml",
                Path.home() / ".config" / "ngrok" / "ngrok.yml",
                Path.home() / ".ngrok2" / "ngrok.yml",
            ]:
                if cfg_path.exists():
                    with open(cfg_path) as f:
                        cfg = yaml.safe_load(f) or {}
                    authtoken = cfg.get("authtoken")
                    if authtoken:
                        break
        except Exception:
            pass

    if not authtoken:
        console.print("[error]No ngrok authtoken found.[/error]")
        console.print(
            "Set NGROK_AUTHTOKEN in your environment or run [command]ngrok config add-authtoken <token>[/command]."
        )
        raise typer.Exit(1)

    try:
        listener = ngrok_sdk.forward(port, authtoken=authtoken)
        url = listener.url()
        return str(url)
    except Exception as exc:
        console.print(f"[error]Failed to start ngrok tunnel: {exc}[/error]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


def serve_cmd(
    agent_file: Path = typer.Argument(..., help="Python file defining agent(s) to serve."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on."),
    protocol: str = typer.Option("rest", "--protocol", help="Protocol: rest, mcp, or a2a."),
    ngrok: bool = typer.Option(False, "--ngrok", help="Expose via ngrok tunnel (requires ngrok SDK + auth)."),
    playground: bool = typer.Option(False, "--playground", help="Launch web playground UI."),
) -> None:
    """Serve agents as APIs, MCP servers, or A2A endpoints.

    [dim]Examples:[/dim]
      [command]ag2 serve my_team.py[/command]
      [command]ag2 serve my_agent.py --protocol mcp[/command]
      [command]ag2 serve my_agent.py --protocol a2a --port 9000[/command]
      [command]ag2 serve my_agent.py --ngrok[/command]
    """
    if protocol not in ("rest", "mcp", "a2a"):
        console.print(f"[error]Unknown protocol: {protocol}[/error]")
        console.print("Supported protocols: rest, mcp, a2a")
        raise typer.Exit(1)

    if playground:
        console.print("[warning]Web playground is coming soon.[/warning]")
        if protocol == "rest":
            console.print("Use [command]GET /docs[/command] for Swagger UI instead.")

    _require_ag2()

    path = Path(agent_file).resolve()
    if not path.exists():
        console.print(f"[error]File not found: {path}[/error]")
        raise typer.Exit(1)

    # Discover agents
    from ..core.discovery import discover

    try:
        if path.suffix in (".yaml", ".yml"):
            from ..core.discovery import build_agents_from_yaml, load_yaml_config

            config = load_yaml_config(path)
            discovered = build_agents_from_yaml(config)
        else:
            discovered = discover(path)
    except (ValueError, ImportError) as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1)

    # Start ngrok tunnel if requested
    public_url = None
    if ngrok:
        public_url = _start_ngrok(port)

    # Display header
    console.print(f"\n[heading]AG2 Serve[/heading] — {', '.join(discovered.agent_names)}")
    console.print(f"  Protocol: [info]{protocol}[/info]")
    console.print(f"  Endpoint: [info]http://localhost:{port}[/info]")
    if public_url:
        console.print(f"  Public:   [info]{public_url}[/info]")
    if protocol == "rest":
        console.print(f"  Docs:     [info]http://localhost:{port}/docs[/info]")
    console.print()

    # Build and run the appropriate server
    if protocol == "rest":
        _, uvicorn = _require_fastapi()
        fast_app = _build_rest_app(discovered)
        uvicorn.run(fast_app, host="0.0.0.0", port=port)

    elif protocol == "mcp":
        mcp = _build_mcp_server(discovered, host="0.0.0.0", port=port)
        mcp.run(transport="sse")

    elif protocol == "a2a":
        try:
            import uvicorn
        except ImportError:
            console.print("[error]uvicorn is required for A2A serving.[/error]")
            console.print("Install with: [command]pip install uvicorn[/command]")
            raise typer.Exit(1)

        a2a_app = _build_a2a_app(discovered, port)
        uvicorn.run(a2a_app, host="0.0.0.0", port=port)
