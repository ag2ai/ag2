"""ag2 serve — expose agents as REST APIs, MCP servers, or A2A endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from ..ui import console


def _require_fastapi() -> tuple[Any, Any]:
    """Import FastAPI + uvicorn or exit with a helpful error."""
    try:
        import fastapi
        import uvicorn

        return fastapi, uvicorn
    except ImportError:
        console.print("[error]FastAPI and uvicorn are required for ag2 serve.[/error]")
        console.print("Install with: [command]pip install fastapi uvicorn[standard][/command]")
        raise typer.Exit(1)


def _build_rest_app(discovered: Any) -> Any:
    """Build a FastAPI app from a discovered agent."""
    from ..core.discovery import DiscoveredAgent

    fastapi, _ = _require_fastapi()
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
        agent_names: list[str]

    class AgentInfo(BaseModel):
        name: str
        kind: str

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/agents")
    async def list_agents() -> list[AgentInfo]:
        return [AgentInfo(name=n, kind=d.kind) for n in d.agent_names]

    @app.post("/chat")
    async def chat(req: ChatRequest) -> ChatResponse:
        import autogen

        if d.kind == "main" and d.main_fn is not None:
            import asyncio
            import inspect

            kwargs: dict[str, Any] = {}
            sig = inspect.signature(d.main_fn)
            if "message" in sig.parameters:
                kwargs["message"] = req.message

            result = d.main_fn(**kwargs)
            if asyncio.iscoroutine(result):
                result = await result

            output = str(result)
            if hasattr(result, "chat_history"):
                msgs = result.chat_history or []
                agent_msgs = [m for m in msgs if m.get("role") != "user" and m.get("content")]
                output = agent_msgs[-1]["content"] if agent_msgs else ""
                return ChatResponse(output=output, turns=len(msgs), agent_names=d.agent_names)
            return ChatResponse(output=output, turns=1, agent_names=d.agent_names)

        if d.kind == "agent":
            user_proxy = autogen.UserProxyAgent(
                name="user",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False,
            )
            result = user_proxy.initiate_chat(
                d.agent,
                message=req.message,
                max_turns=req.max_turns,
            )
            history = result.chat_history or []
            agent_msgs = [m for m in history if m.get("role") != "user" and m.get("content")]
            output = agent_msgs[-1]["content"] if agent_msgs else ""
            return ChatResponse(output=output, turns=len(history), agent_names=d.agent_names)

        if d.kind == "agents":
            groupchat = autogen.GroupChat(
                agents=d.agents,
                messages=[],
                max_round=req.max_turns,
            )
            manager = autogen.GroupChatManager(groupchat=groupchat)
            user_proxy = autogen.UserProxyAgent(
                name="user",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False,
            )
            result = user_proxy.initiate_chat(manager, message=req.message)
            history = result.chat_history or []
            agent_msgs = [m for m in history if m.get("role") != "user" and m.get("content")]
            output = agent_msgs[-1]["content"] if agent_msgs else ""
            return ChatResponse(output=output, turns=len(history), agent_names=d.agent_names)

        return ChatResponse(output="Unknown agent kind", turns=0, agent_names=[])

    return app


def serve_cmd(
    agent_file: Path = typer.Argument(..., help="Python file defining agent(s) to serve."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on."),
    protocol: str = typer.Option("rest", "--protocol", help="Protocol: rest, mcp, or a2a."),
    playground: bool = typer.Option(False, "--playground", help="Launch web playground UI."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on file changes."),
) -> None:
    """Serve agents as APIs, MCP servers, or A2A endpoints.

    [dim]Examples:[/dim]
      [command]ag2 serve my_team.py[/command]
      [command]ag2 serve my_agent.py --protocol mcp[/command]
      [command]ag2 serve my_agent.py --protocol a2a --port 9000[/command]
      [command]ag2 serve my_team.py --playground[/command]
    """
    if protocol not in ("rest", "mcp", "a2a"):
        console.print(f"[error]Unknown protocol: {protocol}[/error]")
        console.print("Supported protocols: rest, mcp, a2a")
        raise typer.Exit(1)

    if protocol == "mcp":
        console.print("[warning]MCP protocol support is coming soon.[/warning]")
        console.print("Use [command]--protocol rest[/command] for now.")
        raise typer.Exit(0)

    if protocol == "a2a":
        console.print("[warning]A2A protocol support is coming soon.[/warning]")
        console.print("Use [command]--protocol rest[/command] for now.")
        raise typer.Exit(0)

    if playground:
        console.print("[warning]Web playground is coming soon.[/warning]")
        console.print("Use [command]GET /docs[/command] for Swagger UI instead.")

    # Require ag2 and FastAPI
    try:
        import autogen  # noqa: F401
    except ImportError:
        console.print("[error]ag2 is not installed.[/error]")
        console.print("Install with: [command]pip install ag2[/command]")
        raise typer.Exit(1)

    _, uvicorn = _require_fastapi()

    path = Path(agent_file).resolve()
    if not path.exists():
        console.print(f"[error]File not found: {path}[/error]")
        raise typer.Exit(1)

    # Discover agents
    from ..core.discovery import discover

    try:
        discovered = discover(path)
    except (ValueError, ImportError) as exc:
        console.print(f"[error]{exc}[/error]")
        raise typer.Exit(1)

    console.print(
        f"\n[heading]AG2 Serve[/heading] — {', '.join(discovered.agent_names)}"
    )
    console.print(f"  Protocol: [info]{protocol}[/info]")
    console.print(f"  Endpoint: [info]http://localhost:{port}[/info]")
    console.print(f"  Docs:     [info]http://localhost:{port}/docs[/info]")
    console.print()

    # Build and run the FastAPI app
    fast_app = _build_rest_app(discovered)
    uvicorn.run(fast_app, host="0.0.0.0", port=port, reload=reload)
