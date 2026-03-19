# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .models import BreakpointType, BreakpointView, InjectRequest, ResumeRequest, SessionView
from .session import DebugSession

_UI_FILE = Path(__file__).parent / "ui.html"

# ── Event type registry for injection ──────────────────────────────────────
# Only constructable event types are listed; read-only / internal types are
# omitted intentionally.

def _build_event_registry() -> dict[str, type]:  # type: ignore[type-arg]
    from ..events import (
        HumanInputRequest,
        HumanMessage,
        ModelMessage,
        ModelReasoning,
        ModelRequest,
        ToolCallEvent,
    )

    return {
        "ModelRequest": ModelRequest,
        "ModelMessage": ModelMessage,
        "ModelReasoning": ModelReasoning,
        "HumanInputRequest": HumanInputRequest,
        "HumanMessage": HumanMessage,
        "ToolCallEvent": ToolCallEvent,
    }


def _serialize_event(event: Any) -> dict[str, Any]:
    """Import lazily to avoid a circular dependency at module load time."""
    from .client import serialize_event
    from ..events.base import BaseEvent

    if isinstance(event, BaseEvent):
        return serialize_event(event)
    return {"type": type(event).__name__, "data": {}}


def _session_view(session: DebugSession) -> SessionView:
    serialized_events = [_serialize_event(e) for e in session.events]

    bp_views: list[BreakpointView] = []
    for meta in session.breakpoints:
        idx = meta.event_index
        ev_snapshot = serialized_events[idx] if 0 <= idx < len(serialized_events) else {}
        bp_views.append(
            BreakpointView(
                id=meta.id,
                type=meta.type,
                event_index=idx,
                timestamp=meta.timestamp,
                resumed=meta.resumed,
                event=ev_snapshot,
            )
        )

    return SessionView(
        id=session.id,
        status=session.status,
        prompt=session.prompt,
        events=serialized_events,
        breakpoints=bp_views,
        pending_bp_id=session.pending_bp_id,
    )


# ── FastAPI app factory ────────────────────────────────────────────────────

def _create_fastapi_app(sessions: dict[str, DebugSession]) -> FastAPI:
    app = FastAPI(title="AG2 Debug Server")

    @app.get("/", include_in_schema=False)
    async def ui() -> FileResponse:
        return FileResponse(_UI_FILE, media_type="text/html")

    # ── Sessions ──────────────────────────────────────────────────────────

    @app.get("/sessions")
    async def list_sessions() -> list[dict[str, str]]:
        return [{"id": s.id, "status": s.status} for s in sessions.values()]

    @app.get("/sessions/{session_id}", response_model=SessionView)
    async def get_session(session_id: str) -> SessionView:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return _session_view(session)

    # ── Breakpoints ───────────────────────────────────────────────────────

    @app.post("/sessions/{session_id}/breakpoints/{bp_id}/resume")
    async def resume_breakpoint(session_id: str, bp_id: str, body: ResumeRequest) -> dict[str, Any]:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        success = await session.resume(
            bp_id,
            event_modifications=body.event_modifications or None,
            prompt=body.prompt,
            variables=body.variables or None,
        )
        return {"success": success}

    # ── Stream injection ──────────────────────────────────────────────────

    @app.post("/sessions/{session_id}/inject")
    async def inject_event(session_id: str, body: InjectRequest) -> dict[str, str]:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        registry = _build_event_registry()
        event_cls = registry.get(body.event_type)
        if not event_cls:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown injectable event type '{body.event_type}'. "
                       f"Supported: {sorted(registry)}",
            )
        try:
            event = event_cls(**body.event_data)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not construct event: {exc}") from exc
        await session.inject(event)
        return {"status": "injected", "event_type": body.event_type}

    # ── Context inspection / mutation ─────────────────────────────────────

    @app.get("/sessions/{session_id}/context")
    async def get_context(session_id: str) -> dict[str, Any]:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "prompt": session.context.prompt,
            "variables": session.context.variables,
        }

    @app.patch("/sessions/{session_id}/context")
    async def patch_context(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Directly mutate context.prompt and/or context.variables (outside a breakpoint)."""
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if "prompt" in body:
            session.context.prompt[:] = body["prompt"]
        if "variables" in body:
            session.context.variables.update(body["variables"])
        return {"prompt": session.context.prompt, "variables": session.context.variables}

    return app


# ── DebugServer — runs in-process on the same asyncio event loop ──────────

class DebugServer:
    """
    Manages a FastAPI/uvicorn server that runs as an asyncio task inside the
    agent's own event loop.  All ``DebugSession`` objects it holds are live
    Python objects — no serialisation happens until an HTTP client requests
    data.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, DebugSession] = {}
        self._app: FastAPI = _create_fastapi_app(self._sessions)
        self._started = False

    def register(self, session: DebugSession) -> None:
        self._sessions[session.id] = session

    async def start(self, host: str = "localhost", port: int = 8765) -> None:
        if self._started:
            return
        import uvicorn

        config = uvicorn.Config(self._app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        # Suppress uvicorn's signal-handler installation — we're a guest in
        # someone else's event loop and must not replace their SIGINT handler.
        server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        asyncio.create_task(server.serve())
        self._started = True


# ── Module-level singleton (one server per host:port) ─────────────────────

_servers: dict[tuple[str, int], DebugServer] = {}


async def get_or_create_server(host: str = "localhost", port: int = 8765) -> DebugServer:
    key = (host, port)
    if key not in _servers:
        srv = DebugServer()
        await srv.start(host, port)
        _servers[key] = srv
    return _servers[key]


def run_debug_server(host: str = "localhost", port: int = 8765) -> None:
    """Blocking entry-point for running the server standalone (e.g. from a shell)."""
    import uvicorn

    uvicorn.run(_create_fastapi_app({}), host=host, port=port)
