# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .models import BreakpointView, ResumeRequest, SessionView

_UI_FILE = Path(__file__).parent / "ui.html"


# ── Server-side session state ──────────────────────────────────────────────


class _BreakpointMeta:
    __slots__ = ("id", "type", "event_index", "timestamp", "resumed", "event_snapshot")

    def __init__(self, bp_id: str, bp_type: str, event_index: int, event_snapshot: dict[str, Any]) -> None:
        self.id = bp_id
        self.type = bp_type
        self.event_index = event_index
        self.timestamp: datetime = datetime.now(timezone.utc)
        self.resumed: bool = False
        self.event_snapshot = event_snapshot


class _Session:
    """
    Server-side representation of an agent debug session.

    Events and breakpoints arrive via HTTP from the agent process.
    The pending breakpoint is kept alive with an ``asyncio.Event`` so that
    the long-polling ``POST /sessions/{id}/breakpoints`` request blocks until
    the UI calls ``/resume``.
    """

    def __init__(self, session_id: str, prompt: list[str]) -> None:
        self.id = session_id
        self.prompt = list(prompt)
        self.events: list[dict[str, Any]] = []
        self.breakpoints: list[_BreakpointMeta] = []
        self._pending_waiters: dict[str, asyncio.Event] = {}
        self._pending_mods_map: dict[str, dict[str, Any]] = {}
        self.status = "running"

    @property
    def pending_bp_ids(self) -> list[str]:
        return list(self._pending_waiters.keys())


# ── View helpers ───────────────────────────────────────────────────────────


def _session_view(session: _Session) -> SessionView:
    bp_views = [
        BreakpointView(
            id=meta.id,
            type=meta.type,
            event_index=meta.event_index,
            timestamp=meta.timestamp,
            resumed=meta.resumed,
            event=meta.event_snapshot,
        )
        for meta in session.breakpoints
    ]
    return SessionView(
        id=session.id,
        status=session.status,
        prompt=session.prompt,
        events=session.events,
        breakpoints=bp_views,
        pending_bp_ids=session.pending_bp_ids,
    )


# ── FastAPI app factory ────────────────────────────────────────────────────


def _create_fastapi_app(sessions: dict[str, _Session]) -> FastAPI:
    app = FastAPI(title="AG2 Debug Server")
    _ws_clients: set[WebSocket] = set()

    async def _broadcast(msg: dict[str, Any]) -> None:
        gone: list[WebSocket] = []
        for client in _ws_clients:
            try:
                await client.send_json(msg)
            except Exception:
                gone.append(client)
        for c in gone:
            _ws_clients.discard(c)

    @app.get("/", include_in_schema=False)
    async def ui() -> FileResponse:
        return FileResponse(_UI_FILE, media_type="text/html")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        _ws_clients.add(ws)
        await ws.send_json({
            "type": "sessions_list",
            "sessions": [{"id": s.id, "status": s.status} for s in sessions.values()],
        })
        try:
            while True:
                await ws.receive_text()  # keep connection alive; ignore client messages
        except WebSocketDisconnect:
            pass
        finally:
            _ws_clients.discard(ws)

    # ── Session registration (called by agent on startup) ─────────────────

    @app.post("/sessions/{session_id}", status_code=201)
    async def register_session(session_id: str, body: dict[str, Any]) -> dict[str, str]:
        prompt: list[str] = body.get("prompt", [])
        if session_id in sessions:
            # Stream was reloaded — reset events (the agent will replay full history
            # from storage immediately after registering) and mark as running again.
            existing = sessions[session_id]
            existing.prompt = prompt
            existing.status = "running"
            existing.events.clear()
            existing.breakpoints.clear()
        else:
            sessions[session_id] = _Session(session_id, prompt)
        await _broadcast({
            "type": "sessions_list",
            "sessions": [{"id": s.id, "status": s.status} for s in sessions.values()],
        })
        return {"session_id": session_id}

    # ── Event ingestion (called by agent stream subscriber) ───────────────

    @app.post("/sessions/{session_id}/events")
    async def receive_event(session_id: str, body: dict[str, Any]) -> dict[str, str]:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        session.events.append(
            {
                "id": f"ev-{len(session.events)}",
                "event_type": body.get("event_type", ""),
                "event_data": body.get("event_data", {}),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        await _broadcast({"type": "session_update", "session": _session_view(session).model_dump(mode="json")})
        return {"ok": "true"}

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

    @app.post("/sessions/{session_id}/breakpoints")
    async def hit_breakpoint(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """
        Called by the agent when it hits a breakpoint.  Blocks (long-poll)
        until the UI calls ``/resume``, then returns the modification dict.
        """
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        bp_id = str(uuid4())
        bp_type = body.get("type", "")
        # TURN_START fires *before* the triggering event flows through the stream,
        # so the event will land at the current length (future index).
        # All other breakpoints (TOOL_CALL, etc.) fire *after* the triggering event
        # has already been recorded, so point to the last recorded event.
        if bp_type == "TURN_START":
            event_index = len(session.events)
        else:
            event_index = max(0, len(session.events) - 1)
        meta = _BreakpointMeta(bp_id, bp_type, event_index, body.get("event", {}))
        session.breakpoints.append(meta)
        waiter = asyncio.Event()
        session._pending_waiters[bp_id] = waiter

        await _broadcast({"type": "session_update", "session": _session_view(session).model_dump(mode="json")})
        await waiter.wait()

        meta.resumed = True
        session._pending_waiters.pop(bp_id, None)
        mods = session._pending_mods_map.pop(bp_id, {})

        await _broadcast({"type": "session_update", "session": _session_view(session).model_dump(mode="json")})
        return mods

    @app.post("/sessions/{session_id}/breakpoints/{bp_id}/resume")
    async def resume_breakpoint(session_id: str, bp_id: str, body: ResumeRequest) -> dict[str, Any]:
        session = sessions.get(session_id)
        if not session or bp_id not in session._pending_waiters:
            return {"success": False}

        session._pending_mods_map[bp_id] = {
            "event_modifications": body.event_modifications,
            "prompt": body.prompt,
            "variables": body.variables,
        }
        session._pending_waiters[bp_id].set()
        return {"success": True}

    # ── Context inspection ────────────────────────────────────────────────

    @app.get("/sessions/{session_id}/context")
    async def get_context(session_id: str) -> dict[str, Any]:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"prompt": session.prompt}

    return app


# ── DebugServer ────────────────────────────────────────────────────────────


class DebugServer:
    """
    Wraps the FastAPI app and its session registry.  Must be started externally
    via :func:`start_debug_server` or :func:`run_debug_server` — the agent
    never starts a server itself.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._app: FastAPI = _create_fastapi_app(self._sessions)


async def start_debug_server(host: str = "localhost", port: int = 8765) -> DebugServer:
    """
    Start a :class:`DebugServer` as an asyncio background task in the **current**
    event loop and return it.  Call this once before running any agents::

        server = await start_debug_server()
        # set AG2_DEBUG_SERVER_URL=http://localhost:8765
        await agent.ask(...)
    """
    import uvicorn

    srv = DebugServer()
    config = uvicorn.Config(srv._app, host=host, port=port, log_level="warning")
    uv_server = uvicorn.Server(config)
    uv_server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    asyncio.create_task(uv_server.serve())
    return srv


def run_debug_server(host: str = "localhost", port: int = 8765) -> None:
    """Blocking entry-point for running the server as a standalone process."""
    import uvicorn

    srv = DebugServer()
    uvicorn.run(srv._app, host=host, port=port)
