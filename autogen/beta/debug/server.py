# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .models import BreakpointRecord, BreakpointType, EventRecord, SessionState
from .session import DebugSession

_UI_FILE = Path(__file__).parent / "ui.html"


def create_app() -> FastAPI:
    app = FastAPI(title="AG2 Debug Server")

    @app.get("/", include_in_schema=False)
    async def ui() -> FileResponse:
        return FileResponse(_UI_FILE, media_type="text/html")
    sessions: dict[str, DebugSession] = {}

    @app.post("/sessions")
    async def create_session(body: dict[str, Any]) -> dict[str, str]:
        session_id = body.get("session_id") or str(uuid4())
        if session_id not in sessions:
            sessions[session_id] = DebugSession(session_id)
        return {"session_id": session_id}

    @app.get("/sessions")
    async def list_sessions() -> list[dict[str, str]]:
        return [{"id": s.id, "status": s.status} for s in sessions.values()]

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> SessionState:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionState(
            id=session.id,
            events=session.events,
            breakpoints=session.breakpoints,
            status=session.status,
        )

    @app.post("/sessions/{session_id}/events")
    async def add_event(session_id: str, body: dict[str, Any]) -> EventRecord:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        record = await session.add_event(
            event_type=body["event_type"],
            event_data=body.get("event_data", {}),
        )
        return record

    @app.post("/sessions/{session_id}/breakpoints")
    async def add_breakpoint(session_id: str, body: dict[str, Any]) -> BreakpointRecord:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        bp_id = await session.add_breakpoint(
            bp_type=BreakpointType(body["bp_type"]),
            event_type=body["event_type"],
            event_data=body.get("event_data", {}),
        )
        for bp in session.breakpoints:
            if bp.id == bp_id:
                return bp
        raise HTTPException(status_code=500, detail="Breakpoint not found after creation")  # pragma: no cover

    @app.post("/sessions/{session_id}/breakpoints/{bp_id}/resume")
    async def resume_breakpoint(session_id: str, bp_id: str) -> dict[str, bool]:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        success = await session.resume(bp_id)
        return {"success": success}

    return app


def run_debug_server(host: str = "localhost", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
