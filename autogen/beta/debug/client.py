# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
HTTP client the agent uses to communicate with a running debug server,
plus serialisation utilities for converting live BaseEvent objects into
JSON-compatible dicts.
"""

from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any

import httpx

from ..events.base import BaseEvent

# ── Serialisation ──────────────────────────────────────────────────────────


def _serialize_value(value: Any) -> Any:
    """Recursively convert a value to a JSON-compatible type."""
    if isinstance(value, BaseEvent):
        return {
            "type": type(value).__name__,
            **{k: _serialize_value(v) for k, v in value.__dict__.items() if not k.startswith("_")},
        }
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Exception):
        return {"type": type(value).__name__, "message": str(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _serialize_value(getattr(value, f.name)) for f in dataclass_fields(value)}
    return repr(value)


def serialize_event(event: BaseEvent) -> dict[str, Any]:
    """Return ``{type: str, data: dict}`` for a single BaseEvent."""
    data = {k: _serialize_value(v) for k, v in event.__dict__.items() if not k.startswith("_")}
    return {"type": type(event).__name__, "data": data}


# ── HTTP client ────────────────────────────────────────────────────────────


class DebugClient:
    """
    Thin async HTTP client the agent uses to talk to an external debug server.

    Every ``DebugSession`` holds one of these.  The server must be started
    separately (e.g. via ``start_debug_server()`` or ``run_debug_server()``)
    before the agent runs.
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def register_stream(self, stream_id: str, prompt: list[str]) -> None:
        """Tell the server a new stream has started (or reset an existing one)."""
        async with httpx.AsyncClient() as c:
            await c.post(f"{self._base_url}/streams/{stream_id}", json={"prompt": prompt})

    async def register_session(self, session_id: str, name: str, stream_id: str) -> None:
        """Register a named session that references a stream."""
        async with httpx.AsyncClient() as c:
            await c.post(
                f"{self._base_url}/sessions",
                json={"session_id": session_id, "name": name, "stream_id": stream_id},
            )

    async def send_event(self, stream_id: str, event_type: str, event_data: dict[str, Any]) -> None:
        """Forward a stream event to the server (fire-and-forget)."""
        try:
            async with httpx.AsyncClient() as c:
                await c.post(
                    f"{self._base_url}/streams/{stream_id}/events",
                    json={"event_type": event_type, "event_data": event_data},
                )
        except Exception:
            pass  # never crash the agent due to a debug side-channel failure

    async def add_stream_to_session(self, session_id: str, stream_id: str) -> None:
        """Add an additional stream to an existing session."""
        try:
            async with httpx.AsyncClient() as c:
                await c.post(
                    f"{self._base_url}/sessions/{session_id}/streams",
                    json={"stream_id": stream_id},
                )
        except Exception:
            pass  # never crash the agent due to a debug side-channel failure

    async def end_session(self, session_id: str) -> None:
        """Mark a session as done, freezing its event snapshot."""
        try:
            async with httpx.AsyncClient() as c:
                await c.post(f"{self._base_url}/sessions/{session_id}/done")
        except Exception:
            pass  # never crash the agent due to a debug side-channel failure


def get_server(base_url: str) -> DebugClient:
    """Create and return an HTTP client connected to the debug server at *base_url*."""
    return DebugClient(base_url)
