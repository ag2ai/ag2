# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields as dataclass_fields, is_dataclass
from typing import Any

import httpx

from ..events.base import BaseEvent


def _serialize_value(value: Any) -> Any:
    """Recursively serialize a value to a JSON-compatible type."""
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
    """Serialize a BaseEvent to a JSON-compatible dict with a 'type' and 'data' key."""
    data = {k: _serialize_value(v) for k, v in event.__dict__.items() if not k.startswith("_")}
    return {"type": type(event).__name__, "data": data}


class DebugClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        # timeout=None for breakpoints (they block until resumed); short timeout for events
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=None)

    async def create_session(self, session_id: str) -> None:
        await self._client.post("/sessions", json={"session_id": session_id})

    async def send_event(self, session_id: str, event: BaseEvent) -> None:
        """Fire-and-forget event forwarding — errors are silently ignored."""
        try:
            serialized = serialize_event(event)
            await self._client.post(
                f"/sessions/{session_id}/events",
                json={"event_type": serialized["type"], "event_data": serialized["data"]},
                timeout=5.0,
            )
        except Exception:
            pass

    async def hit_breakpoint(self, session_id: str, bp_type: str, event: BaseEvent) -> None:
        """POST to /breakpoints and block until the server responds (i.e. until resumed)."""
        serialized = serialize_event(event)
        await self._client.post(
            f"/sessions/{session_id}/breakpoints",
            json={"bp_type": bp_type, "event_type": serialized["type"], "event_data": serialized["data"]},
        )

    async def aclose(self) -> None:
        await self._client.aclose()
