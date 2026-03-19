# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .models import BreakpointRecord, BreakpointType, EventRecord


class DebugSession:
    def __init__(self, session_id: str) -> None:
        self.id = session_id
        self.events: list[EventRecord] = []
        self.breakpoints: list[BreakpointRecord] = []
        self._pending_bp: asyncio.Event | None = None
        self._pending_bp_id: str | None = None
        self.status = "running"

    async def add_event(self, event_type: str, event_data: dict[str, Any]) -> EventRecord:
        record = EventRecord(
            id=str(uuid4()),
            event_type=event_type,
            event_data=event_data,
            timestamp=datetime.now(timezone.utc),
        )
        self.events.append(record)
        return record

    async def add_breakpoint(self, bp_type: BreakpointType, event_type: str, event_data: dict[str, Any]) -> str:
        """Create a breakpoint record and block until resumed. Returns the breakpoint id."""
        bp_id = str(uuid4())
        waiter = asyncio.Event()
        record = BreakpointRecord(
            id=bp_id,
            type=bp_type,
            event_type=event_type,
            event_data=event_data,
            timestamp=datetime.now(timezone.utc),
            resumed=False,
        )
        self.breakpoints.append(record)
        self._pending_bp = waiter
        self._pending_bp_id = bp_id
        await waiter.wait()
        return bp_id

    async def resume(self, bp_id: str) -> bool:
        """Resume the pending breakpoint. Returns True if successful."""
        if self._pending_bp_id != bp_id or self._pending_bp is None:
            return False
        for bp in self.breakpoints:
            if bp.id == bp_id:
                bp.resumed = True
                break
        self._pending_bp.set()
        self._pending_bp = None
        self._pending_bp_id = None
        return True
