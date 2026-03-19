# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BreakpointType(str, Enum):
    TURN_START = "TURN_START"
    TOOL_CALL = "TOOL_CALL"
    LLM_CALL = "LLM_CALL"


class BreakpointRecord(BaseModel):
    id: str
    type: BreakpointType
    event_type: str
    event_data: dict[str, Any]
    timestamp: datetime
    resumed: bool = False


class EventRecord(BaseModel):
    id: str
    event_type: str
    event_data: dict[str, Any]
    timestamp: datetime


class SessionState(BaseModel):
    id: str
    events: list[EventRecord]
    breakpoints: list[BreakpointRecord]
    status: str  # "pending" | "running" | "done"
    prompt: list[str] = Field(default_factory=list)
