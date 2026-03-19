# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BreakpointType(str, Enum):
    TURN_START = "TURN_START"
    TOOL_CALL = "TOOL_CALL"
    LLM_CALL = "LLM_CALL"


class BreakpointView(BaseModel):
    """HTTP representation of a single breakpoint."""

    id: str
    type: str
    event_index: int  # index into SessionView.events
    timestamp: datetime
    resumed: bool
    event: dict[str, Any]  # serialized snapshot of the event at pause time


class SessionView(BaseModel):
    """Full HTTP representation of a live DebugSession."""

    id: str
    status: str
    prompt: list[str]
    events: list[dict[str, Any]]  # [{type, data}, …] — serialized at response time
    breakpoints: list[BreakpointView]
    pending_bp_id: str | None


class ResumeRequest(BaseModel):
    """Body for POST /sessions/{id}/breakpoints/{bp_id}/resume."""

    # Mutate fields on the paused event before it continues
    event_modifications: dict[str, Any] = Field(default_factory=dict)
    # Replace context.prompt in-place
    prompt: list[str] | None = None
    # Merge into context.variables
    variables: dict[str, Any] = Field(default_factory=dict)


class InjectRequest(BaseModel):
    """Body for POST /sessions/{id}/inject — push an event into the live stream."""

    event_type: str
    event_data: dict[str, Any] = Field(default_factory=dict)
