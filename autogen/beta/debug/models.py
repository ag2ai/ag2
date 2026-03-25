# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel


class StreamView(BaseModel):
    """Full HTTP representation of a stream's event log."""

    id: str
    prompt: list[str]
    events: list[dict[str, Any]]


class SessionView(BaseModel):
    """Full HTTP representation of a debug session."""

    id: str
    name: str
    stream_ids: list[str]
    status: str
    prompt: list[str]
    events: list[dict[str, Any]]  # live stream events if running; snapshot if done
