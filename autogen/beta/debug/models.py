# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel


class SessionView(BaseModel):
    """Full HTTP representation of a live DebugSession."""

    id: str
    status: str
    prompt: list[str]
    events: list[dict[str, Any]]  # [{type, data}, …] — serialized at response time
