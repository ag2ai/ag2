# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Alert events — structured notifications from Observers.

ObserverAlert replaces the old Signal event type. Severity and HaltEvent
are framework-core concepts used by observers and AlertPolicy.
"""

from __future__ import annotations

from enum import Enum

from autogen.beta.events.base import BaseEvent, Field


class Severity(str, Enum):
    """Severity levels for observer alerts. Extensible via string values."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class ObserverAlert(BaseEvent):
    """Structured notification from an observer.

    Replaces the old Signal event. Same fields, clearer name.
    Produced by observers, consumed by AlertPolicy.
    """

    source: str  # Observer name that produced this alert
    severity: str  # Severity level (uses Severity enum values, but accepts any string)
    message: str  # Human/LLM-readable description
    data: dict = Field(default_factory=dict)  # Optional structured payload


class HaltEvent(BaseEvent):
    """Emitted when a FATAL alert triggers execution halt."""

    reason: str
    source: str
    alerts: list = Field(default_factory=list)
