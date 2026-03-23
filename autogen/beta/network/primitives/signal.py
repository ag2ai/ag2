# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Signal primitives — structured notifications from Observers to Actors.

A Signal carries severity, source identification, and a human/LLM-readable
message. SignalPolicy defines how signals are delivered to the actor.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Protocol, runtime_checkable

from autogen.beta.context import Context
from autogen.beta.events.base import BaseEvent, Field


class Severity(str, Enum):
    """Severity levels for signals. Extensible via string values."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class Signal(BaseEvent):
    """Structured notification from an observer."""

    source: str  # Observer name that produced this signal
    severity: str  # Severity level (uses Severity enum values, but accepts any string)
    message: str  # Human/LLM-readable description
    data: dict = Field(default_factory=dict)  # Optional structured payload


class HaltEvent(BaseEvent):
    """Emitted when a FATAL signal triggers execution halt."""

    reason: str
    source: str
    signals: list = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Signal delivery policies
# ---------------------------------------------------------------------------


@runtime_checkable
class SignalPolicy(Protocol):
    """Defines how signals are delivered to the actor."""

    async def deliver(self, signals: list[Signal], context: Context) -> None: ...


class InjectToPrompt:
    """Default signal policy. Injects signals into the LLM system prompt.

    - INFO/WARNING/CRITICAL: appended as alert text, removed after LLM call.
    - FATAL: halts execution immediately via stream interrupter. The agent
      returns with the FATAL signal message as the response. No LLM call.

    This uses MemoryStream's existing _interrupters mechanism — FATAL signals
    register an interrupter that short-circuits the next event wait, causing
    _execute_turn to exit the tool loop and return.
    """

    async def deliver(self, signals: list[Signal], context: Context) -> None:
        fatal = [s for s in signals if s.severity == Severity.FATAL]
        if fatal:
            await context.send(
                HaltEvent(
                    reason=f"FATAL: {fatal[0].message}",
                    source=fatal[0].source,
                    signals=fatal,
                )
            )
            return

        # Non-fatal: inject into prompt temporarily
        alert_text = self._format_alerts(signals)
        if alert_text:
            context.prompt.append(alert_text)

    @staticmethod
    def _format_alerts(signals: list[Signal]) -> str:
        if not signals:
            return ""
        lines = ["[OBSERVER MONITORING ALERTS]"]
        for s in signals:
            level = s.severity.upper() if isinstance(s.severity, str) else str(s.severity)
            lines.append(f"- [{level}] ({s.source}): {s.message}")
        return "\n".join(lines)


class EmitToStream:
    """Emit signals as events on the stream. Observer pattern — subscribers react."""

    async def deliver(self, signals: list[Signal], context: Context) -> None:
        for signal in signals:
            await context.send(signal)


class CallHandler:
    """Call a handler function with collected signals."""

    def __init__(self, handler: Callable[[list[Signal]], Awaitable[None]]) -> None:
        self._handler = handler

    async def deliver(self, signals: list[Signal], context: Context) -> None:
        await self._handler(signals)


class HaltOnFatal:
    """Wraps another policy. Halts on FATAL, delegates rest to inner policy.

    Example::

        policy = HaltOnFatal(inner=EmitToStream())
    """

    def __init__(self, inner: SignalPolicy | None = None) -> None:
        self._inner = inner or InjectToPrompt()

    async def deliver(self, signals: list[Signal], context: Context) -> None:
        fatal = [s for s in signals if s.severity == Severity.FATAL]
        non_fatal = [s for s in signals if s.severity != Severity.FATAL]

        # Deliver non-fatal signals first (even if fatal is present)
        if non_fatal:
            await self._inner.deliver(non_fatal, context)

        if fatal:
            await context.send(
                HaltEvent(
                    reason=f"FATAL: {fatal[0].message}",
                    source=fatal[0].source,
                    signals=fatal,
                )
            )
