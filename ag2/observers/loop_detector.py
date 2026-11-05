# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""LoopDetector — detects repetitive tool-call patterns and alerts on potential loops."""

from collections import deque

from ag2.annotations import Context
from ag2.events import BaseEvent, ChildEvent, ObserverAlert, Severity, ToolCallEvent
from ag2.watch import EventWatch

from .observer import BaseObserver


class LoopDetector(BaseObserver):
    """Detects repetitive tool-call patterns and alerts on potential loops.

    Watches ToolCallEvent events and maintains a sliding window.
    Emits a WARNING alert when ``repeat_threshold`` consecutive identical calls
    (same tool name and arguments) are observed.

    Parameters
    ----------
    window_size:
        Number of recent tool calls to keep.
    repeat_threshold:
        Number of identical consecutive calls that trigger an alert.
    name:
        Observer display name.
    timeout:
        Per-call limit for ``process`` in seconds, or ``None`` for no limit.
    """

    def __init__(
        self,
        window_size: int = 10,
        repeat_threshold: int = 3,
        *,
        name: str = "loop-detector",
        timeout: float | None = None,
    ) -> None:
        super().__init__(name, watch=EventWatch(ToolCallEvent | ChildEvent), timeout=timeout)
        self._window_size = window_size
        self._repeat_threshold = repeat_threshold
        self._history: dict[str, deque[tuple[str, str]]] = {}
        self._flagged: set[tuple[str, str, str]] = set()

    async def process(self, events: list[BaseEvent], ctx: Context) -> ObserverAlert | None:
        for envelope in events:
            origin = envelope.task_id if isinstance(envelope, ChildEvent) else "local"
            event = envelope.source_event
            if not isinstance(event, ToolCallEvent):
                continue

            key = (event.name, event.arguments)
            if origin not in self._history:
                self._history[origin] = deque(maxlen=self._window_size)
            history = self._history[origin]
            history.append(key)

            if len(history) < self._repeat_threshold:
                continue

            tail = list(history)[-self._repeat_threshold :]
            if all(k == key for k in tail) and (origin, *key) not in self._flagged:
                self._flagged.add((origin, *key))
                return ObserverAlert(
                    source=self.name,
                    severity=Severity.WARNING,
                    data={"task_id": origin, "path": envelope.path} if isinstance(envelope, ChildEvent) else {},
                    message=(
                        f"Potential loop detected: tool '{event.name}' called "
                        f"{self._repeat_threshold} times consecutively with "
                        f"identical arguments. Consider a different approach."
                    ),
                )

        return None

    def reset(self) -> None:
        """Reset state for a fresh session."""
        self._history.clear()
        self._flagged.clear()
