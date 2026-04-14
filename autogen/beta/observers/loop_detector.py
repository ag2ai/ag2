# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""LoopDetector — detects repetitive tool-call patterns and alerts on potential loops."""

from collections import deque

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ToolCallEvent
from autogen.beta.events.alert import ObserverAlert, Severity
from autogen.beta.observer import BaseObserver
from autogen.beta.watch import EventWatch


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
    """

    def __init__(
        self,
        window_size: int = 10,
        repeat_threshold: int = 3,
        *,
        name: str = "loop-detector",
    ) -> None:
        super().__init__(name, watch=EventWatch(ToolCallEvent))
        self._window_size = window_size
        self._repeat_threshold = repeat_threshold
        self._history: deque[tuple[str, str]] = deque(maxlen=window_size)
        self._flagged: set[tuple[str, str]] = set()

    async def process(self, events: list[BaseEvent], ctx: Context) -> ObserverAlert | None:
        for event in events:
            if not isinstance(event, ToolCallEvent):
                continue

            key = (event.name, event.arguments)
            self._history.append(key)

            if len(self._history) < self._repeat_threshold:
                continue

            tail = list(self._history)[-self._repeat_threshold :]
            if all(k == key for k in tail) and key not in self._flagged:
                self._flagged.add(key)
                return ObserverAlert(
                    source=self.name,
                    severity=Severity.WARNING,
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
