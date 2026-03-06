# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections import deque

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ToolCall

from ..events import SatelliteFlag, Severity
from ..satellite import BaseSatellite
from ..triggers import OnEvent


class LoopDetector(BaseSatellite):
    """Detects repetitive tool-call patterns and flags potential loops.

    Watches :class:`ToolCall` events and maintains a sliding window.
    Flags a WARNING when *repeat_threshold* consecutive identical calls
    (same tool name and arguments) are observed.

    Parameters
    ----------
    window_size:
        Number of recent tool calls to keep.
    repeat_threshold:
        Number of identical consecutive calls that trigger a flag.
    name:
        Satellite display name.
    """

    def __init__(
        self,
        window_size: int = 10,
        repeat_threshold: int = 3,
        *,
        name: str = "loop-detector",
    ) -> None:
        super().__init__(name, trigger=OnEvent(ToolCall))
        self._window_size = window_size
        self._repeat_threshold = repeat_threshold
        self._history: deque[tuple[str, str]] = deque(maxlen=window_size)
        self._flagged: set[tuple[str, str]] = set()

    async def process(
        self, events: list[BaseEvent], ctx: Context
    ) -> SatelliteFlag | None:
        for event in events:
            if not isinstance(event, ToolCall):
                continue

            key = (event.name, event.arguments)
            self._history.append(key)

            if len(self._history) < self._repeat_threshold:
                continue

            # Check consecutive repeats at the tail
            tail = list(self._history)[-self._repeat_threshold :]
            if all(k == key for k in tail) and key not in self._flagged:
                self._flagged.add(key)
                return SatelliteFlag(
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

    def detach(self) -> None:
        super().detach()
