# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ModelResponse

from ..events import SatelliteFlag, Severity, TaskSatelliteResult
from ..satellite import BaseSatellite
from ..triggers import OnEvent


class TokenMonitor(BaseSatellite):
    """Tracks cumulative token usage and flags when thresholds are exceeded.

    Observes :class:`ModelResponse` and :class:`TaskSatelliteResult` events
    to aggregate usage across the planet agent and all task satellites.

    Parameters
    ----------
    warn_threshold:
        Total tokens at which a WARNING flag is emitted.
    alert_threshold:
        Total tokens at which a CRITICAL flag is emitted.
    name:
        Satellite display name.
    """

    def __init__(
        self,
        warn_threshold: int = 50_000,
        alert_threshold: int = 100_000,
        *,
        name: str = "token-monitor",
    ) -> None:
        super().__init__(name, trigger=OnEvent(ModelResponse | TaskSatelliteResult))
        self._warn_threshold = warn_threshold
        self._alert_threshold = alert_threshold
        self._total_tokens: int = 0
        self._warned = False
        self._alerted = False

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    async def process(
        self, events: list[BaseEvent], ctx: Context
    ) -> SatelliteFlag | None:
        for event in events:
            if isinstance(event, ModelResponse):
                usage = event.usage or {}
                self._total_tokens += int(usage.get("total_tokens", 0))
            elif isinstance(event, TaskSatelliteResult):
                usage = event.usage or {}
                self._total_tokens += int(usage.get("total_tokens", 0))

        if not self._alerted and self._total_tokens >= self._alert_threshold:
            self._alerted = True
            return SatelliteFlag(
                source=self.name,
                severity=Severity.CRITICAL,
                message=(
                    f"Token usage critical: {self._total_tokens:,} tokens "
                    f"(threshold: {self._alert_threshold:,}). "
                    "Consider wrapping up to control costs."
                ),
            )

        if not self._warned and self._total_tokens >= self._warn_threshold:
            self._warned = True
            return SatelliteFlag(
                source=self.name,
                severity=Severity.WARNING,
                message=(
                    f"Token usage warning: {self._total_tokens:,} tokens "
                    f"(threshold: {self._warn_threshold:,}). "
                    "Be mindful of remaining budget."
                ),
            )

        return None

    def reset(self) -> None:
        """Reset counters for a fresh session."""
        self._total_tokens = 0
        self._warned = False
        self._alerted = False

    def detach(self) -> None:
        super().detach()
