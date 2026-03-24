# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""TelemetryPlugin — system plugin that tracks delegation metrics."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from autogen.beta.annotations import Context
from autogen.beta.events.conditions import TypeCondition

from ..events import DelegationRequest, DelegationResult
from ..topology import BasePlugin

if TYPE_CHECKING:
    from ..hub import Hub


@dataclass
class DelegationMetrics:
    """Aggregated metrics for delegations."""

    total_delegations: int = 0
    total_completions: int = 0
    by_source: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_target: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_delegation_time: float | None = None


class TelemetryPlugin(BasePlugin):
    """System plugin that tracks delegation traffic on the Hub stream.

    Does NOT sit in the delegation path — it observes via Hub.stream.
    Access metrics via the ``metrics`` property.

    Example::

        telemetry = TelemetryPlugin()
        hub = Hub(plugins=[telemetry])
        # ... delegations happen ...
        print(telemetry.metrics.total_delegations)
    """

    def __init__(self) -> None:
        self.metrics = DelegationMetrics()
        self._hub: Hub | None = None
        self._sub_ids: list = []

    def install(self, hub: Hub) -> None:
        self._hub = hub
        self._sub_ids.append(
            hub.stream.subscribe(
                self._on_request,
                condition=TypeCondition(DelegationRequest),
            )
        )
        self._sub_ids.append(
            hub.stream.subscribe(
                self._on_result,
                condition=TypeCondition(DelegationResult),
            )
        )

    def uninstall(self) -> None:
        if self._hub:
            for sub_id in self._sub_ids:
                self._hub.stream.unsubscribe(sub_id)
        self._sub_ids.clear()
        self._hub = None

    async def _on_request(self, event: DelegationRequest, ctx: Context) -> None:
        self.metrics.total_delegations += 1
        self.metrics.by_source[event.source] += 1
        self.metrics.by_target[event.target] += 1
        self.metrics.last_delegation_time = time.monotonic()

    async def _on_result(self, event: DelegationResult, ctx: Context) -> None:
        self.metrics.total_completions += 1
