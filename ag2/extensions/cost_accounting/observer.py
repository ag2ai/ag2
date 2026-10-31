# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Observer integration for estimated cost accounting."""

from decimal import Decimal
from typing import Any

from ag2.annotations import Context
from ag2.events import BaseEvent, ObserverAlert, Severity, UsageEvent
from ag2.observers import BaseObserver
from ag2.watch import EventWatch

from .estimator import CostEstimate, UsageCostEstimator


def _decimal(value: Decimal | float | int | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


class CostAccountingObserver(BaseObserver):
    """Tracks estimated cost from ``UsageEvent`` and emits threshold alerts.

    ``UsageEvent`` is AG2's source of truth for token accounting, so this
    observer avoids double-counting model responses and sub-agent rollups.
    """

    def __init__(
        self,
        estimator: UsageCostEstimator,
        *,
        warn_threshold_usd: Decimal | float | int | str = Decimal("1"),
        alert_threshold_usd: Decimal | float | int | str = Decimal("5"),
        name: str = "cost-accounting",
    ) -> None:
        super().__init__(name, watch=EventWatch(UsageEvent))
        self._estimator = estimator
        self._warn_threshold = _decimal(warn_threshold_usd)
        self._alert_threshold = _decimal(alert_threshold_usd)
        self._total = CostEstimate(total_cost_usd=Decimal("0"))
        self._warned = False
        self._alerted = False

    @property
    def estimate(self) -> CostEstimate:
        return self._total

    async def process(self, events: list[BaseEvent], ctx: Context) -> ObserverAlert | None:
        for event in events:
            if isinstance(event, UsageEvent) and event.usage:
                self._total += self._estimator.estimate_usage(
                    event.usage,
                    model=event.model,
                    provider=event.provider,
                )

        if not self._alerted and self._total.total_cost_usd >= self._alert_threshold:
            self._alerted = True
            return ObserverAlert(
                source=self.name,
                severity=Severity.CRITICAL,
                message=(
                    f"Estimated usage cost critical: ${self._total.total_cost_usd:.6f} "
                    f"(threshold: ${self._alert_threshold:.6f})."
                ),
                data=self._alert_data(),
            )

        if not self._warned and self._total.total_cost_usd >= self._warn_threshold:
            self._warned = True
            return ObserverAlert(
                source=self.name,
                severity=Severity.WARNING,
                message=(
                    f"Estimated usage cost warning: ${self._total.total_cost_usd:.6f} "
                    f"(threshold: ${self._warn_threshold:.6f})."
                ),
                data=self._alert_data(),
            )

        return None

    def reset(self) -> None:
        self._total = CostEstimate(total_cost_usd=Decimal("0"))
        self._warned = False
        self._alerted = False

    def _alert_data(self) -> dict[str, Any]:
        return {
            "total_cost_usd": str(self._total.total_cost_usd),
            "input_cost_usd": str(self._total.input_cost_usd),
            "output_cost_usd": str(self._total.output_cost_usd),
            "cache_read_input_cost_usd": str(self._total.cache_read_input_cost_usd),
            "cache_creation_input_cost_usd": str(self._total.cache_creation_input_cost_usd),
            "thinking_cost_usd": str(self._total.thinking_cost_usd),
            "pricing_source": self._total.pricing_source,
            "catalog_source": self._total.catalog_source,
            "catalog_version": self._total.catalog_version,
            "warnings": list(self._total.warnings),
        }
