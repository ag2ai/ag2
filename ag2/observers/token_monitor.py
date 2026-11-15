# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""TokenMonitor — tracks cumulative token usage and alerts when thresholds are exceeded."""

from ag2._import_utils import require_optional_import
from ag2.annotations import Context
from ag2.events import BaseEvent, ObserverAlert, Severity, Usage, UsageEvent
from ag2.usage import price_usage, usage_event_from
from ag2.watch import EventWatch

from .observer import BaseObserver


def _billable_tokens(usage: Usage) -> int:
    """Tokens to count against the budget.

    Every shipped provider mapper populates ``total_tokens``, but ``Usage`` does
    not synthesize it, so usage assembled by hand — a custom client, a fixture —
    would otherwise register as zero and leave the guard silently inert. The
    fallback lives here rather than on ``Usage`` itself: whether a synthesized
    total is honest is a question about the shared value type, not about this
    observer's budget arithmetic.

    A reported total *below* prompt plus completion is not believed. Summed
    usage — a sub-task rollup covers several calls — adds ``total_tokens``
    field-wise, so one call from a provider that omits the total drags the sum
    below the parts it is supposed to cover. Taking the larger of the two keeps
    a partial total from understating the budget, while still believing a
    provider whose total legitimately exceeds the two counts (reasoning tokens,
    for instance).

    Cache tokens are counted, once: ``prompt_tokens`` already includes the
    tokens read from and written to the cache on every provider (ADR 0036), so
    the cache fields are not added on top of it. See :class:`TokenMonitor` for
    what the guard's total does and does not cover.
    """
    counted = int(usage.prompt_tokens or 0) + int(usage.completion_tokens or 0)
    if usage.total_tokens is None:
        return counted
    return max(int(usage.total_tokens), counted)


@require_optional_import("llm_prices", "pricing")
def _require_pricing() -> None:
    """Fail at construction, not on the first event, when ``max_usd`` cannot be priced."""


class TokenMonitor(BaseObserver):
    """Tracks cumulative token usage and alerts when thresholds are exceeded.

    Observes ``UsageEvent`` — the framework's accounting record, emitted at the
    point tokens are spent by every unit of billable work: the main agent loop,
    descendant source records, history compaction, memory aggregation and live
    session clients — the same source ``UsageReport`` reads, so the guard sees
    every unit of work the report does.

    It counts every input token at the same weight, cached or not: the
    threshold bounds how much the model read and wrote, not what that cost. A
    heavily prompt-cached turn moves the guard by more than it bills, since
    cache reads are priced below fresh input. To bound the bill instead, pass
    ``max_usd``: every record is then priced with :func:`ag2.usage.price_usage`
    (the ``pricing`` extra), cache reads, cache writes and long-context tiers at
    their own prices, the same figure :meth:`ag2.usage.UsageReport.cost` reports.

    Deliberately *not* ``ModelResponse`` or ``TaskCompleted``. Those are derived
    events and reading them made the guard wrong three ways: a sub-task that
    billed and then failed was invisible (``TaskFailed`` carries no usage, and
    usage is a source record); ``TaskCompleted.usage`` is a *cumulative
    snapshot* of the sub-task's stream, so accumulating it with ``+=`` inflated
    the total whenever the stream was reused; and compaction, aggregation and
    live sessions report usage without ever producing a ``ModelResponse``.
    Reading ``ModelResponse`` *as well* would double-count every main-loop call,
    which emits both.

    Parameters
    ----------
    warn_threshold:
        Total tokens at which a WARNING alert is emitted.
    alert_threshold:
        Total tokens at which a CRITICAL alert is emitted.
    max_usd:
        Total list-price cost in USD at which a CRITICAL alert is emitted, or
        ``None`` (the default) to track tokens only. Requires the ``pricing``
        extra. Spend the price catalog cannot price (a local model, a model it
        does not list) is not counted; the first such record raises a WARNING
        alert naming the model, so the gap is never silent.
    name:
        Observer display name.
    timeout:
        Per-call limit for ``process`` in seconds, or ``None`` for no limit.
    """

    def __init__(
        self,
        warn_threshold: int = 50_000,
        alert_threshold: int = 100_000,
        *,
        max_usd: float | None = None,
        name: str = "token-monitor",
        timeout: float | None = None,
    ) -> None:
        if max_usd is not None:
            _require_pricing()
        super().__init__(name, watch=EventWatch(UsageEvent), timeout=timeout)
        self._warn_threshold = warn_threshold
        self._alert_threshold = alert_threshold
        self._max_usd = max_usd
        self._total_tokens: int = 0
        self._total_usd: float = 0.0
        self._unpriced: UsageEvent | None = None
        self._seen_usage: set[str] = set()
        self._warned = False
        self._alerted = False
        self._cost_alerted = False
        self._unpriced_warned = False

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_usd(self) -> float:
        """List-price cost counted so far; stays ``0.0`` unless ``max_usd`` is set."""
        return self._total_usd

    async def process(self, events: list[BaseEvent], ctx: Context) -> ObserverAlert | None:
        for event in events:
            record = usage_event_from(event)
            if record is not None and record.event_id not in self._seen_usage:
                self._seen_usage.add(record.event_id)
                usage = record.usage
                if usage:
                    self._total_tokens += _billable_tokens(usage)
                    if self._max_usd is not None:
                        self._count_cost(record)

        if self._max_usd is not None and not self._cost_alerted and self._total_usd >= self._max_usd:
            self._cost_alerted = True
            return ObserverAlert(
                source=self.name,
                severity=Severity.CRITICAL,
                message=(
                    f"Cost critical: ${self._total_usd:,.4f} at list price "
                    f"(limit: ${self._max_usd:,.2f}). "
                    "Consider wrapping up to control costs."
                ),
            )

        if not self._alerted and self._total_tokens >= self._alert_threshold:
            self._alerted = True
            return ObserverAlert(
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
            return ObserverAlert(
                source=self.name,
                severity=Severity.WARNING,
                message=(
                    f"Token usage warning: {self._total_tokens:,} tokens "
                    f"(threshold: {self._warn_threshold:,}). "
                    "Be mindful of remaining budget."
                ),
            )

        if self._unpriced is not None and not self._unpriced_warned:
            self._unpriced_warned = True
            return ObserverAlert(
                source=self.name,
                severity=Severity.WARNING,
                message=(
                    f"Cost not tracked for model {self._unpriced.model!r} "
                    f"(provider {self._unpriced.provider!r}): the price catalog has no price for it, "
                    "so max_usd does not cover its spend."
                ),
            )

        return None

    def _count_cost(self, record: UsageEvent) -> None:
        usd = price_usage(record.usage, model=record.model, provider=record.provider)
        if usd is None:
            if self._unpriced is None:
                self._unpriced = record
        else:
            self._total_usd += usd

    def reset(self) -> None:
        """Reset counters for a fresh session."""
        self._total_tokens = 0
        self._total_usd = 0.0
        self._unpriced = None
        self._seen_usage.clear()
        self._warned = False
        self._alerted = False
        self._cost_alerted = False
        self._unpriced_warned = False
