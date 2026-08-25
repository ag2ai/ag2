# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""TokenMonitor — tracks cumulative token usage and alerts when thresholds are exceeded."""

from ag2.annotations import Context
from ag2.events import BaseEvent, ObserverAlert, Severity, Usage, UsageEvent
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

    Cache tokens are left out, as they were before this read ``UsageEvent``.
    That understates a prompt-cached turn on any provider whose prompt count
    excludes them, and changing it would move every cached workload's threshold
    — a behaviour change that belongs on its own, not folded into a change of
    accounting *source*. See :class:`TokenMonitor` for what the guard's total
    does and does not cover.
    """
    counted = int(usage.prompt_tokens or 0) + int(usage.completion_tokens or 0)
    if usage.total_tokens is None:
        return counted
    return max(int(usage.total_tokens), counted)


class TokenMonitor(BaseObserver):
    """Tracks cumulative token usage and alerts when thresholds are exceeded.

    Observes ``UsageEvent`` — the framework's accounting record, emitted at the
    point tokens are spent by every unit of billable work: the main agent loop,
    each sub-task rollup, history compaction, memory aggregation and the live
    session clients — the same source ``UsageReport`` reads, so the guard sees
    every unit of work the report does.

    It does not count the same *tokens*: :func:`_billable_tokens` reads the
    prompt and completion counts and ignores the cache fields, which
    ``UsageReport.total`` keeps. On a provider whose reported prompt count
    excludes cached tokens — Anthropic's ``input_tokens`` does — a heavily
    prompt-cached turn moves this guard by a fraction of what it bills. Treat
    the threshold as a bound on uncached spend, and read ``UsageReport`` for the
    bill.

    Deliberately *not* ``ModelResponse`` or ``TaskCompleted``. Those are derived
    events and reading them made the guard wrong three ways: a sub-task that
    billed and then failed was invisible (``TaskFailed`` carries no usage, and
    the rollup is a ``UsageEvent``); ``TaskCompleted.usage`` is a *cumulative
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
    name:
        Observer display name.
    """

    def __init__(
        self,
        warn_threshold: int = 50_000,
        alert_threshold: int = 100_000,
        *,
        name: str = "token-monitor",
    ) -> None:
        super().__init__(name, watch=EventWatch(UsageEvent))
        self._warn_threshold = warn_threshold
        self._alert_threshold = alert_threshold
        self._total_tokens: int = 0
        self._warned = False
        self._alerted = False

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    async def process(self, events: list[BaseEvent], ctx: Context) -> ObserverAlert | None:
        for event in events:
            if isinstance(event, UsageEvent):
                usage = event.usage
                if usage:
                    self._total_tokens += _billable_tokens(usage)

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

        return None

    def reset(self) -> None:
        """Reset counters for a fresh session."""
        self._total_tokens = 0
        self._warned = False
        self._alerted = False
