# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Prebuilt scorers for cost-discipline checks."""

from ag2._import_utils import require_optional_import
from ag2.usage import UsageReport

from .._types import Feedback
from ..scorer import Scorer
from ..trace import Trace

__all__ = (
    "cost_budget",
    "token_budget",
)


def token_budget(max_tokens: int) -> Scorer:
    """Pass iff a task's total ``input + output`` tokens stay at or under ``max_tokens``.

    The check is **per task** — ``trace.tokens.total`` is one task's usage. Cache
    reads and writes count, once, as part of ``input``; their split is reported
    on :class:`~ag2.eval.trace.TokenUsage` because most providers price them
    differently (:func:`cost_budget` bounds the priced bill instead). This
    emits a pass/fail signal into the run's pass-rate aggregate; for the same
    per-task limit recorded as a dedicated ``budget_violation`` count instead, use :class:`~ag2.eval.BudgetThresholds` (also observational —
    neither aborts the run).
    """

    def _check(trace: Trace) -> bool:
        return trace.tokens.total <= max_tokens

    return Scorer(_check, key="token_budget")


@require_optional_import("llm_prices", "pricing")
def cost_budget(max_usd: float) -> Scorer:
    """Pass iff a task's list-price cost stays at or under ``max_usd``.

    The check is **per task**, like :func:`token_budget`, but on dollars: every
    model call is priced on its own with :meth:`ag2.usage.UsageReport.cost`, so
    cache reads, cache writes (default and extended retention) and long-context
    tiers bill at the provider's prices rather than as fresh input. The cost is
    kept in ``Feedback.detail["usd"]``.

    When some of the task's spend cannot be priced (a local model, a model the
    catalog does not list), the scorer cannot tell whether the task stayed in
    budget and returns no grade (``score=None``), naming the unpriced models in
    the comment. Requires the ``pricing`` extra.
    """

    def _check(trace: Trace) -> Feedback:
        cost = UsageReport.from_events(trace.events).cost()
        detail = {"usd": cost.total_usd, "max_usd": max_usd}
        if cost.unpriced:
            models = sorted({f"{r.provider}/{r.model}" for r in cost.unpriced})
            return Feedback(
                key="cost_budget",
                comment=f"no price for {', '.join(models)}; the priced calls alone cost ${cost.total_usd:.6f}",
                detail=detail,
            )
        return Feedback(key="cost_budget", score=cost.total_usd <= max_usd, detail=detail)

    return Scorer(_check, key="cost_budget")
