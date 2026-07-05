# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Prebuilt scorers for cost-discipline checks."""

from ..scorer import Scorer
from ..trace import Trace

__all__ = ("token_budget",)


def token_budget(max_tokens: int) -> Scorer:
    """Pass iff a task's total ``input + output`` tokens stay at or under ``max_tokens``.

    The check is **per task** — ``trace.tokens.total`` is one task's usage. Cache
    tokens are excluded; they're reported separately on
    :class:`~ag2.eval.trace.TokenUsage` and priced differently by most
    providers. This emits a pass/fail signal into the run's pass-rate aggregate;
    for the same per-task limit recorded as a dedicated ``budget_violation`` count
    instead, use :class:`~ag2.eval.BudgetThresholds` (also observational —
    neither aborts the run).
    """

    def _check(trace: Trace) -> bool:
        return trace.tokens.total <= max_tokens

    return Scorer(_check, key="token_budget")
