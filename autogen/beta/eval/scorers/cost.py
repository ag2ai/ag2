# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Prebuilt scorers for cost-discipline checks."""

from ..scorer import Scorer
from ..trace import Trace

__all__ = ("token_budget",)


def token_budget(max_tokens: int) -> Scorer:
    """Pass iff total ``input + output`` tokens across the run stay at or under ``max_tokens``.

    Cache tokens are not counted — they are reported separately on
    :class:`~autogen.beta.eval.trace.TokenUsage` and (in most providers'
    pricing) charged at a different rate. If you want to enforce a
    *hard* per-task cap that aborts the run, use
    :class:`~autogen.beta.eval.BudgetThresholds` instead; this scorer
    only records a pass/fail signal that flows into aggregates.
    """

    def _check(trace: Trace) -> bool:
        return trace.tokens.total <= max_tokens

    return Scorer(_check, key="token_budget")
