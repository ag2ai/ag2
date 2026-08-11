# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Per-task budget thresholds — observational guardrails for the runner."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..trace import Trace

__all__ = ("BudgetThresholds",)


@dataclass(frozen=True, slots=True)
class BudgetThresholds:
    """Per-task budget thresholds.

    Budgets are **observational** in v0 — the runner records violations
    in ``RunResult.aggregates.budget_violations`` but never aborts a
    task that exceeds them. The count is intended as a regression signal
    in CI ("zero tasks may go over budget"), not as a kill switch.

    A field set to ``None`` (the default) means "no limit".

    Args:
        max_tokens_per_task: Maximum sum of input + output tokens across
            every model call in one task.
        max_seconds_per_task: Maximum wall-clock duration of one task,
            measured around ``agent.ask(...)``.
    """

    max_tokens_per_task: int | None = None
    max_seconds_per_task: float | None = None

    def exceeded_by(self, trace: "Trace") -> bool:
        """``True`` iff ``trace`` exceeds any set threshold (tokens or wall-clock)."""
        if self.max_tokens_per_task is not None and trace.tokens.total > self.max_tokens_per_task:
            return True
        return self.max_seconds_per_task is not None and trace.duration_ms / 1000 > self.max_seconds_per_task
