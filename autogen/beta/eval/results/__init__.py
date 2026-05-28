# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Results layer: per-run aggregation, persistence, and budget thresholds."""

from .budgets import BudgetThresholds
from .diff import RunDiff, RunsNotComparableError
from .result import Aggregates, RunResult, ScoreStats, TaskResult
from .store import load_run

__all__ = (
    "Aggregates",
    "BudgetThresholds",
    "RunDiff",
    "RunResult",
    "RunsNotComparableError",
    "ScoreStats",
    "TaskResult",
    "load_run",
)
