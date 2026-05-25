# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""AG2 Beta evaluation framework.

Offline evaluation of ``autogen.beta`` agents against curated datasets,
with prebuilt scorers, deterministic runs via ``TestConfig`` cassettes,
and persisted run JSON suitable for run-vs-run diffing.
"""

from ._types import Feedback, ScorerReturnTypeError
from .dataset import EvalTarget, Suite, Task
from .pairwise import (
    Agreement,
    PairwiseCase,
    PairwiseComparator,
    PairwiseOutcome,
    PairwiseRunResult,
    WinRate,
    evaluate_pairwise,
)
from .reporters import console_reporter
from .results import Aggregates, BudgetThresholds, RunResult, ScoreStats, TaskResult
from .runtime import (
    LeaderboardRow,
    VariantRunResult,
    Variants,
    evaluate,
    run,
    run_pairwise,
    run_variants,
)
from .scorer import Scorer, scorer
from .sources import DirectoryTraceSource, InMemoryTraceSource, TempoTraceSource, TraceRef, TraceSource
from .trace import Trace

__all__ = (
    "Aggregates",
    "Agreement",
    "BudgetThresholds",
    "DirectoryTraceSource",
    "EvalTarget",
    "Feedback",
    "InMemoryTraceSource",
    "LeaderboardRow",
    "PairwiseCase",
    "PairwiseComparator",
    "PairwiseOutcome",
    "PairwiseRunResult",
    "RunResult",
    "ScoreStats",
    "Scorer",
    "ScorerReturnTypeError",
    "Suite",
    "Task",
    "TaskResult",
    "TempoTraceSource",
    "Trace",
    "TraceRef",
    "TraceSource",
    "VariantRunResult",
    "Variants",
    "WinRate",
    "console_reporter",
    "evaluate",
    "evaluate_pairwise",
    "run",
    "run_pairwise",
    "run_variants",
    "scorer",
)
