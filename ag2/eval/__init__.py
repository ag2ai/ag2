# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""AG2 Beta evaluation framework.

Offline evaluation of ``ag2`` agents against curated datasets,
with prebuilt scorers, deterministic runs via ``TestConfig`` cassettes,
and persisted run JSON suitable for run-vs-run diffing.
"""

from ._types import Feedback, ScorerReturnTypeError
from .dataset import Suite, Task
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
from .results import (
    Aggregates,
    BudgetThresholds,
    RunDiff,
    RunResult,
    RunsNotComparableError,
    ScoreStats,
    TaskResult,
    load_run,
)
from .runtime import (
    LeaderboardRow,
    VariantRunResult,
    Variants,
    evaluate_traces,
    run_agent,
    run_pairwise,
    run_variants,
)
from .scorer import Scorer, scorer
from .sources import (
    DEFAULT_CONVENTIONS,
    AG2GenAIConvention,
    DirectoryTraceSource,
    InMemoryTraceSource,
    OpenInferenceConvention,
    SpanConvention,
    TempoTraceSource,
    TraceRef,
    TraceSource,
)
from .trace import Trace

__all__ = (
    "DEFAULT_CONVENTIONS",
    "AG2GenAIConvention",
    "Aggregates",
    "Agreement",
    "BudgetThresholds",
    "DirectoryTraceSource",
    "Feedback",
    "InMemoryTraceSource",
    "LeaderboardRow",
    "OpenInferenceConvention",
    "PairwiseCase",
    "PairwiseComparator",
    "PairwiseOutcome",
    "PairwiseRunResult",
    "RunDiff",
    "RunResult",
    "RunsNotComparableError",
    "ScoreStats",
    "Scorer",
    "ScorerReturnTypeError",
    "SpanConvention",
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
    "evaluate_pairwise",
    "evaluate_traces",
    "load_run",
    "run_agent",
    "run_pairwise",
    "run_variants",
    "scorer",
)
