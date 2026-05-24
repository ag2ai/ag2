# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""AG2 Beta evaluation framework.

Offline evaluation of ``autogen.beta`` agents against curated datasets,
with prebuilt scorers, deterministic runs via ``TestConfig`` cassettes,
and persisted run JSON suitable for run-vs-run diffing.
"""

from ._types import Feedback, ScorerReturnTypeError
from .budgets import BudgetThresholds
from .evaluate import evaluate
from .result import Aggregates, RunResult, ScoreStats, TaskResult
from .runner import run
from .scorer import Scorer, scorer
from .suite import Suite
from .target import EvalTarget
from .task import Task
from .tempo import TempoTraceSource
from .trace import Trace
from .trace_source import DirectoryTraceSource, InMemoryTraceSource, TraceRef, TraceSource

__all__ = (
    "Aggregates",
    "BudgetThresholds",
    "DirectoryTraceSource",
    "EvalTarget",
    "Feedback",
    "InMemoryTraceSource",
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
    "evaluate",
    "run",
    "scorer",
)
