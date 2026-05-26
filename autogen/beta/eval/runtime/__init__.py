# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime layer: execute targets (live) or replay traces to produce results."""

from .evaluate import evaluate_traces
from .runner import run_agent, run_pairwise
from .variants import LeaderboardRow, VariantRunResult, Variants, run_variants

__all__ = (
    "LeaderboardRow",
    "VariantRunResult",
    "Variants",
    "evaluate_traces",
    "run_agent",
    "run_pairwise",
    "run_variants",
)
