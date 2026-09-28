# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Compaction verifier extension for AG2.

Measures whether a :class:`~ag2.compact.CompactStrategy` makes an agent's next
actions worse: at a boundary of a recorded run, the agent is resumed from the
raw history and from the compacted history, in an environment restored to the
same state, and the two are compared on blocked and repeated tool calls. This
is the TRACE method (Trajectory-Relative Agent Context comprEssion), expressed
in AG2's events.

Maintained by @wuliang211.
"""

from .actions import Action, actions_from_events, call_signature
from .boundary import (
    Boundary,
    CompactedContext,
    ContextSize,
    compact_context,
    make_boundary,
    resumable_cuts,
    select_cuts,
)
from .burden import (
    ArmBurden,
    BoundaryDelta,
    Burden,
    SignatureKey,
    arm_burden,
    burden,
    history_signatures,
    score_boundary,
)
from .control import IdentityCompact
from .environment import Environment, ReplayEnvironment, ReplayMismatchError
from .report import BoundaryResult, HorizonSummary, Rollout, StrategyReport, VerificationReport
from .runner import ActionBudget, CompactionVerifier, Recording, run_rollout
from .stats import Interval, bootstrap_mean, exact_permutation_test, exact_sign_test

__all__ = (
    "Action",
    "ActionBudget",
    "ArmBurden",
    "Boundary",
    "BoundaryDelta",
    "BoundaryResult",
    "Burden",
    "CompactedContext",
    "CompactionVerifier",
    "ContextSize",
    "Environment",
    "HorizonSummary",
    "IdentityCompact",
    "Interval",
    "Recording",
    "ReplayEnvironment",
    "ReplayMismatchError",
    "Rollout",
    "SignatureKey",
    "StrategyReport",
    "VerificationReport",
    "actions_from_events",
    "arm_burden",
    "bootstrap_mean",
    "burden",
    "call_signature",
    "compact_context",
    "exact_permutation_test",
    "exact_sign_test",
    "history_signatures",
    "make_boundary",
    "resumable_cuts",
    "run_rollout",
    "score_boundary",
    "select_cuts",
)
