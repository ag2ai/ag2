# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Eval lifecycle events — published on a run's stream for live observation.

Pass a ``stream`` to :func:`~ag2.eval.run_agent` / :func:`~ag2.eval.evaluate_traces`
/ :func:`~ag2.eval.run_variants` and you can observe an evaluation exactly
like you observe an agent: subscribe to these events
(``stream.where(VariantCompleted).subscribe(...)``), or attach the ready-made
:func:`~ag2.eval.console_reporter`.

All are ``__transient__`` — observational only. The durable record of a run is
its persisted :class:`~ag2.eval.RunResult` JSON, not these events.
"""

from typing import TYPE_CHECKING

from ag2.events.base import BaseEvent, Field

from ._types import Feedback

if TYPE_CHECKING:
    from .pairwise import PairwiseRunResult
    from .results import RunResult

__all__ = (
    "EvalCompleted",
    "EvalEvent",
    "EvalStarted",
    "PairwiseCompared",
    "PairwiseCompleted",
    "PairwiseStarted",
    "TaskEvaluated",
    "VariantCompleted",
    "VariantStarted",
)


class EvalEvent(BaseEvent):
    """Base for eval lifecycle events — carries the run id and optional user label."""

    run_id: str
    label: str | None = Field(None)


class EvalStarted(EvalEvent):
    """Emitted once when a run begins."""

    __transient__ = True

    suite: str = Field("")
    total: int = Field(0)  # number of task-runs to execute (tasks x repeats)


class TaskEvaluated(EvalEvent):
    """Emitted when one task finishes and its scorers have run."""

    __transient__ = True

    task_id: str
    feedback: tuple[Feedback, ...] = Field(default_factory=tuple)
    variant: str | None = Field(None)  # set when produced inside a variant run


class EvalCompleted(EvalEvent):
    """Emitted once when a run finishes; carries the finished result."""

    __transient__ = True

    result: "RunResult"


class VariantStarted(EvalEvent):
    """Emitted before each variant's run in a sweep."""

    __transient__ = True

    variant: str
    index: int = Field(0)
    total: int = Field(0)


class VariantCompleted(EvalEvent):
    """Emitted after each variant's run in a sweep; carries that variant's result."""

    __transient__ = True

    variant: str
    result: "RunResult"


class PairwiseStarted(EvalEvent):
    """Emitted once when a pairwise comparison begins."""

    __transient__ = True

    variant_a: str = Field("")
    variant_b: str = Field("")
    total: int = Field(0)  # number of task_id-matched pairs to compare


class PairwiseCompared(EvalEvent):
    """Emitted when one comparator finishes one paired task (per case)."""

    __transient__ = True

    task_id: str
    key: str
    winner: str = Field("tie")  # "a" / "b" / "tie"


class PairwiseCompleted(EvalEvent):
    """Emitted once when a pairwise comparison finishes; carries the result."""

    __transient__ = True

    result: "PairwiseRunResult"
