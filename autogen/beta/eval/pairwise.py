# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Pairwise comparison — compare two agent variants (A vs B) on the same task.

The unit is a :class:`PairwiseComparator`: given a task and the two variants'
traces, it returns a :class:`PairwiseOutcome` (which won, or tie). *Who* decides
is swappable behind the protocol — an LLM judge (``pairwise_judge``), a human
(``human_labels`` / inline HITL), or a user's own implementation. Each
comparator encapsulates its own position strategy (the LLM judge runs a
dual-order swap; a human gets one blinded order), so the runner just calls
``compare()`` and tallies.

This module holds the stable contracts; ``evaluate_pairwise`` / ``run_pairwise``
and the win-rate aggregation build on them.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from .task import Task
from .trace import Trace

__all__ = (
    "PairwiseComparator",
    "PairwiseOutcome",
)


@dataclass(frozen=True, slots=True)
class PairwiseOutcome:
    """The result of comparing variant A against variant B on one task.

    ``winner`` is ``"a"`` / ``"b"`` / ``"tie"``. ``detail`` carries audit info
    such as the per-order verdicts behind a swapped LLM judgment.
    """

    winner: Literal["a", "b", "tie"]
    reasoning: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class PairwiseComparator(Protocol):
    """Decides A vs B for one task. LLM / human / custom implementations interchange."""

    key: str
    """Feedback key this comparator reports under (its column in the result)."""

    async def compare(
        self,
        *,
        task: Task,
        trace_a: Trace,
        trace_b: Trace,
        reference_outputs: dict[str, Any] | None,
    ) -> PairwiseOutcome: ...
