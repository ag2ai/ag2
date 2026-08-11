# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared types for the eval framework.

:class:`Feedback` is the normalized output of every scorer — ``score`` for
numeric / boolean grades, ``value`` for categorical labels, ``comment`` for
free-form explanation. Plain ``bool`` / ``int`` / ``float`` / ``str`` returns
from scorers are wrapped into a :class:`Feedback` by the ``@scorer``
decorator; a scorer can also return a :class:`Feedback` (or list of them)
directly to set the key explicitly or attach a comment.
"""

from dataclasses import dataclass, field
from typing import Any, TypeAlias

__all__ = (
    "Feedback",
    "ScoreValue",
    "ScorerReturnTypeError",
    "ValueLabel",
)


ScoreValue: TypeAlias = bool | int | float | None
"""A numeric or boolean grade. ``True`` / ``1`` count as "pass" for pass-rate aggregates."""

ValueLabel: TypeAlias = str | None
"""A categorical label (e.g. ``"completed"`` / ``"timeout"``), used for slicing."""


@dataclass(frozen=True, slots=True)
class Feedback:
    """A single piece of feedback produced by a scorer.

    Exactly one of ``score`` or ``value`` is typically populated — ``score``
    for numeric / boolean grades, ``value`` for categorical labels. Both
    being ``None`` is valid and represents a "no signal" feedback (e.g. a
    scorer that crashed mid-evaluation).

    Args:
        key: Stable identifier for this feedback, usually the scorer's
            function name. Pass rates and stats on
            :class:`~ag2.eval.RunResult` are looked up by this key.
        score: Numeric or boolean grade.
        value: Categorical label, used for slicing aggregates.
        comment: Free-form human-readable explanation. Surfaces in run
            JSON; useful for LLM-as-judge rationales or scorer error traces.
        detail: Optional structured *evidence* behind this feedback — a
            JSON-safe mapping serialized into the run JSON for programmatic
            access (e.g. a failure attribution's decisive step / responsible
            agent, or a judge's per-order swap verdicts). Supplementary only:
            aggregation never reads ``detail`` — ``score`` / ``value`` remain
            the graded signal. Typed at the source (producers serialize a typed
            model into it), so it is evidence, not a grab-bag.
    """

    key: str
    score: ScoreValue = None
    value: ValueLabel = None
    comment: str | None = None
    detail: dict[str, Any] | None = field(default=None, compare=False)


class ScorerReturnTypeError(TypeError):
    """Raised when a scorer returns a value the ``@scorer`` decorator can't normalize.

    Supported return types: ``bool``, ``int``, ``float``, ``str``,
    :class:`Feedback`, ``list[Feedback]``, ``None``.
    """
