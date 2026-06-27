# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Threshold combinator — turn a numeric scorer into a Pass/Fail gate.

``threshold`` wraps any scorer that produces a numeric grade and returns a new
:class:`~ag2.eval.Scorer` whose grade is a boolean pass/fail derived from that
number — for automation gating ("reject anything that fails")::

    from ag2.eval.scorers import agent_judge, threshold

    quality = agent_judge(config, criterion="The answer is helpful.", key="quality")
    gate = threshold(quality, at_least=0.7)  # scorer -> scorer

    # ...or the agent_judge convenience param (same thing):
    gate = agent_judge(config, criterion="...", key="quality", threshold=0.7)

A gated criterion emits ONE :class:`~ag2.eval.Feedback`: ``score`` is the boolean
(so it lands in ``result.pass_rate(key)`` and ``diff().regressions``), and the raw number +
bounds are recorded in ``detail`` (persisted in the run JSON). A feedback with no numeric
grade — the judge returned no verdict, or the scorer raised → ``score=None`` — is a **fail**.
Already-boolean and categorical (``value``) feedback passes through unchanged.

This mirrors DeepEval's / promptfoo's ``threshold`` and Braintrust's "pass threshold". For a
per-task token/time *resource* gate, see :class:`~ag2.eval.BudgetThresholds` — a
different axis.
"""

from typing import Any

from .._types import Feedback
from ..dataset import Task
from ..scorer import Scorer
from ..trace import Trace

__all__ = ("threshold",)


def threshold(
    scorer: Scorer,
    *,
    at_least: float | None = None,
    at_most: float | None = None,
    key: str | None = None,
) -> Scorer:
    """Wrap a numeric ``scorer`` into a Pass/Fail gate.

    Args:
        scorer: The scorer to gate. Its numeric feedback is converted to a boolean
            pass/fail; already-boolean, categorical, and no-signal feedback passes
            through unchanged.
        at_least: Inclusive lower bound — pass requires ``score >= at_least``.
        at_most: Inclusive upper bound — pass requires ``score <= at_most``.
        key: Feedback key for the gate. Defaults to the source feedback's key (the
            column simply becomes pass/fail).

    At least one of ``at_least`` / ``at_most`` must be set. A numeric feedback becomes
    ``Feedback(score=<pass bool>, detail={"score": <n>, ...})`` (the raw number is kept in
    ``detail``); a ``None`` score (ungradeable) becomes ``False`` (fail).
    """
    if at_least is None and at_most is None:
        raise ValueError("threshold(): set at_least and/or at_most")

    async def _threshold(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any] | None,
        trace: Trace,
        task: Task,
    ) -> list[Feedback]:
        feedbacks = await scorer(
            inputs=inputs,
            outputs=outputs,
            reference_outputs=reference_outputs,
            trace=trace,
            task=task,
        )
        return [_gate(fb, at_least, at_most, key) for fb in feedbacks]

    return Scorer(_threshold, key=key or scorer.key)


def _gate(fb: Feedback, at_least: float | None, at_most: float | None, key: str | None) -> Feedback:
    """Reshape one feedback: numeric → pass/fail, ungradeable → fail, else pass through."""
    out_key = key or fb.key
    bounds = {"at_least": at_least, "at_most": at_most}

    if isinstance(fb.score, (int, float)) and not isinstance(fb.score, bool):
        passed = (at_least is None or fb.score >= at_least) and (at_most is None or fb.score <= at_most)
        return Feedback(
            key=out_key,
            score=passed,
            comment=_comment(f"score {fb.score} {_bound_text(at_least, at_most)} → {'pass' if passed else 'fail'}", fb),
            detail={**(fb.detail or {}), "score": fb.score, **bounds},
        )

    if fb.score is None and fb.value is None:  # ungradeable → fail
        return Feedback(
            key=out_key,
            score=False,
            comment=_comment("ungradeable → fail", fb),
            detail={**(fb.detail or {}), "score": None, **bounds},
        )

    return fb  # already bool, or categorical: not a numeric grade to threshold


def _bound_text(at_least: float | None, at_most: float | None) -> str:
    parts = []
    if at_least is not None:
        parts.append(f">= {at_least}")
    if at_most is not None:
        parts.append(f"<= {at_most}")
    return " and ".join(parts)


def _comment(message: str, fb: Feedback) -> str:
    return f"{message}; {fb.comment}" if fb.comment else message
