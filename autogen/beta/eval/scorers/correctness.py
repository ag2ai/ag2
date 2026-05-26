# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Prebuilt scorers for final-answer correctness."""

from typing import Any, Literal

from ..scorer import Scorer

__all__ = ("final_answer_matches",)


_Matcher = Literal["exact", "casefold", "contains"]


def final_answer_matches(
    field: str = "answer",
    *,
    matcher: _Matcher = "casefold",
) -> Scorer:
    """Pass iff the agent's final answer matches the task's reference output.

    Reads ``reference_outputs[field]`` for the expected value and compares it
    against the agent's final answer: ``outputs[field]`` when a structured field
    of that name is present, otherwise ``outputs["body"]`` — the final response
    text, which is what the trace projection provides today.

    Args:
        field: Key to read from ``reference_outputs`` and (when
            available) from the agent's structured ``outputs``.
            Defaults to ``"answer"``.
        matcher: How to compare the two strings:

            * ``"exact"`` — strict equality (good for closed-form
              answers in structured outputs).
            * ``"casefold"`` — case-insensitive equality (the default;
              forgives capitalization drift).
            * ``"contains"`` — pass iff ``expected`` appears as a
              substring of ``actual`` (good when the reply is a
              sentence and the reference is a single field value).

    The scorer is reference-based: it returns ``False`` when no
    reference output is available for ``field`` rather than raising,
    so an unreferenced task simply scores as a fail. Wrap your own
    scorer if you'd rather skip such tasks.
    """

    def _check(outputs: dict[str, Any], reference_outputs: dict[str, Any] | None) -> bool:
        if reference_outputs is None or field not in reference_outputs:
            return False
        expected = reference_outputs[field]
        actual = outputs.get(field, outputs.get("body"))
        if actual is None:
            return False
        if matcher == "exact":
            return actual == expected
        if matcher == "casefold":
            return str(actual).casefold() == str(expected).casefold()
        return str(expected) in str(actual)

    return Scorer(_check, key="final_answer_matches")
