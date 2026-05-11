# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Prebuilt scorers — frozen versions of the most common scorer patterns.

Each name below is a *factory* that returns a :class:`Scorer`. Drop
them straight into ``scorers=[...]``::

    from autogen.beta.eval.scorers import (
        tool_called,
        no_tool_errors,
        final_answer_matches,
        token_budget,
    )

    scorers = [
        tool_called("get_weather"),
        no_tool_errors(),
        final_answer_matches(field="city", matcher="contains"),
        token_budget(2_000),
    ]

These five are deliberately the only ones shipped in v0 — every new
prebuilt is an API surface decision that ought to wait for a real
user asking for it.
"""

from .correctness import final_answer_matches
from .cost import token_budget
from .tools import no_tool_errors, no_tool_not_found, tool_called

__all__ = (
    "final_answer_matches",
    "no_tool_errors",
    "no_tool_not_found",
    "token_budget",
    "tool_called",
)
