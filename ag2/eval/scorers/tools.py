# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Prebuilt scorers for tool-use correctness.

These are factory functions: each call returns a :class:`Scorer` ready
to drop into ``scorers=[...]``::

    from ag2.eval.scorers import tool_called, no_tool_errors

    scorers = [
        tool_called("get_weather"),
        no_tool_errors(),
    ]

The closures inside each factory are created once per scorer instance
(at user setup time), not per task — same shape as a decorator factory,
which AGENTS.md explicitly permits.
"""

from ag2.events import ToolCallEvent, ToolErrorEvent

from ..scorer import Scorer
from ..trace import Trace

__all__ = (
    "no_tool_errors",
    "tool_called",
)


def tool_called(name: str, *, exactly: int | None = None) -> Scorer:
    """Pass iff the agent called the tool ``name``.

    Args:
        name: The tool name to look for in ``ToolCallEvent.name``.
        exactly: If set, pass iff the call count equals this value.
            Default (``None``) means "at least one call".

    Returns:
        A :class:`Scorer` with key ``"tool_called[<name>]"`` so multiple
        instances coexist in one run without clashing.
    """

    def _check(trace: Trace) -> bool:
        count = len(trace.events_of(ToolCallEvent, name=name))
        if exactly is not None:
            return count == exactly
        return count >= 1

    return Scorer(_check, key=f"tool_called[{name}]")


def no_tool_errors() -> Scorer:
    """Pass iff zero :class:`ToolErrorEvent`\\ s fired during the run."""

    def _check(trace: Trace) -> bool:
        return len(trace.events_of(ToolErrorEvent)) == 0

    return Scorer(_check, key="no_tool_errors")
