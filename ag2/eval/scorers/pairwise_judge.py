# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Pairwise Agent-as-judge — an LLM :class:`PairwiseComparator`.

``pairwise_judge`` compares two responses against one criterion and returns a
:class:`~ag2.eval.PairwiseOutcome`. It defends against LLM positional
bias with a **dual-order swap**: it asks the judge twice with the responses
swapped and only declares a winner if the *same* answer wins in both orders —
a flip resolves to a tie. The judge's per-call schema is position-based
(``first`` / ``second``, never A/B) so order can't leak, and has no optional
fields (OpenAI strict structured-output compatibility).

Single-purpose, like ``agent_judge``: one criterion, one ``key``. A
multi-criteria pairwise scorecard is a list of these.
"""

import json
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field

from ag2.agent import Agent
from ag2.config import ModelConfig
from ag2.events import ModelResponse
from ag2.middleware.base import MiddlewareFactory

from ..dataset import Task
from ..pairwise import PairwiseComparator, PairwiseOutcome
from ..trace import Trace
from .judge import _render_trajectory

__all__ = (
    "PairwiseVerdict",
    "pairwise_judge",
)


class PairwiseVerdict(BaseModel):
    """One judge call's preference, by position (not variant). Both fields required."""

    preferred: Literal["first", "second", "tie"] = Field(
        description="Which response is better on the criterion: 'first', 'second', or 'tie'."
    )
    reasoning: str = Field(description="A brief justification.")


def pairwise_judge(
    config: ModelConfig,
    *,
    criterion: str,
    key: str,
    include_trace: bool = False,
    include_reference: bool = True,
    retries: int = 1,
    swap: bool = True,
    middleware: Iterable[MiddlewareFactory] = (),
) -> PairwiseComparator:
    """Build an LLM pairwise comparator for one criterion.

    Args:
        config: Judge model config (pin temperature 0; use a different model
            family than the variants to avoid self-preference bias).
        criterion: The single standard to compare on, in plain English.
        key: Result column this comparator reports under.
        include_trace: Render each response's tool-call trajectory into the prompt.
        include_reference: When ``True`` (default), render the reference answer
            into the prompt as a ``## Reference`` section whenever
            ``reference_outputs`` is present. Set ``False`` for dimensions that must
            judge the responses on their own (e.g. faithfulness / grounding), so the
            golden answer cannot leak into the comparison.
        retries: ``content()`` re-asks on schema-validation failure.
        swap: Run the dual-order position-swap (default, recommended). When
            ``False``, a single call is used (faster, position-biased).
        middleware: Middleware for the judge agent (e.g. ``TelemetryMiddleware``).
    """
    judge = Agent(
        f"pairwise_judge_{key}",
        _system_prompt(criterion),
        config=config,
        response_schema=PairwiseVerdict,
        middleware=middleware,
    )
    return _PairwiseJudge(
        judge, key, include_trace=include_trace, include_reference=include_reference, retries=retries, swap=swap
    )


class _PairwiseJudge:
    """A :class:`PairwiseComparator` backed by a judge :class:`Agent`."""

    def __init__(
        self, judge: Agent, key: str, *, include_trace: bool, include_reference: bool, retries: int, swap: bool
    ) -> None:
        self._judge = judge
        self.key = key
        self._include_trace = include_trace
        self._include_reference = include_reference
        self._retries = retries
        self._swap = swap

    async def compare(
        self,
        *,
        task: Task,
        trace_a: Trace,
        trace_b: Trace,
        reference_outputs: dict[str, Any] | None,
    ) -> PairwiseOutcome:
        answer_a, answer_b = _final_text(trace_a), _final_text(trace_b)

        v1 = await self._verdict(task, reference_outputs, answer_a, answer_b, trace_a, trace_b)
        if not self._swap:
            return PairwiseOutcome(
                winner=_pref_to_ab(v1.preferred, "a", "b"), reasoning=v1.reasoning, detail={"order1": v1.preferred}
            )

        v2 = await self._verdict(task, reference_outputs, answer_b, answer_a, trace_b, trace_a)
        w1 = _pref_to_ab(v1.preferred, "a", "b")  # order 1: Response 1 = A
        w2 = _pref_to_ab(v2.preferred, "b", "a")  # order 2: Response 1 = B
        winner = w1 if (w1 == w2 and w1 != "tie") else "tie"  # conservative: win only if consistent
        return PairwiseOutcome(
            winner=winner,
            reasoning=v1.reasoning,
            detail={"order1": v1.preferred, "order2": v2.preferred},
        )

    async def _verdict(
        self,
        task: Task,
        reference_outputs: dict[str, Any] | None,
        first: str,
        second: str,
        trace_first: Trace,
        trace_second: Trace,
    ) -> PairwiseVerdict:
        prompt = _render(
            task,
            reference_outputs,
            first,
            second,
            trace_first,
            trace_second,
            self._include_trace,
            self._include_reference,
        )
        reply = await self._judge.ask(prompt)
        verdict = await reply.content(retries=self._retries)
        return (
            verdict if verdict is not None else PairwiseVerdict(preferred="tie", reasoning="judge returned no verdict")
        )


def _pref_to_ab(preferred: str, first: str, second: str) -> str:
    if preferred == "first":
        return first
    if preferred == "second":
        return second
    return "tie"


def _final_text(trace: Trace) -> str:
    responses = trace.events_of(ModelResponse)
    if responses and responses[-1].content is not None:
        return responses[-1].content
    return "(no answer)"


def _system_prompt(criterion: str) -> str:
    return (
        "You are comparing two AI responses against a single criterion. "
        f"Criterion: {criterion}\n"
        'Decide which response is better on THIS criterion only: answer "first", "second", or "tie". '
        "Do not favor a response for its position or its length."
    )


def _render(
    task: Task,
    reference_outputs: dict[str, Any] | None,
    first: str,
    second: str,
    trace_first: Trace,
    trace_second: Trace,
    include_trace: bool,
    include_reference: bool,
) -> str:
    sections: list[str] = []
    task_input = task.inputs.get("input")
    if task_input is not None:
        sections.append(f"## Task\n{task_input}")
    if include_reference and reference_outputs:
        sections.append(f"## Reference\n{json.dumps(reference_outputs)}")
    sections.append(f"## Response 1\n{first}")
    if include_trace:
        sections.append(f"### Response 1 trajectory\n{_render_trajectory(trace_first)}")
    sections.append(f"## Response 2\n{second}")
    if include_trace:
        sections.append(f"### Response 2 trajectory\n{_render_trajectory(trace_second)}")
    return "\n\n".join(sections)
