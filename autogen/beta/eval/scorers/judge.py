# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Agent-as-judge scorer — grade one criterion with a beta ``Agent``.

``agent_judge`` is a *single-purpose* judge: one call grades exactly one
criterion and emits exactly one :class:`~autogen.beta.eval.Feedback` key. A
multi-dimensional scorecard is a *list* of these::

    from autogen.beta.eval.scorers import agent_judge

    scorers = [
        agent_judge(config, criterion="Answer is correct vs the reference.", key="correctness"),
        agent_judge(config, criterion="Every claim is grounded in the tool results.", key="faithfulness"),
    ]

Each judge becomes its own column in ``RunResult`` (numeric ``score`` →
``score_stats[key]``), so the per-dimension scores are available structurally,
not just in a rendered summary.

The judge is composed, not subclassed: the factory builds and holds an
``Agent`` whose ``response_schema`` is locked to :class:`Verdict`. Because the
scorer only reads the injected :class:`~autogen.beta.eval.Trace` and dicts, the
same judge works under both ``run()`` (live) and ``evaluate()`` (trace-based).
"""

import json
from typing import Any

from pydantic import BaseModel, Field

from autogen.beta.agent import Agent
from autogen.beta.config import ModelConfig
from autogen.beta.events import ToolCallEvent

from .._types import Feedback
from ..scorer import Scorer
from ..trace import Trace

__all__ = (
    "Verdict",
    "agent_judge",
)


class Verdict(BaseModel):
    """One judge's structured grade on a single criterion."""

    score: float = Field(
        description="Numeric grade for this one criterion, within the judge's scale (higher is better)."
    )
    reasoning: str = Field(description="A brief justification for the score.")
    label: str | None = Field(default=None, description="Optional short categorical label for this criterion.")


def agent_judge(
    config: ModelConfig,
    *,
    criterion: str,
    key: str,
    scale: tuple[float, float] = (0.0, 1.0),
    include_trace: bool = False,
    retries: int = 1,
) -> Scorer:
    """Build a single-purpose Agent-as-judge :class:`Scorer`.

    Args:
        config: Model config for the judge agent (e.g. an ``AnthropicConfig``;
            pin temperature to 0 for stable grading).
        criterion: The single standard this judge grades against, in plain
            English. One judge grades one criterion — compose several judges
            for a multi-dimensional scorecard.
        key: The ``Feedback`` key this judge emits; becomes its column in
            ``RunResult`` aggregates. Use a distinct key per criterion.
        scale: ``(low, high)`` numeric range the judge is told to score within.
            Default ``(0.0, 1.0)``.
        include_trace: When ``True``, the agent's tool-call trajectory is
            rendered into the judge prompt (process grading). Default grades the
            final answer only.
        retries: How many times ``content()`` re-asks the judge if its output
            fails :class:`Verdict` validation. Default ``1``.
    """
    low, high = scale
    judge = Agent(
        f"judge_{key}",
        _system_prompt(criterion, low, high),
        config=config,
        response_schema=Verdict,
    )

    async def _judge(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any] | None,
        trace: Trace,
    ) -> Feedback:
        prompt = _render_prompt(inputs, outputs, reference_outputs, trace, include_trace=include_trace)
        reply = await judge.ask(prompt)
        verdict = await reply.content(retries=retries)
        if verdict is None:
            return Feedback(key=key, score=None, comment="judge returned no verdict")
        return Feedback(key=key, score=verdict.score, value=verdict.label, comment=verdict.reasoning)

    return Scorer(_judge, key=key)


def _system_prompt(criterion: str, low: float, high: float) -> str:
    return (
        "You are a strict evaluator grading an AI agent's response against a single criterion. "
        f"Criterion: {criterion}\n"
        f"Return a numeric score from {low} to {high} (higher is better), a brief reasoning, "
        "and an optional short categorical label. Judge only this criterion — nothing else."
    )


def _render_prompt(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None,
    trace: Trace,
    *,
    include_trace: bool,
) -> str:
    sections: list[str] = []
    task_input = inputs.get("input")
    if task_input is not None:
        sections.append(f"## Task input\n{task_input}")
    answer = outputs.get("body")
    sections.append(f"## Agent answer\n{answer if answer is not None else '(no answer)'}")
    if reference_outputs:
        sections.append(f"## Reference\n{json.dumps(reference_outputs)}")
    if include_trace:
        sections.append(f"## Trajectory\n{_render_trajectory(trace)}")
    return "\n\n".join(sections)


def _render_trajectory(trace: Trace) -> str:
    lines = [f"- {call.name}({call.arguments})" for call in trace.events_of(ToolCallEvent)]
    return "\n".join(lines) if lines else "(no tool calls)"
