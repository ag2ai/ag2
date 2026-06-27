# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Agent-as-judge scorer — grade one criterion with a beta ``Agent``.

``agent_judge`` is a *single-purpose* judge: one call grades exactly one
criterion and emits exactly one :class:`~ag2.eval.Feedback` key. A
multi-dimensional scorecard is a *list* of these::

    from ag2.eval.scorers import agent_judge

    scorers = [
        agent_judge(config, criterion="Answer is correct vs the reference.", key="correctness"),
        agent_judge(config, criterion="Every claim is grounded in the tool results.", key="faithfulness"),
    ]

Each judge becomes its own column in ``RunResult`` (numeric ``score`` →
``score_stats[key]``), so the per-dimension scores are available structurally,
not just in a rendered summary.

The judge is composed, not subclassed: the factory builds and holds an
``Agent`` whose ``response_schema`` is locked to :class:`Verdict`. Because the
scorer only reads the injected :class:`~ag2.eval.Trace` and dicts, the
same judge works under both ``run_agent()`` (live) and ``evaluate_traces()`` (trace-based).
"""

import json
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from ag2.agent import Agent
from ag2.config import ModelConfig
from ag2.events import ToolCallEvent, ToolErrorEvent, ToolResultEvent
from ag2.middleware.base import MiddlewareFactory

from .._types import Feedback
from ..scorer import Scorer
from ..trace import Trace
from .threshold import threshold as _threshold

__all__ = (
    "Verdict",
    "agent_judge",
)


class Verdict(BaseModel):
    """One judge's structured grade on a single criterion.

    Both fields are required (no optionals): OpenAI's strict structured-output
    mode rejects a schema whose ``required`` omits any property, so an optional
    field would make the judge unusable on OpenAI.
    """

    score: float = Field(
        description="Numeric grade for this one criterion, within the judge's scale (higher is better)."
    )
    reasoning: str = Field(description="A brief justification for the score.")


def agent_judge(
    config: ModelConfig,
    *,
    criterion: str,
    key: str,
    scale: tuple[float, float] = (0.0, 1.0),
    include_trace: bool = False,
    include_reference: bool = True,
    retries: int = 1,
    middleware: Iterable[MiddlewareFactory] = (),
    threshold: float | None = None,
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
        scale: ``(low, high)`` numeric range. **Enforced** — a score outside the
            range is clamped to the nearest bound (and the clamp is noted in the
            feedback comment). Default ``(0.0, 1.0)``.
        include_trace: When ``True``, the agent's tool-call trajectory (calls,
            results, errors) is rendered into the judge prompt (process grading).
            Default grades the final answer only.
        include_reference: When ``True`` (default), render the task's reference
            answer into the prompt as a ``## Reference`` section whenever
            ``reference_outputs`` is present. Set ``False`` for dimensions that must
            judge the answer on its own (e.g. faithfulness / grounding), so the
            golden answer cannot leak into the grade.
        retries: How many times ``content()`` re-asks the judge if its output
            fails :class:`Verdict` validation. Default ``1``.
        middleware: Middleware factories attached to the judge agent. Pass
            ``TelemetryMiddleware`` here to capture the judge's own LLM spans /
            token usage (judge cost), tracked separately from the agent graded.
        threshold: When set, gate the numeric score into a Pass/Fail — the judge's
            column then lands in ``result.pass_rate(key)`` (pass iff
            ``score >= threshold``) and the raw number is recorded in the feedback's
            ``detail``. A judge that returns no verdict counts as a fail. Default
            ``None`` keeps the numeric score (``score_stats``). Shorthand for wrapping
            the judge in :func:`~ag2.eval.scorers.threshold`.
    """
    low, high = scale
    judge = Agent(
        f"judge_{key}",
        _system_prompt(criterion, low, high),
        config=config,
        response_schema=Verdict,
        middleware=middleware,
    )

    async def _judge(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any] | None,
        trace: Trace,
    ) -> Feedback:
        prompt = _render_prompt(
            inputs,
            outputs,
            reference_outputs,
            trace,
            include_trace=include_trace,
            include_reference=include_reference,
        )
        reply = await judge.ask(prompt)
        verdict = await reply.content(retries=retries)
        if verdict is None:
            return Feedback(key=key, score=None, comment="judge returned no verdict")
        score = min(max(verdict.score, low), high)
        comment = verdict.reasoning
        if score != verdict.score:
            comment = f"{comment} [score clamped from {verdict.score} to scale {low}-{high}]"
        return Feedback(key=key, score=score, comment=comment)

    judge_scorer = Scorer(_judge, key=key)
    if threshold is not None:
        return _threshold(judge_scorer, at_least=threshold)
    return judge_scorer


def _system_prompt(criterion: str, low: float, high: float) -> str:
    return (
        "You are a strict evaluator grading an AI agent's response against a single criterion. "
        f"Criterion: {criterion}\n"
        f"Return a numeric score from {low} to {high} (higher is better) and a brief reasoning. "
        "Judge only this criterion — nothing else."
    )


def _render_prompt(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None,
    trace: Trace,
    *,
    include_trace: bool,
    include_reference: bool,
) -> str:
    sections: list[str] = []
    task_input = inputs.get("input")
    if task_input is not None:
        sections.append(f"## Task input\n{task_input}")
    answer = outputs.get("body")
    sections.append(f"## Agent answer\n{answer if answer is not None else '(no answer)'}")
    if include_reference and reference_outputs:
        sections.append(f"## Reference\n{json.dumps(reference_outputs)}")
    if include_trace:
        sections.append(f"## Trajectory\n{_render_trajectory(trace)}")
    return "\n\n".join(sections)


def _render_trajectory(trace: Trace) -> str:
    lines: list[str] = []
    for event in trace.events:
        if isinstance(event, ToolErrorEvent):
            lines.append(f"  -> ERROR: {event.error}")
        elif isinstance(event, ToolResultEvent):
            lines.append(f"  -> result: {_first_text(event)}")
        elif isinstance(event, ToolCallEvent):
            lines.append(f"- call {event.name}({event.arguments})")
    return "\n".join(lines) if lines else "(no tool calls)"


def _first_text(event: ToolResultEvent) -> str:
    parts = event.result.parts
    if parts and hasattr(parts[0], "content"):
        return str(parts[0].content)
    return "(result)"
