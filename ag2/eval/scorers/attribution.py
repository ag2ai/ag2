# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Failure attribution — for a failed run, name the step, agent, and error mode.

The Who&When-style question: when a run fails, *what* went wrong, *where*, and
(in a multi-agent trace) *who* was responsible. Two complementary detectors:

* **Deterministic** — scans the typed :class:`Trace` for unambiguous mechanical
  failures (crash, terminal tool error, no answer) and names the exact event.
  Fast, free, reproducible.
* **LLM attributor** (when a ``config`` is given) — reads the numbered
  trajectory and attributes *semantic* failures (incorrect/incomplete answer,
  ignored constraint, hallucinated fact) the rules can't see.

Exposed as a single-purpose :class:`Scorer`: it emits one :class:`Feedback`
with ``value`` = the error mode (so ``RunResult.value_counts`` gives the
failure-mode distribution for free) and the typed :class:`Attribution`
serialized into ``Feedback.detail`` (decisive step, responsible agent, …) for
programmatic access. Single-agent today; ``responsible_agent`` is the extension
point for multi-agent/network traces.
"""

import json
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from ag2.agent import Agent
from ag2.config import ModelConfig
from ag2.events import (
    BaseEvent,
    ModelResponse,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
)
from ag2.middleware.base import MiddlewareFactory

from .._types import Feedback
from ..scorer import Scorer
from ..trace import Trace

__all__ = (
    "ERROR_MODES",
    "Attribution",
    "failure_attribution",
)

# Curated, extensible starter taxonomy. Mechanical modes come from the
# deterministic detector; semantic modes from the LLM attributor.
ERROR_MODES: tuple[str, ...] = (
    "none",  # no failure
    "tool_failure",  # a tool errored (terminally)
    "hallucinated_tool",  # called a tool that does not exist (semantic)
    "loop",  # stuck repeating without progress (semantic)
    "crash",  # the run raised an exception
    "no_answer",  # finished with no final answer
    "incorrect_answer",  # answered, but wrong (semantic)
    "premature_termination",  # stopped before completing the task (semantic)
    "ignored_constraint",  # violated an instruction/constraint (semantic)
    "hallucinated_fact",  # asserted a false fact (semantic)
    "other",
)


class Attribution(BaseModel):
    """Structured failure attribution for one run (serialized into Feedback.detail)."""

    failed: bool
    error_mode: str
    decisive_step: int | None = None  # index into trace.events of the decisive event
    responsible_agent: str | None = None
    reasoning: str


class _AttributionVerdict(BaseModel):
    """LLM-facing schema — all fields required (OpenAI strict structured output)."""

    failed: bool = Field(description="Did the run fail to accomplish the task?")
    error_mode: str = Field(description="The single error mode from the provided taxonomy ('none' if it succeeded).")
    decisive_step: int = Field(description="The step number where it went wrong, or -1 if not applicable.")
    reasoning: str = Field(description="A brief justification.")


def failure_attribution(
    config: ModelConfig | None = None,
    *,
    key: str = "failure",
    agent_name: str | None = None,
    taxonomy: Iterable[str] = ERROR_MODES,
    retries: int = 1,
    middleware: Iterable[MiddlewareFactory] = (),
) -> Scorer:
    """Build a failure-attribution :class:`Scorer`.

    Args:
        config: Optional judge model for the LLM attributor. Without it the
            scorer is deterministic-only (mechanical failures + ``none``).
        key: Result key; its ``value_counts`` is the failure-mode distribution.
        agent_name: Label used for ``responsible_agent`` (single-agent today).
        taxonomy: Allowed error modes shown to the LLM attributor.
        retries: ``content()`` re-asks on schema-validation failure.
        middleware: Middleware for the attributor agent (e.g. ``TelemetryMiddleware``).
    """
    modes = tuple(taxonomy)
    attributor = (
        Agent(
            f"attributor_{key}",
            _system_prompt(modes),
            config=config,
            response_schema=_AttributionVerdict,
            middleware=middleware,
        )
        if config is not None
        else None
    )

    async def _attribute(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any] | None,
        trace: Trace,
    ) -> Feedback:
        attribution = _detect_mechanical(trace, agent_name)
        if attribution is None:
            if attributor is not None:
                attribution = await _llm_attribute(
                    attributor, inputs, outputs, reference_outputs, trace, retries, agent_name
                )
            else:
                attribution = Attribution(
                    failed=False,
                    error_mode="none",
                    responsible_agent=agent_name,
                    reasoning="no mechanical failure detected",
                )
        return Feedback(
            key=key, value=attribution.error_mode, comment=attribution.reasoning, detail=attribution.model_dump()
        )

    return Scorer(_attribute, key=key)


def _detect_mechanical(trace: Trace, agent_name: str | None) -> Attribution | None:
    """Attribute unambiguous mechanical failures; ``None`` to defer to the LLM."""
    events = trace.events

    if trace.exception is not None:
        return _mech(
            events,
            events[-1] if events else None,
            "crash",
            agent_name,
            f"run raised {type(trace.exception).__name__}: {trace.exception}",
        )

    has_answer = any(r.content for r in trace.events_of(ModelResponse))
    tool_errors = trace.events_of(ToolErrorEvent)
    if tool_errors and not has_answer:
        return _mech(
            events,
            tool_errors[0],
            "tool_failure",
            agent_name,
            f"tool {tool_errors[0].name!r} errored and the run produced no answer",
        )
    if not has_answer:
        return Attribution(
            failed=True, error_mode="no_answer", responsible_agent=agent_name, reasoning="run produced no final answer"
        )
    return None


def _mech(
    events: tuple[BaseEvent, ...], target: BaseEvent | None, mode: str, agent_name: str | None, reasoning: str
) -> Attribution:
    return Attribution(
        failed=True,
        error_mode=mode,
        decisive_step=_index_of(events, target),
        responsible_agent=agent_name,
        reasoning=reasoning,
    )


async def _llm_attribute(
    attributor: Agent,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None,
    trace: Trace,
    retries: int,
    agent_name: str | None,
) -> Attribution:
    reply = await attributor.ask(_render(inputs, outputs, reference_outputs, trace))
    verdict = await reply.content(retries=retries)
    if verdict is None:
        return Attribution(
            failed=False, error_mode="other", responsible_agent=agent_name, reasoning="attributor returned no verdict"
        )
    step = verdict.decisive_step if verdict.decisive_step is not None and verdict.decisive_step >= 0 else None
    return Attribution(
        failed=verdict.failed,
        error_mode=verdict.error_mode,
        decisive_step=step,
        responsible_agent=agent_name,
        reasoning=verdict.reasoning,
    )


def _index_of(events: tuple[BaseEvent, ...], target: BaseEvent | None) -> int | None:
    if target is None:
        return None
    for index, event in enumerate(events):
        if event is target:
            return index
    return None


def _system_prompt(taxonomy: tuple[str, ...]) -> str:
    return (
        "You analyze an agent run and attribute any failure. Given the task and the numbered "
        "trajectory, decide: whether the run failed; the single error_mode from this taxonomy — "
        f"[{', '.join(taxonomy)}] ('none' if it succeeded); the decisive_step (the step number where "
        "it went wrong, or -1 if not applicable); and a brief reasoning. Judge the trajectory as a whole."
    )


def _render(
    inputs: dict[str, Any], outputs: dict[str, Any], reference_outputs: dict[str, Any] | None, trace: Trace
) -> str:
    sections: list[str] = []
    task_input = inputs.get("input")
    if task_input is not None:
        sections.append(f"## Task\n{task_input}")
    if reference_outputs:
        sections.append(f"## Reference\n{json.dumps(reference_outputs)}")
    answer = outputs.get("body")
    sections.append(f"## Final answer\n{answer if answer is not None else '(none)'}")
    sections.append(f"## Trajectory (numbered steps)\n{_render_steps(trace)}")
    return "\n\n".join(sections)


def _render_steps(trace: Trace) -> str:
    lines = [f"[{index}] {_describe(event)}" for index, event in enumerate(trace.events)]
    return "\n".join(lines) if lines else "(no steps)"


def _describe(event: BaseEvent) -> str:
    if isinstance(event, ToolErrorEvent):
        return f"tool error: {event.name} -> {event.error}"
    if isinstance(event, ToolResultEvent):
        return f"tool result: {event.name}"
    if isinstance(event, ToolCallEvent):
        return f"tool call: {event.name}({event.arguments})"
    if isinstance(event, ModelResponse):
        return f"model: {event.content!r}"
    return type(event).__name__
