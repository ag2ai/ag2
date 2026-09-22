# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``ag2.eval`` scorer that attributes failed runs with a fitted probe."""

import asyncio
import json
from typing import Any

from ag2.eval import Feedback, Scorer, Trace

from .attributor import ProbeAttributor
from .trace import conversation_from_trace

__all__ = ("probe_failure_attribution",)


def probe_failure_attribution(
    attributor: ProbeAttributor,
    *,
    key: str = "failure_probe",
    agent_name: str = "assistant",
    ground_truth_field: str | None = None,
) -> Scorer:
    """Build a :class:`~ag2.eval.Scorer` that names each run's decisive step with a probe.

    The white-box counterpart of :func:`ag2.eval.scorers.failure_attribution`:
    the trace is converted with :func:`conversation_from_trace`, the fitted
    ``attributor`` scores every turn, and the result is one :class:`Feedback`
    whose ``value`` is the responsible agent (so ``RunResult.value_counts(key)``
    gives the blame distribution) and whose ``detail`` is the serialized
    :class:`~ag2.eval.scorers.Attribution`, with ``decisive_step`` an index
    into ``trace.events`` as in the core scorer.

    The probe always names a step, so it assumes the run failed: score only
    failed runs, or read it alongside a correctness scorer. The model forward
    pass runs in a worker thread so the event loop is not blocked.

    Args:
        attributor: A fitted (or loaded) attributor.
        key: Feedback key.
        agent_name: Speaker assigned to the traced agent's own turns.
        ground_truth_field: Field of ``reference_outputs`` holding the expected
            answer. When omitted, a single-field ``reference_outputs`` uses its
            only value and anything else is shown as JSON.
    """
    return Scorer(_ProbeScorer(attributor, key, agent_name, ground_truth_field), key=key)


class _ProbeScorer:
    def __init__(self, attributor: ProbeAttributor, key: str, agent_name: str, ground_truth_field: str | None) -> None:
        self._attributor = attributor
        self._key = key
        self._agent_name = agent_name
        self._ground_truth_field = ground_truth_field

    async def __call__(
        self, inputs: dict[str, Any], reference_outputs: dict[str, Any] | None, trace: Trace
    ) -> Feedback:
        question = inputs.get("input")
        converted = conversation_from_trace(
            trace,
            question="" if question is None else str(question),
            ground_truth=_ground_truth(reference_outputs, self._ground_truth_field),
            agent_name=self._agent_name,
        )
        if converted is None:
            return Feedback(key=self._key, comment="trace has no turns to attribute")

        attribution = await asyncio.to_thread(self._attributor.attribute, converted.conversation)
        if attribution.decisive_step is not None:
            attribution = attribution.model_copy(
                update={"decisive_step": converted.event_indices[attribution.decisive_step]}
            )
        return Feedback(
            key=self._key,
            value=attribution.responsible_agent,
            comment=attribution.reasoning,
            detail=attribution.model_dump(),
        )


def _ground_truth(reference_outputs: dict[str, Any] | None, field: str | None) -> str:
    if not reference_outputs:
        return ""
    if field is not None:
        value = reference_outputs.get(field)
        return "" if value is None else str(value)
    if len(reference_outputs) == 1:
        return str(next(iter(reference_outputs.values())))
    return json.dumps(reference_outputs, default=str)
