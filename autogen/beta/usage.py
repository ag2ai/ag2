# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Usage reporting — aggregate token usage over a stream's event log.

The event log is the source of truth: every :class:`~autogen.beta.events.ModelResponse`
carries the ``usage`` of one model call (with ``model``/``provider``), and every
:class:`~autogen.beta.events.TaskCompleted` carries the rolled-up ``usage`` of a
sub-agent. A sub-agent's own ``ModelResponse`` events live on its private stream and
never reach the parent history, so summing parent ``ModelResponse`` + ``TaskCompleted``
yields the correct grand total with no double counting.

``cost`` is left ``None`` until a :class:`CostModel` is supplied — the hook is in place
so dollar accounting can be layered on later without changing the report shape.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .events import BaseEvent, ModelResponse, TaskCompleted, Usage

__all__ = (
    "CostModel",
    "UsageRecord",
    "UsageReport",
)

MODEL_CALL = "model_call"
SUBTASK = "subtask"


@runtime_checkable
class CostModel(Protocol):
    """Maps token usage to a dollar cost. Not used in token-only reporting;
    supply one to :meth:`UsageReport.from_events` to populate ``cost``."""

    def price(self, usage: Usage, model: str | None, provider: str | None) -> float | None: ...


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """Usage attributed to a single stage of a run."""

    usage: Usage
    kind: str
    """``"model_call"`` for a direct LLM call, ``"subtask"`` for a sub-agent rollup."""
    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    label: str | None = None
    """Sub-agent name for ``"subtask"`` records; ``None`` for model calls."""


@dataclass(frozen=True, slots=True)
class UsageReport:
    """Aggregated token usage for a run, broken down by stage."""

    total: Usage = field(default_factory=Usage)
    records: tuple[UsageRecord, ...] = ()
    by_model: Mapping[str, Usage] = field(default_factory=dict)
    by_provider: Mapping[str, Usage] = field(default_factory=dict)
    by_kind: Mapping[str, Usage] = field(default_factory=dict)
    cost: float | None = None

    @classmethod
    def from_events(
        cls,
        events: Iterable[BaseEvent],
        *,
        cost_model: CostModel | None = None,
    ) -> "UsageReport":
        records: list[UsageRecord] = []
        for event in events:
            record = _record_for(event)
            if record is not None:
                records.append(record)

        by_model: dict[str, Usage] = {}
        by_provider: dict[str, Usage] = {}
        by_kind: dict[str, Usage] = {}
        for record in records:
            if record.model is not None:
                by_model[record.model] = by_model.get(record.model, Usage()) + record.usage
            if record.provider is not None:
                by_provider[record.provider] = by_provider.get(record.provider, Usage()) + record.usage
            by_kind[record.kind] = by_kind.get(record.kind, Usage()) + record.usage

        cost = _total_cost(records, cost_model)

        return cls(
            total=sum((record.usage for record in records), Usage()),
            records=tuple(records),
            by_model=by_model,
            by_provider=by_provider,
            by_kind=by_kind,
            cost=cost,
        )


def _record_for(event: BaseEvent) -> UsageRecord | None:
    if isinstance(event, ModelResponse):
        if not event.usage:
            return None
        return UsageRecord(
            usage=event.usage,
            kind=MODEL_CALL,
            model=event.model,
            provider=event.provider,
            finish_reason=event.finish_reason,
        )
    if isinstance(event, TaskCompleted):
        if not event.usage:
            return None
        return UsageRecord(
            usage=event.usage,
            kind=SUBTASK,
            label=event.agent_name,
        )
    return None


def _total_cost(records: Iterable[UsageRecord], cost_model: CostModel | None) -> float | None:
    if cost_model is None:
        return None
    total = 0.0
    seen = False
    for record in records:
        price = cost_model.price(record.usage, record.model, record.provider)
        if price is not None:
            total += price
            seen = True
    return total if seen else None
