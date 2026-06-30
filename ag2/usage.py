# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Usage reporting — aggregate token usage over a stream's event log.

The event log is the source of truth. Every place that spends tokens emits a
:class:`~ag2.events.UsageEvent` onto the stream as the tokens are
spent — a direct LLM call (``kind="model_call"``), a live session, history
compaction (``kind="compaction"``), memory aggregation (``kind="aggregation"``),
or a sub-agent rollup (``kind="subtask"``). A sub-agent's per-call
``UsageEvent`` events live on its private stream and never reach the parent
history; only its rolled-up ``"subtask"`` event is emitted on the parent, so
summing the parent's ``UsageEvent`` events yields the correct grand total with
no double counting.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .events import BaseEvent, Usage, UsageEvent

__all__ = (
    "UsageRecord",
    "UsageReport",
)


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """Usage attributed to a single stage of a run."""

    usage: Usage
    kind: str
    """``"model_call"`` for a direct LLM call, ``"subtask"`` for a sub-agent
    rollup, ``"compaction"`` / ``"aggregation"`` for internal maintenance calls."""
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

    @classmethod
    def from_events(cls, events: Iterable[BaseEvent]) -> "UsageReport":
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

        return cls(
            total=sum((record.usage for record in records), Usage()),
            records=tuple(records),
            by_model=by_model,
            by_provider=by_provider,
            by_kind=by_kind,
        )


def _record_for(event: BaseEvent) -> UsageRecord | None:
    if isinstance(event, UsageEvent):
        if not event.usage:
            return None
        return UsageRecord(
            usage=event.usage,
            kind=event.kind,
            model=event.model,
            provider=event.provider,
            finish_reason=event.finish_reason,
            label=event.label,
        )
    return None
