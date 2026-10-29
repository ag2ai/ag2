# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Account once for each source UsageEvent, including descendant costs.

The event log is the source of truth. Delegation preserves the original record,
including its identity, kind, provider and model; it generates no usage rollup.
"""

from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field

from .events import BaseEvent, Usage, UsageEvent

__all__ = (
    "UsageRecord",
    "UsageReport",
    "collect_usage_events",
    "usage_event_from",
)


class _UsageCollector:
    def __init__(self, into: list[UsageEvent]) -> None:
        self.into = into

    async def __call__(self, event: BaseEvent) -> None:
        record = usage_event_from(event)
        if record is not None:
            self.into.append(record)


def collect_usage_events(into: list[UsageEvent]) -> Callable[[BaseEvent], Awaitable[None]]:
    """Collect source accounting records from a UsageEvent subscription."""
    return _UsageCollector(into)


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """Usage attributed to a single stage of a run."""

    usage: Usage
    kind: str
    """The original source stage, preserved across delegation boundaries."""
    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    label: str | None = None
    """Optional source label; archived subtask records use an agent name."""


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
        seen: set[str] = set()
        for event in events:
            usage_event = usage_event_from(event)
            if usage_event is None or usage_event.event_id in seen:
                continue
            seen.add(usage_event.event_id)
            event = usage_event
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


def usage_event_from(event: BaseEvent) -> UsageEvent | None:
    """Read a source accounting record, also accepting archived observation envelopes."""
    event = event.source_event
    return event if isinstance(event, UsageEvent) else None
