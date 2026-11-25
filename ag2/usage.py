# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Account once for each source UsageEvent, including descendant costs.

The event log is the source of truth. Delegation preserves the original record,
including its identity, kind, provider and model; it generates no usage rollup.

With the ``pricing`` extra installed (``pip install "ag2[pricing]"``), each record
is priced in USD from the ``llm-prices`` catalog: :attr:`UsageRecord.cost_usd`
and :meth:`UsageReport.cost`.
"""

from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace

from ._import_utils import optional_import_block, require_optional_import
from .events import BaseEvent, Usage, UsageEvent

with optional_import_block():
    import llm_prices

__all__ = (
    "UsageCost",
    "UsageRecord",
    "UsageReport",
    "collect_usage_events",
    "price_usage",
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

    @property
    def cost_usd(self) -> float | None:
        """List-price cost of this record in USD, or ``None`` when it cannot be priced.

        See :func:`price_usage`. Requires the ``pricing`` extra.
        """
        return price_usage(self.usage, model=self.model, provider=self.provider)


@dataclass(frozen=True, slots=True)
class UsageCost:
    """Dollar cost of a :class:`UsageReport`, priced one record at a time.

    Each record is priced on its own because prices depend on the single
    request: a long-context tier applies when *that* request's prompt is over
    the model's threshold, which a sum over several requests cannot tell.
    """

    total_usd: float = 0.0
    """Sum of every priced record. Leaves out :attr:`unpriced`."""
    by_model: Mapping[str, float] = field(default_factory=dict)
    by_provider: Mapping[str, float] = field(default_factory=dict)
    unpriced: tuple[UsageRecord, ...] = ()
    """Records the catalog has no price for: a provider or model it does not list
    (a local or self-hosted model, for instance), a cache-write lifetime the model
    has no price for, or an archived rollup that names no model."""


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

    @require_optional_import("llm_prices", "pricing")
    def cost(self) -> UsageCost:
        """Price every record in USD from the ``llm-prices`` catalog.

        Prompt-cache reads, cache writes by retention (including the extended
        1-hour and 12-hour writes) and long-context tiers are billed at their
        own prices; see :func:`price_usage`. Records that cannot be priced are
        returned in :attr:`UsageCost.unpriced` rather than counted as free.
        Requires the ``pricing`` extra.
        """
        total = 0.0
        by_model: dict[str, float] = {}
        by_provider: dict[str, float] = {}
        unpriced: list[UsageRecord] = []
        for record in self.records:
            usd = record.cost_usd
            if usd is None:
                unpriced.append(record)
                continue
            total += usd
            if record.model is not None:
                by_model[record.model] = by_model.get(record.model, 0.0) + usd
            if record.provider is not None:
                by_provider[record.provider] = by_provider.get(record.provider, 0.0) + usd
        return UsageCost(total_usd=total, by_model=by_model, by_provider=by_provider, unpriced=tuple(unpriced))


@require_optional_import("llm_prices", "pricing")
def price_usage(usage: Usage, *, model: str | None, provider: str | None = None) -> float | None:
    """List-price cost in USD of one :class:`UsageEvent`'s usage.

    Prices come from the ``llm-prices`` catalog (dated rows read from each
    provider's pricing page). With ``provider`` set, only what that provider
    sells is searched, so a model hosted by several providers gets the right
    one; cloud platforms (Bedrock, Vertex AI, Azure) fall back to the model
    maker's list price. Model ids are matched with their dated, ``-latest``
    and provider-prefixed forms.

    The usage is split the way providers bill it (ADR 0036 fields): uncached
    input, cache reads, default-retention cache writes and extended-retention
    writes (``cache_creation_extended_input_tokens``: Anthropic's 1-hour,
    OpenAI's 12-hour), and output with reasoning tokens. A model whose provider
    does not charge separately for cache writes bills them as input.

    Returns ``None`` when the model, the provider or a price the usage needs is
    not in the catalog. Requires the ``pricing`` extra.
    """
    if model is None or not usage:
        return None
    price = llm_prices.lookup(model) if provider is None else llm_prices.lookup_provider(provider, model)
    if price is None:
        return None
    rated, tokens = _billable(price, usage)
    breakdown = rated.cost_usage(tokens)
    return None if breakdown is None else breakdown.total


_RENAMED_TTLS: Mapping[str, str] = {"30m": "5m", "12h": "1h"}
"""OpenAI's default and extended cache-write lifetimes, renamed to the two
``llm_prices.Usage`` carries default and extended writes in."""


def _billable(price: "llm_prices.Price", usage: Usage) -> "tuple[llm_prices.Price, llm_prices.Usage]":
    """Split ``usage`` into the token kinds ``price`` bills separately.

    ``llm_prices.Usage`` carries default-retention writes in ``cache_write_5m``
    and extended ones in ``cache_write_1h``, Anthropic's lifetimes. A row that
    names its lifetimes differently (OpenAI's ``30m`` and ``12h``) has its write
    prices renamed to match, so ``cost_usage`` still applies the long-context
    switch to the whole prompt, cache writes included.
    """
    read = _count(usage.cache_read_input_tokens)
    write = _count(usage.cache_creation_input_tokens)
    extended = min(_count(usage.cache_creation_extended_input_tokens), write)
    uncached = max(_count(usage.prompt_tokens) - read - write, 0)
    output = _count(usage.completion_tokens)
    if not price.cache_write:
        # The provider does not charge separately for cache writes.
        return price, llm_prices.Usage(input=uncached + write, cache_read=read, output=output)
    if any(ttl in _RENAMED_TTLS for ttl in price.cache_write):
        long_context = price.long_context
        if long_context is not None:
            long_context = replace(long_context, cache_write=_renamed(long_context.cache_write))
        price = replace(price, cache_write=_renamed(price.cache_write), long_context=long_context)
    tokens = llm_prices.Usage(
        input=uncached,
        cache_read=read,
        cache_write_5m=write - extended,
        cache_write_1h=extended,
        output=output,
    )
    return price, tokens


def _renamed(rates: Mapping[str, float]) -> dict[str, float]:
    return {_RENAMED_TTLS.get(ttl, ttl): rate for ttl, rate in rates.items()}


def _count(tokens: float | None) -> int:
    return max(int(tokens or 0), 0)


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
