# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Token usage as AG-UI reports it: per (provider, model), in the protocol's accounting."""

from collections.abc import Iterable
from math import isfinite

from ag_ui.core import TokenUsage

from ag2.events import BaseEvent, Usage, UsageEvent
from ag2.usage import UsageRecord, UsageReport


def map_usage_events_to_ag_ui(usage_events: Iterable[BaseEvent]) -> list[TokenUsage] | None:
    """Attributed spend for a set of events, as AG-UI's per-(provider, model) list."""
    # Both AG-UI transports call this, so the two cannot compose attribution and
    # grouping differently. They differ only in where the events come from.
    return map_usage_records_to_ag_ui(UsageReport.from_events(_model_calls(usage_events)).records)


def _model_calls(usage_events: Iterable[BaseEvent]) -> list[BaseEvent]:
    # A delegation's rollup is replaced by the calls it sums. Each provider
    # counts its own way, so the correction below has to see each call under
    # its own provider: a rollup spanning two models carries neither label.
    calls: list[BaseEvent] = []
    for event in usage_events:
        if isinstance(event, UsageEvent) and event.parts:
            calls.extend(event.parts)
        else:
            calls.append(event)
    return calls


def map_usage_records_to_ag_ui(records: Iterable[UsageRecord]) -> list[TokenUsage] | None:
    """Attributed spend, as AG-UI's per-(provider, model) list, in the protocol's accounting.

    Counts a provider did not report are omitted, never zero-filled.
    """
    # Records, not the report's by_model / by_provider: those are independent
    # maps, so the (provider, model) pair cannot be recovered from them, and each
    # drops what the other side did not label — where a sub-agent's spend lives.
    grouped: dict[tuple[str | None, str | None], list[Usage]] = {}
    for record in records:
        grouped.setdefault((record.provider, record.model), []).append(record.usage)

    # Pairs are never folded together: absent counts add as zero, so merging a
    # provider that reports reasoning tokens with one that does not would read as
    # a complete measurement. Within a pair the calls are summed, because there an
    # absent additive count does mean the provider had nothing to report.
    entries = []
    for (provider, model), usages in grouped.items():
        summed = sum(usages, Usage())
        input_tokens = _token_count(_input_total(provider, summed))
        output_tokens = _token_count(_output_total(provider, summed))
        entries.append(
            TokenUsage(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                # Computed, never copied: a provider's own total need not count
                # the way the two totals beside it now do.
                total_tokens=None if input_tokens is None or output_tokens is None else input_tokens + output_tokens,
                reasoning_tokens=_token_count(summed.thinking_tokens),
                cached_input_tokens=_token_count(summed.cache_read_input_tokens),
                cache_write_input_tokens=_token_count(summed.cache_creation_input_tokens),
            )
        )
    return entries or None


# The correction to AG-UI 1.0's accounting is made here, where usage leaves for
# the wire, and nowhere else. `Usage` keeps each provider's numbers as the
# provider reported them, because budgets and limiters read them that way;
# "fixing" the provider normalizers instead would shift every one of those.
# AG-UI's input and output are totals, and its cache and reasoning counts parts
# of them, so where a provider reports those beside a smaller count, they are
# added in here.

# Providers whose prompt count leaves out the tokens read from and written to
# the cache. Bedrock is not among them until a live call confirms that Converse
# counts the way Anthropic does.
_CACHE_OUTSIDE_PROMPT = frozenset({"anthropic"})

# Providers whose completion count leaves out the reasoning tokens:
# Gemini's `thoughts_token_count` sits beside `candidates_token_count`. xAI's
# reasoning count is not known to do either, so it is left as reported.
_REASONING_OUTSIDE_COMPLETION = frozenset({"google"})


def _input_total(provider: str | None, usage: Usage) -> float | None:
    if usage.prompt_tokens is None or provider not in _CACHE_OUTSIDE_PROMPT:
        return usage.prompt_tokens
    return usage.prompt_tokens + (usage.cache_read_input_tokens or 0) + (usage.cache_creation_input_tokens or 0)


def _output_total(provider: str | None, usage: Usage) -> float | None:
    if usage.completion_tokens is None or provider not in _REASONING_OUTSIDE_COMPLETION:
        return usage.completion_tokens
    return usage.completion_tokens + (usage.thinking_tokens or 0)


def _token_count(value: float | None) -> int | None:
    # The wire type admits only non-negative integers, and this runs on the
    # failure path while the run is being reported — so a value the wire would
    # reject is omitted rather than left to raise over the real cause.
    if value is None or not isfinite(value) or value < 0:
        return None
    return int(value)
