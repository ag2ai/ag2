# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Dollar cost of usage (the ``pricing`` extra).

Expected costs are computed from the catalog row the code is supposed to use,
not written down as numbers: the catalog tracks the providers' price pages, so
a hard-coded price would go stale while the pricing rules stay the same.
"""

import pytest

llm_prices = pytest.importorskip("llm_prices")

from ag2.events import Usage, UsageEvent  # noqa: E402
from ag2.usage import UsageReport, price_usage  # noqa: E402

_M = 1_000_000


def _anthropic_row() -> "llm_prices.Price":
    return next(p for p in llm_prices.Provider.ANTHROPIC.rows() if {"5m", "1h"} <= set(p.cache_write))


def _openai_extended_row() -> "llm_prices.Price":
    row = next((p for p in llm_prices.Provider.OPENAI.rows() if {"30m", "12h"} <= set(p.cache_write)), None)
    if row is None:
        pytest.skip("the catalog lists no OpenAI model with 12-hour cache-write prices")
    return row


def _long_context_row() -> "llm_prices.Price":
    row = next(
        (
            p
            for p in llm_prices.all_prices()
            if p.output is not None
            and p.long_context is not None
            and p.long_context.input is not None
            and p.long_context.output is not None
        ),
        None,
    )
    if row is None:
        pytest.skip("the catalog lists no long-context tier")
    return row


class TestPriceUsage:
    def test_cache_reads_and_writes_bill_at_their_own_anthropic_rates(self) -> None:
        row = _anthropic_row()
        # 2,000 fresh input, 40,000 read from cache, 3,000 written (1,000 of them for 1 hour)
        usage = Usage(
            prompt_tokens=45_000,
            completion_tokens=500,
            cache_read_input_tokens=40_000,
            cache_creation_input_tokens=3_000,
            cache_creation_extended_input_tokens=1_000,
        )

        cost = price_usage(usage, model=row.model, provider="anthropic")

        expected = (
            2_000 * row.input
            + 40_000 * row.cache_read
            + 2_000 * row.cache_write["5m"]
            + 1_000 * row.cache_write["1h"]
            + 500 * row.output
        ) / _M
        assert cost == pytest.approx(expected)
        # What a per-token "input x rate" estimate would bill for the same call.
        flat = (45_000 * row.input + 500 * row.output) / _M
        assert cost < flat

    def test_openai_extended_writes_bill_at_the_12_hour_rate(self) -> None:
        row = _openai_extended_row()
        usage = Usage(
            prompt_tokens=10_000,
            completion_tokens=100,
            cache_creation_input_tokens=8_000,
            cache_creation_extended_input_tokens=3_000,
        )

        cost = price_usage(usage, model=row.model, provider="openai")

        expected = (
            2_000 * row.input + 5_000 * row.cache_write["30m"] + 3_000 * row.cache_write["12h"] + 100 * row.output
        ) / _M
        assert cost == pytest.approx(expected)

    def test_writes_bill_as_input_when_the_provider_has_no_write_price(self) -> None:
        row = next(p for p in llm_prices.Provider.GOOGLE.rows() if not p.cache_write and p.output is not None)
        usage = Usage(prompt_tokens=1_000, completion_tokens=10, cache_creation_input_tokens=400)

        cost = price_usage(usage, model=row.model, provider="google")

        assert cost == pytest.approx((1_000 * row.input + 10 * row.output) / _M)

    def test_long_context_tier_applies_above_the_threshold(self) -> None:
        row = _long_context_row()
        prompt = row.long_context.threshold_tokens + 1

        cost = price_usage(Usage(prompt_tokens=prompt, completion_tokens=0), model=row.model, provider=row.provider)

        assert cost == pytest.approx(prompt * row.long_context.input / _M)

    def test_cloud_platform_falls_back_to_the_model_makers_price(self) -> None:
        row = _anthropic_row()
        usage = Usage(prompt_tokens=1_000, completion_tokens=100)

        on_bedrock = price_usage(usage, model=f"us.anthropic.{row.model}-v1:0", provider="bedrock")

        assert on_bedrock == price_usage(usage, model=row.model, provider="anthropic")

    def test_provider_scopes_the_lookup(self) -> None:
        # Sold by Groq, not by OpenAI: a provider-scoped lookup never borrows another seller's row.
        usage = Usage(prompt_tokens=1_000, completion_tokens=100)

        assert price_usage(usage, model="gpt-oss-120b", provider="openai") is None

    def test_unknown_provider_or_model_is_not_priced(self) -> None:
        usage = Usage(prompt_tokens=1_000, completion_tokens=100)

        assert price_usage(usage, model="llama3", provider="ollama") is None
        assert price_usage(usage, model="not-a-model", provider="openai") is None
        assert price_usage(usage, model=None, provider="openai") is None


class TestUsageReportCost:
    def test_prices_each_record_and_groups_the_cost(self) -> None:
        row = _anthropic_row()
        a = Usage(prompt_tokens=1_000, completion_tokens=100)
        b = Usage(prompt_tokens=5_000, completion_tokens=50, cache_read_input_tokens=4_000)
        report = UsageReport.from_events([
            UsageEvent(a, model=row.model, provider="anthropic"),
            UsageEvent(b, model=row.model, provider="anthropic"),
        ])

        cost = report.cost()

        expected = price_usage(a, model=row.model, provider="anthropic") + price_usage(
            b, model=row.model, provider="anthropic"
        )
        assert cost.total_usd == pytest.approx(expected)
        assert cost.by_model == {row.model: pytest.approx(expected)}
        assert cost.by_provider == {"anthropic": pytest.approx(expected)}
        assert cost.unpriced == ()
        assert [r.cost_usd for r in report.records] == [
            price_usage(a, model=row.model, provider="anthropic"),
            price_usage(b, model=row.model, provider="anthropic"),
        ]

    def test_long_context_is_decided_per_call_not_on_the_sum(self) -> None:
        row = _long_context_row()
        half = row.long_context.threshold_tokens // 2 + 1
        usage = Usage(prompt_tokens=half, completion_tokens=0)
        report = UsageReport.from_events([
            UsageEvent(usage, model=row.model, provider=row.provider),
            UsageEvent(usage, model=row.model, provider=row.provider),
        ])

        # Together the two prompts are over the threshold; each call on its own is not.
        assert report.cost().total_usd == pytest.approx(2 * half * row.input / _M)

    def test_unpriced_records_are_reported_not_counted_as_free(self) -> None:
        row = _anthropic_row()
        usage = Usage(prompt_tokens=1_000, completion_tokens=100)
        report = UsageReport.from_events([
            UsageEvent(usage, model=row.model, provider="anthropic"),
            UsageEvent(usage, model="llama3", provider="ollama"),
        ])

        cost = report.cost()

        assert cost.total_usd == pytest.approx(price_usage(usage, model=row.model, provider="anthropic"))
        assert [(r.provider, r.model) for r in cost.unpriced] == [("ollama", "llama3")]
        assert "ollama" not in cost.by_provider

    def test_empty_report_costs_nothing(self) -> None:
        cost = UsageReport().cost()

        assert cost.total_usd == 0.0
        assert cost.unpriced == ()
