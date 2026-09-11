# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import ExitStack
from decimal import Decimal

import pytest

from ag2 import Agent, Context
from ag2.events import ModelMessage, ModelResponse, ObserverAlert, Severity, Usage, UsageEvent
from ag2.extensions.cost_accounting import CostAccountingObserver, CostCatalog, UsageCostEstimator
from ag2.stream import MemoryStream
from ag2.testing import TestConfig
from ag2.usage import UsageReport


def _catalog() -> CostCatalog:
    return CostCatalog.from_litellm_mapping(
        {
            "openai/gpt-4.1-nano": {
                "litellm_provider": "openai",
                "input_cost_per_token": 0.0000001,
                "output_cost_per_token": 0.0000004,
                "cache_read_input_token_cost": 0.000000025,
                "cache_creation_input_token_cost": 0.000000125,
                "output_cost_per_reasoning_token": 0.0000005,
            }
        },
        source="test-catalog",
        version="2026-07-10",
    )


class TestCostCatalog:
    def test_loads_litellm_style_entries(self) -> None:
        catalog = _catalog()

        pricing = catalog.pricing_for("gpt-4.1-nano", provider="openai")

        assert pricing is not None
        assert pricing.input_cost_per_token == Decimal("1E-7")
        assert pricing.output_cost_per_token == Decimal("4E-7")
        assert pricing.cache_read_input_token_cost == Decimal("2.5E-8")
        assert pricing.provider == "openai"

    def test_ignores_non_pricing_entries(self) -> None:
        catalog = CostCatalog.from_litellm_mapping({"sample_spec": {"max_tokens": 1000}})

        assert catalog.pricing_for("sample_spec") is None

    def test_resolves_dated_provider_model_revision_to_base_alias(self) -> None:
        catalog = _catalog()

        pricing = catalog.pricing_for("gpt-4.1-nano-2025-04-14", provider="openai")

        assert pricing is not None
        assert pricing.output_cost_per_token == Decimal("4E-7")


class TestUsageCostEstimator:
    def test_estimates_regular_cache_and_reasoning_cost(self) -> None:
        estimator = UsageCostEstimator(_catalog())
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            cache_read_input_tokens=20,
            cache_creation_input_tokens=10,
            thinking_tokens=5,
        )

        estimate = estimator.estimate_usage(usage, model="gpt-4.1-nano", provider="openai")

        # input: 70 * .0000001 = .000007
        # cache read: 20 * .000000025 = .0000005
        # cache creation: 10 * .000000125 = .00000125
        # output: 45 * .0000004 = .000018
        # thinking: 5 * .0000005 = .0000025
        assert estimate.total_cost_usd == Decimal("0.00002925")
        assert estimate.input_cost_usd == Decimal("0.0000070")
        assert estimate.cache_read_input_cost_usd == Decimal("5.00E-7")
        assert estimate.cache_creation_input_cost_usd == Decimal("0.000001250")
        assert estimate.output_cost_usd == Decimal("0.0000180")
        assert estimate.thinking_cost_usd == Decimal("0.0000025")
        assert estimate.pricing_source == "catalog"
        assert estimate.catalog_source == "test-catalog"

    def test_unknown_model_returns_not_configured(self) -> None:
        estimate = UsageCostEstimator(_catalog()).estimate_usage(
            Usage(prompt_tokens=10, completion_tokens=5),
            model="unknown-model",
        )

        assert estimate.total_cost_usd == Decimal("0")
        assert estimate.pricing_source == "not_configured"
        assert estimate.warnings

    def test_estimates_usage_report_from_records(self) -> None:
        report = UsageReport.from_events([
            UsageEvent(Usage(prompt_tokens=10, completion_tokens=5), model="gpt-4.1-nano", provider="openai"),
            UsageEvent(Usage(prompt_tokens=20, completion_tokens=10), model="gpt-4.1-nano", provider="openai"),
        ])

        estimate = UsageCostEstimator(_catalog()).estimate_report(report)

        assert estimate.total_cost_usd == Decimal("0.000009000")


@pytest.mark.asyncio
class TestCostAccountingObserver:
    async def test_emits_warning_at_cost_threshold(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        estimator = UsageCostEstimator(_catalog())
        observer = CostAccountingObserver(
            estimator,
            warn_threshold_usd=Decimal("0.00001"),
            alert_threshold_usd=Decimal("1"),
        )
        alerts: list[ObserverAlert] = []
        stream.where(ObserverAlert).subscribe(lambda event: alerts.append(event))
        observer.register(ExitStack(), ctx)

        await ctx.send(
            UsageEvent(
                Usage(prompt_tokens=10, completion_tokens=50),
                model="gpt-4.1-nano",
                provider="openai",
            )
        )

        assert len(alerts) == 1
        assert alerts[0].severity is Severity.WARNING
        assert alerts[0].source == "cost-accounting"
        assert alerts[0].data["total_cost_usd"] == "0.000021000"
        assert observer.estimate.total_cost_usd == Decimal("0.0000210")

    async def test_reset_clears_estimate_and_allows_rewarning(self) -> None:
        stream = MemoryStream()
        ctx = Context(stream=stream)
        observer = CostAccountingObserver(
            UsageCostEstimator(_catalog()),
            warn_threshold_usd=Decimal("0.00001"),
            alert_threshold_usd=Decimal("1"),
        )
        alerts: list[ObserverAlert] = []
        stream.where(ObserverAlert).subscribe(lambda event: alerts.append(event))
        observer.register(ExitStack(), ctx)

        event = UsageEvent(Usage(prompt_tokens=10, completion_tokens=50), model="gpt-4.1-nano", provider="openai")
        await ctx.send(event)
        observer.reset()
        await ctx.send(event)

        assert len(alerts) == 2
        assert observer.estimate.total_cost_usd == Decimal("0.0000210")

    async def test_observer_accounts_for_agent_ask_usage_events(self) -> None:
        stream = MemoryStream()
        alerts: list[ObserverAlert] = []
        stream.where(ObserverAlert).subscribe(lambda event: alerts.append(event))
        observer = CostAccountingObserver(
            UsageCostEstimator(_catalog()),
            warn_threshold_usd=Decimal("0.000001"),
            alert_threshold_usd=Decimal("1"),
        )
        agent = Agent(
            "cost-test",
            config=TestConfig(
                ModelResponse(
                    ModelMessage("ok"),
                    usage=Usage(prompt_tokens=12, completion_tokens=2, cache_read_input_tokens=3),
                    model="gpt-4.1-nano",
                    provider="openai",
                )
            ),
            observers=[observer],
        )

        reply = await agent.ask("Reply with ok", stream=stream)
        report = await reply.usage()

        assert report.total.prompt_tokens == 12
        assert report.total.completion_tokens == 2
        assert report.total.cache_read_input_tokens == 3
        assert observer.estimate.total_cost_usd == Decimal("0.000001775")
        assert len(alerts) == 1
        assert alerts[0].severity is Severity.WARNING
