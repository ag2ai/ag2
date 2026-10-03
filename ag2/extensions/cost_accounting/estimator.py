# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Cost estimation over AG2 ``Usage`` and ``UsageReport`` values."""

from dataclasses import dataclass
from decimal import Decimal

from ag2.events import Usage
from ag2.usage import UsageRecord, UsageReport

from .catalog import CostCatalog, ModelPricing


def _tokens(value: float | int | None) -> Decimal:
    if value is None:
        return Decimal("0")
    return max(Decimal("0"), Decimal(str(value)))


def _cost(tokens: Decimal, rate: Decimal | None) -> Decimal:
    if rate is None:
        return Decimal("0")
    return tokens * rate


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Estimated USD cost for one usage item or aggregated report."""

    total_cost_usd: Decimal
    input_cost_usd: Decimal = Decimal("0")
    output_cost_usd: Decimal = Decimal("0")
    cache_read_input_cost_usd: Decimal = Decimal("0")
    cache_creation_input_cost_usd: Decimal = Decimal("0")
    thinking_cost_usd: Decimal = Decimal("0")
    model: str | None = None
    provider: str | None = None
    currency: str = "USD"
    pricing_source: str = "catalog"
    catalog_source: str | None = None
    catalog_version: str | None = None
    warnings: tuple[str, ...] = ()

    def __add__(self, other: "CostEstimate") -> "CostEstimate":
        if not isinstance(other, CostEstimate):
            return NotImplemented
        warnings = (*self.warnings, *other.warnings)
        if self.pricing_source == "catalog" and self.total_cost_usd == 0 and not self.warnings:
            pricing_source = other.pricing_source
        elif other.pricing_source == "catalog" and other.total_cost_usd == 0 and not other.warnings:
            pricing_source = self.pricing_source
        elif self.pricing_source != other.pricing_source:
            pricing_source = "mixed"
        else:
            pricing_source = self.pricing_source
        return CostEstimate(
            total_cost_usd=self.total_cost_usd + other.total_cost_usd,
            input_cost_usd=self.input_cost_usd + other.input_cost_usd,
            output_cost_usd=self.output_cost_usd + other.output_cost_usd,
            cache_read_input_cost_usd=self.cache_read_input_cost_usd + other.cache_read_input_cost_usd,
            cache_creation_input_cost_usd=self.cache_creation_input_cost_usd + other.cache_creation_input_cost_usd,
            thinking_cost_usd=self.thinking_cost_usd + other.thinking_cost_usd,
            currency=self.currency,
            pricing_source=pricing_source,
            catalog_source=self.catalog_source or other.catalog_source,
            catalog_version=self.catalog_version or other.catalog_version,
            warnings=warnings,
        )


class UsageCostEstimator:
    """Estimate cost from AG2 usage data and a pricing catalog."""

    def __init__(self, catalog: CostCatalog) -> None:
        self.catalog = catalog

    def estimate_usage(
        self,
        usage: Usage,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> CostEstimate:
        pricing = self.catalog.pricing_for(model, provider=provider)
        if pricing is None:
            return CostEstimate(
                total_cost_usd=Decimal("0"),
                model=model,
                provider=provider,
                pricing_source="not_configured",
                catalog_source=self.catalog.source,
                catalog_version=self.catalog.version,
                warnings=(f"No pricing configured for model {model!r}.",),
            )

        return _estimate_with_pricing(
            usage=usage,
            pricing=pricing,
            model=model,
            provider=provider or pricing.provider,
            catalog=self.catalog,
        )

    def estimate_record(self, record: UsageRecord) -> CostEstimate:
        return self.estimate_usage(record.usage, model=record.model, provider=record.provider)

    def estimate_report(self, report: UsageReport) -> CostEstimate:
        estimate = CostEstimate(total_cost_usd=Decimal("0"), catalog_source=self.catalog.source)
        for record in report.records:
            estimate += self.estimate_record(record)
        return estimate


def _estimate_with_pricing(
    *,
    usage: Usage,
    pricing: ModelPricing,
    model: str | None,
    provider: str | None,
    catalog: CostCatalog,
) -> CostEstimate:
    prompt_tokens = _tokens(usage.prompt_tokens)
    cache_read_tokens = _tokens(usage.cache_read_input_tokens)
    cache_creation_tokens = _tokens(usage.cache_creation_input_tokens)
    thinking_tokens = _tokens(usage.thinking_tokens)

    cached_input_tokens = min(prompt_tokens, cache_read_tokens + cache_creation_tokens)
    regular_input_tokens = prompt_tokens - cached_input_tokens

    completion_tokens = _tokens(usage.completion_tokens)
    if pricing.output_cost_per_reasoning_token is not None:
        regular_output_tokens = max(Decimal("0"), completion_tokens - thinking_tokens)
    else:
        regular_output_tokens = completion_tokens

    input_cost = _cost(regular_input_tokens, pricing.input_cost_per_token)
    cache_read_cost = _cost(cache_read_tokens, pricing.cache_read_input_token_cost or pricing.input_cost_per_token)
    cache_creation_cost = _cost(
        cache_creation_tokens,
        pricing.cache_creation_input_token_cost or pricing.input_cost_per_token,
    )
    output_cost = _cost(regular_output_tokens, pricing.output_cost_per_token)
    thinking_cost = _cost(thinking_tokens, pricing.output_cost_per_reasoning_token)
    total = input_cost + cache_read_cost + cache_creation_cost + output_cost + thinking_cost

    return CostEstimate(
        total_cost_usd=total,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        cache_read_input_cost_usd=cache_read_cost,
        cache_creation_input_cost_usd=cache_creation_cost,
        thinking_cost_usd=thinking_cost,
        model=model,
        provider=provider,
        pricing_source="catalog",
        catalog_source=catalog.source,
        catalog_version=catalog.version,
    )
