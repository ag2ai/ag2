# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Pricing catalog primitives for estimated usage cost accounting."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _field_decimal(raw: Mapping[str, Any], *names: str) -> Decimal | None:
    for name in names:
        value = _decimal(raw.get(name))
        if value is not None:
            return value
    return None


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Per-token model pricing in USD.

    Field names mirror the token classes exposed by ``ag2.events.Usage`` and
    the most common LiteLLM catalog keys.
    """

    input_cost_per_token: Decimal | None = None
    output_cost_per_token: Decimal | None = None
    cache_read_input_token_cost: Decimal | None = None
    cache_creation_input_token_cost: Decimal | None = None
    output_cost_per_reasoning_token: Decimal | None = None
    provider: str | None = None

    @classmethod
    def from_litellm_entry(cls, raw: Mapping[str, Any]) -> "ModelPricing | None":
        """Create pricing from a LiteLLM ``model_prices...`` entry."""

        pricing = cls(
            input_cost_per_token=_field_decimal(raw, "input_cost_per_token"),
            output_cost_per_token=_field_decimal(raw, "output_cost_per_token"),
            cache_read_input_token_cost=_field_decimal(raw, "cache_read_input_token_cost"),
            cache_creation_input_token_cost=_field_decimal(raw, "cache_creation_input_token_cost"),
            output_cost_per_reasoning_token=_field_decimal(raw, "output_cost_per_reasoning_token"),
            provider=str(raw.get("litellm_provider") or raw.get("provider") or "") or None,
        )
        if not any((
            pricing.input_cost_per_token,
            pricing.output_cost_per_token,
            pricing.cache_read_input_token_cost,
            pricing.cache_creation_input_token_cost,
            pricing.output_cost_per_reasoning_token,
        )):
            return None
        return pricing


@dataclass(frozen=True, slots=True)
class CostCatalog:
    """Model pricing lookup table.

    The catalog is deliberately immutable and local. Applications may refresh
    or replace it from any maintained source, including LiteLLM's public JSON
    file, without coupling AG2 core to mutable provider pricing.
    """

    prices: Mapping[str, ModelPricing]
    source: str | None = None
    version: str | None = None

    @classmethod
    def from_litellm_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        source: str | None = None,
        version: str | None = None,
    ) -> "CostCatalog":
        prices: dict[str, ModelPricing] = {}
        for model, raw in data.items():
            if not isinstance(raw, Mapping):
                continue
            pricing = ModelPricing.from_litellm_entry(raw)
            if pricing is not None:
                prices[_model_key(str(model))] = pricing
        return cls(prices=prices, source=source, version=version)

    @classmethod
    def from_file(cls, path: str | Path, *, source: str | None = None, version: str | None = None) -> "CostCatalog":
        catalog_path = Path(path)
        text = catalog_path.read_text(encoding="utf-8")
        if catalog_path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(text)
        else:
            payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError("cost catalog must be a JSON/YAML object")
        model_payload = payload.get("models", payload)
        if not isinstance(model_payload, Mapping):
            raise ValueError("cost catalog models must be an object")
        return cls.from_litellm_mapping(model_payload, source=source or str(catalog_path), version=version)

    def pricing_for(self, model: str | None, *, provider: str | None = None) -> ModelPricing | None:
        if not model:
            return None
        keys = _model_lookup_keys(model)
        if provider:
            keys = [key for model_key in keys for key in (_model_key(f"{provider}/{model_key}"), model_key)]
        for key in keys:
            pricing = self.prices.get(key)
            if pricing is not None:
                return pricing
        return None


def _model_key(model: str) -> str:
    return model.strip().lower()


def _model_lookup_keys(model: str) -> list[str]:
    key = _model_key(model)
    keys = [key]
    # Providers may return a dated model revision for an alias configured by the
    # caller, e.g. ``gpt-4.1-nano`` -> ``gpt-4.1-nano-2025-04-14``.
    base_key = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", key)
    if base_key != key:
        keys.append(base_key)
    return keys
