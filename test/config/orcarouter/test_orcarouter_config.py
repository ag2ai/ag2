# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.config import ModelProvider, OrcaRouterConfig
from ag2.config.orcarouter import ORCAROUTER_DEFAULT_BASE_URL


def test_provider_is_orcarouter() -> None:
    config = OrcaRouterConfig(model="anthropic/claude-opus-4.8")

    assert config.provider is ModelProvider.ORCAROUTER


def test_base_url_defaults_to_orcarouter_gateway() -> None:
    config = OrcaRouterConfig(model="anthropic/claude-opus-4.8")

    assert config.base_url == ORCAROUTER_DEFAULT_BASE_URL


def test_base_url_can_be_overridden() -> None:
    config = OrcaRouterConfig(model="gpt-5", base_url="https://proxy.internal/v1")

    assert config.base_url == "https://proxy.internal/v1"


def test_copy_without_overrides_returns_new_equal_instance() -> None:
    config = OrcaRouterConfig(model="gpt-5", temperature=0.2, streaming=True)

    copied = config.copy()

    assert copied == config
    assert copied is not config


def test_copy_applies_overrides_without_mutating_original() -> None:
    config = OrcaRouterConfig(model="gpt-5", api_key="key", temperature=0.2, streaming=False)

    copied = config.copy(model="gpt-5-mini", temperature=0.8, streaming=True, api_key=None)

    assert copied.model == "gpt-5-mini"
    assert copied.temperature == 0.8
    assert copied.streaming is True
    assert copied.api_key is None

    assert config.model == "gpt-5"
    assert config.temperature == 0.2
    assert config.streaming is False
    assert config.api_key == "key"


def test_routing_fields_default_to_none() -> None:
    config = OrcaRouterConfig(model="gpt-5")

    assert config.route is None
    assert config.models is None
    assert config.extra_body is None


def test_route_forwarded_via_extra_body() -> None:
    config = OrcaRouterConfig(model="gpt-5", api_key="EMPTY", route="lowest-latency")

    client = config.create()

    assert client._create_options.get("extra_body") == {"route": "lowest-latency"}


def test_models_fallback_forwarded_via_extra_body() -> None:
    fallbacks = ["anthropic/claude-opus-4.8", "openai/gpt-5"]
    config = OrcaRouterConfig(model="anthropic/claude-opus-4.8", api_key="EMPTY", models=fallbacks)

    client = config.create()

    assert client._create_options.get("extra_body") == {"models": fallbacks}


def test_route_and_models_merge_with_explicit_extra_body() -> None:
    config = OrcaRouterConfig(
        model="gpt-5",
        api_key="EMPTY",
        route="fallback",
        models=["a", "b"],
        extra_body={"custom_flag": True},
    )

    client = config.create()

    assert client._create_options.get("extra_body") == {
        "custom_flag": True,
        "route": "fallback",
        "models": ["a", "b"],
    }


def test_explicit_extra_body_takes_precedence_over_route() -> None:
    config = OrcaRouterConfig(
        model="gpt-5",
        api_key="EMPTY",
        route="fallback",
        extra_body={"route": "user-chosen"},
    )

    client = config.create()

    assert client._create_options.get("extra_body") == {"route": "user-chosen"}


def test_no_routing_options_yields_no_extra_body() -> None:
    config = OrcaRouterConfig(model="gpt-5", api_key="EMPTY")

    client = config.create()

    assert client._create_options.get("extra_body") is None
