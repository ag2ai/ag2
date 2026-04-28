# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for end-to-end Agent smoke tests that hit real provider APIs.

Every test parametrizes over ``openai`` / ``anthropic`` / ``gemini`` via
``provider_config`` or ``streaming_config``, so each instance carries
the corresponding per-provider mark and is excluded from
``test-beta-cov`` by ``_beta_llm_filter``.

Credentials are read from the environment — `just` loads ``.env``
automatically; for direct ``pytest`` invocation, export the keys
yourself or run via ``set -a; source .env; pytest …``.
"""

import os

import pytest

from autogen.beta.config import AnthropicConfig, GeminiConfig, OpenAIConfig


def _require(env: str) -> str:
    value = os.getenv(env)
    if not value:
        pytest.skip(f"{env} not set; skipping real-API smoke test")
    return value


def _require_gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY/GOOGLE_API_KEY not set; skipping real-API smoke test")
    return key


@pytest.fixture()
def openai_config() -> OpenAIConfig:
    return OpenAIConfig(
        model="gpt-5.4-nano",
        api_key=_require("OPENAI_API_KEY"),
        temperature=0,
    )


@pytest.fixture()
def anthropic_config() -> AnthropicConfig:
    return AnthropicConfig(
        model="claude-haiku-4-5",
        api_key=_require("ANTHROPIC_API_KEY"),
        temperature=0,
    )


@pytest.fixture()
def gemini_config() -> GeminiConfig:
    return GeminiConfig(
        model="gemini-3.1-flash-lite-preview",
        api_key=_require_gemini_key(),
        temperature=0,
    )


@pytest.fixture(
    params=[
        pytest.param("openai", marks=pytest.mark.openai),
        pytest.param("anthropic", marks=pytest.mark.anthropic),
        pytest.param("gemini", marks=pytest.mark.gemini),
    ]
)
def streaming_config(request):
    """Parametrized streaming-enabled config for each provider.

    Tests using this run once per provider with ``streaming=True``.
    """
    if request.param == "openai":
        return OpenAIConfig(
            model="gpt-5.4-nano",
            api_key=_require("OPENAI_API_KEY"),
            temperature=0,
            streaming=True,
        )
    if request.param == "anthropic":
        return AnthropicConfig(
            model="claude-haiku-4-5",
            api_key=_require("ANTHROPIC_API_KEY"),
            temperature=0,
            streaming=True,
        )
    return GeminiConfig(
        model="gemini-3.1-flash-lite-preview",
        api_key=_require_gemini_key(),
        temperature=0,
        streaming=True,
    )


@pytest.fixture(
    params=[
        pytest.param("openai", marks=pytest.mark.openai),
        pytest.param("anthropic", marks=pytest.mark.anthropic),
        pytest.param("gemini", marks=pytest.mark.gemini),
    ]
)
def provider_config(request):
    """Parametrized fixture that yields a config for each provider.

    Tests using this fixture run once per provider. The per-provider
    fixtures are resolved lazily via ``request.getfixturevalue`` so that
    a CI job installing only one provider's SDK does not trip the missing-
    optional-dependency stubs of the other providers' ``ModelConfig``s.
    """
    return request.getfixturevalue(f"{request.param}_config")
