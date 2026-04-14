# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for end-to-end smoke tests that hit real provider APIs.

Loads credentials from the repo-root ``.env`` so the whole smoke suite
can be run with ``.venv-beta/bin/python -m pytest smoke_tests``.

Every fixture here returns a cheap-but-capable model. Tests that need
deeper reasoning can override the config inline — provider modules are
re-imported from ``autogen.beta.config``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from autogen.beta.config import AnthropicConfig, GeminiConfig, OpenAIConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")


def _require(env: str) -> str:
    value = os.getenv(env)
    if not value:
        pytest.skip(f"{env} not set; skipping real-API smoke test")
    return value


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
def anthropic_sonnet_config() -> AnthropicConfig:
    """Heavier Claude for tests that need real tool-use reasoning."""
    return AnthropicConfig(
        model="claude-sonnet-4-6",
        api_key=_require("ANTHROPIC_API_KEY"),
        temperature=0,
    )


@pytest.fixture()
def gemini_config() -> GeminiConfig:
    return GeminiConfig(
        model="gemini-3.1-flash-lite-preview",
        api_key=_require("GEMINI_API_KEY"),
        temperature=0,
    )


@pytest.fixture()
def gemini_flash_config() -> GeminiConfig:
    """Fast, cheap, and capable enough for tool-dispatch smoke tests.

    Default driver for all non-parity smoke suites.
    """
    return GeminiConfig(
        model="gemini-3-flash-preview",
        api_key=_require("GEMINI_API_KEY"),
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
        api_key=_require("GEMINI_API_KEY"),
        temperature=0,
        streaming=True,
    )


@pytest.fixture()
def gemini_flash_streaming_config() -> GeminiConfig:
    """Streaming-enabled Gemini Flash for chunk-observer smoke tests."""
    return GeminiConfig(
        model="gemini-3-flash-preview",
        api_key=_require("GEMINI_API_KEY"),
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
def provider_config(
    request,
    openai_config,
    anthropic_config,
    gemini_config,
):
    """Parametrized fixture that yields a config for each provider.

    Tests using this fixture run once per provider.
    """
    return {
        "openai": openai_config,
        "anthropic": anthropic_config,
        "gemini": gemini_config,
    }[request.param]
