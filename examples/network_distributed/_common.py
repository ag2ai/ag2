# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the distributed-network example scripts.

Each script in this folder runs as its own OS process. They share only
this tiny module: ``.env`` loading and a provider → ``ModelConfig``
factory so the hub, responder, and initiator can each pick a backend
without duplicating the mapping.
"""

from pathlib import Path

from autogen.beta.config import AnthropicConfig, GeminiConfig, ModelConfig, OpenAIConfig

# One representative model per provider. Swap freely — the network is
# provider-neutral, so any mix of these can share one hub.
MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-5.4-mini",
    "gemini": "gemini-3.5-flash",
}


def load_env() -> None:
    """Load API keys from the repo-root ``.env`` if python-dotenv is present.

    A no-op when dotenv is not installed — the scripts then rely on the
    provider keys already being exported in the environment.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def make_config(provider: str) -> ModelConfig:
    """Build a ``ModelConfig`` for ``provider`` (api_key read from env)."""
    key = provider.lower()
    model = MODELS.get(key)
    if model is None:
        raise SystemExit(f"unknown provider {provider!r}; choose one of {', '.join(MODELS)}")
    if key == "anthropic":
        return AnthropicConfig(model=model, temperature=0)
    if key == "openai":
        return OpenAIConfig(model=model, temperature=0)
    return GeminiConfig(model=model, temperature=0)
