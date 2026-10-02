# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The provider a config serves from, in ag2's vocabulary and in AG-UI's."""

from ag2.config import ModelConfig, ModelProvider

# AG-UI names a file handle's provider by vendor (`openai`, `anthropic`,
# `google`); ag2 names the API, and Gemini and Vertex AI are two of Google's.
_VENDOR = {ModelProvider.GEMINI: "google", ModelProvider.VERTEXAI: "google"}

# Providers whose calls carry a Gemini thought signature: Vertex AI runs on the same client.
GEMINI_FAMILY = frozenset({ModelProvider.GEMINI, ModelProvider.VERTEXAI})


def provider_of(config: ModelConfig | None) -> ModelProvider | None:
    """The provider `config` serves from, in ag2's vocabulary, or `None` if it does not say."""
    if config is None:
        return None
    try:
        return config.provider
    except NotImplementedError:
        return None


def is_same_provider(tag: str, provider: ModelProvider | None) -> bool:
    """Whether a file handle tagged `tag` was issued by the run's `provider`, in either vocabulary."""
    return provider is not None and (tag == provider or tag == _VENDOR.get(provider))


__all__ = ("GEMINI_FAMILY", "is_same_provider", "provider_of")
