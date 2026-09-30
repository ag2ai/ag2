# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The provider a config serves from, in ag2's vocabulary and in AG-UI's."""

from ag2.config import ModelConfig

# AG-UI names a file handle's provider by vendor (`openai`, `anthropic`,
# `google`); ag2 names the API, and Gemini and Vertex AI are two of Google's.
_VENDOR = {"gemini": "google", "vertexai": "google"}


def provider_of(config: ModelConfig | None) -> str | None:
    """The provider `config` serves from, in ag2's vocabulary, or `None` if it does not say."""
    if config is None:
        return None
    try:
        return config.provider.value
    except NotImplementedError:
        return None


def is_same_provider(tag: str, provider: str | None) -> bool:
    """Whether a file handle tagged `tag` was issued by the run's `provider`, in either vocabulary."""
    return provider is not None and (tag == provider or tag == _VENDOR.get(provider))


__all__ = ("is_same_provider", "provider_of")
