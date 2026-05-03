# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Lazy registry of LLMConfigEntry subclasses keyed by api_type literal.

Provider modules are imported on first lookup. Third-party providers
register custom api_types via `register_entry()`.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogen.llm_config.entry import LLMConfigEntry

# api_type literal -> (module path, class name).
# Keys must match the Literal declarations on each entry class — Pydantic uses
# the same string at runtime to dispatch to the right subclass via parse_entry.
_BUILTIN: dict[str, tuple[str, str]] = {
    "openai": ("autogen.oai.client", "OpenAILLMConfigEntry"),
    "azure": ("autogen.oai.client", "AzureOpenAILLMConfigEntry"),
    "deepseek": ("autogen.oai.client", "DeepSeekLLMConfigEntry"),
    "responses": ("autogen.oai.client", "OpenAIResponsesLLMConfigEntry"),
    "openai_v2": ("autogen.oai.client", "OpenAIV2LLMConfigEntry"),
    "responses_v2": (
        "autogen.llm_clients.openai_responses_v2",
        "OpenAIResponsesV2LLMConfigEntry",
    ),
    "anthropic": ("autogen.oai.anthropic", "AnthropicLLMConfigEntry"),
    "bedrock": ("autogen.oai.bedrock", "BedrockLLMConfigEntry"),
    "cerebras": ("autogen.oai.cerebras", "CerebrasLLMConfigEntry"),
    "cohere": ("autogen.oai.cohere", "CohereLLMConfigEntry"),
    "google": ("autogen.oai.gemini", "GeminiLLMConfigEntry"),
    "google_vertex": ("autogen.oai.gemini", "GeminiLLMConfigEntry"),
    "groq": ("autogen.oai.groq", "GroqLLMConfigEntry"),
    "mistral": ("autogen.oai.mistral", "MistralLLMConfigEntry"),
    "ollama": ("autogen.oai.ollama", "OllamaLLMConfigEntry"),
    "together": ("autogen.oai.together", "TogetherLLMConfigEntry"),
}

_resolved: dict[str, type["LLMConfigEntry"]] = {}
_external: dict[str, type["LLMConfigEntry"]] = {}


def register_entry(api_type: str, entry_cls: type["LLMConfigEntry"]) -> None:
    """Register a custom LLMConfigEntry subclass for an api_type.

    Use this from a third-party plugin to make LLMConfig accept your api_type
    without modifying autogen.
    """
    _external[api_type] = entry_cls


def get_entry_class(api_type: str) -> type["LLMConfigEntry"]:
    """Resolve api_type -> LLMConfigEntry subclass, importing the provider
    module on first call."""
    if api_type in _external:
        return _external[api_type]
    if api_type in _resolved:
        return _resolved[api_type]
    if api_type not in _BUILTIN:
        known = sorted(set(_BUILTIN) | set(_external))
        raise ValueError(
            f"Unknown api_type {api_type!r}. Known: {known}. "
            f"To add a custom provider, call "
            f"autogen.oai.entry_registry.register_entry(api_type, EntryClass)."
        )
    mod_name, cls_name = _BUILTIN[api_type]
    cls: type[LLMConfigEntry] = getattr(import_module(mod_name), cls_name)
    _resolved[api_type] = cls
    return cls


__all__ = ["get_entry_class", "register_entry"]
