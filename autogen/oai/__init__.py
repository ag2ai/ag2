# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

"""LLM provider integrations.

Provider-specific entry classes (AnthropicLLMConfigEntry, GeminiLLMConfigEntry,
etc.) are loaded on first attribute access via PEP 562 __getattr__. The
TYPE_CHECKING block below mirrors the lazy entries so static type checkers
still see the real classes.

Adding a new provider requires THREE coordinated edits:
    1. autogen/oai/entry_registry.py - register api_type -> EntryClass
       in `_BUILTIN` so LLMConfig validation can dispatch to it.
    2. THIS FILE - add the EntryClass name to `_LAZY_ENTRIES` and to
       the `TYPE_CHECKING` block below, so `from autogen.oai import
       NewProviderEntry` resolves both at runtime and for type checkers.
    3. autogen/llm_config/types.py - add the EntryClass to the
       `ConfigEntries` Union under `TYPE_CHECKING` for type narrowing.
"""

from typing import TYPE_CHECKING, Any

from ..cache.cache import Cache
from .client import (
    AzureOpenAILLMConfigEntry,
    DeepSeekLLMConfigEntry,
    OpenAILLMConfigEntry,
    OpenAIResponsesLLMConfigEntry,
    OpenAIV2LLMConfigEntry,
    OpenAIWrapper,
)
from .openai_utils import (
    config_list_from_dotenv,
    config_list_from_models,
    config_list_gpt4_gpt35,
    config_list_openai_aoai,
    get_config_list,
    get_first_llm_config,
)

if TYPE_CHECKING:
    from .anthropic import AnthropicLLMConfigEntry
    from .bedrock import BedrockLLMConfigEntry
    from .cerebras import CerebrasLLMConfigEntry
    from .cohere import CohereLLMConfigEntry
    from .gemini import GeminiLLMConfigEntry
    from .groq import GroqLLMConfigEntry
    from .mistral import MistralLLMConfigEntry
    from .ollama import OllamaLLMConfigEntry
    from .together import TogetherLLMConfigEntry

# OpenAI*LLMConfigEntry classes are eagerly available via .client (the always-
# loaded default); only the rest are lazy here.
_LAZY_ENTRIES: dict[str, tuple[str, str]] = {
    "AnthropicLLMConfigEntry": ("autogen.oai.anthropic", "AnthropicLLMConfigEntry"),
    "BedrockLLMConfigEntry": ("autogen.oai.bedrock", "BedrockLLMConfigEntry"),
    "CerebrasLLMConfigEntry": ("autogen.oai.cerebras", "CerebrasLLMConfigEntry"),
    "CohereLLMConfigEntry": ("autogen.oai.cohere", "CohereLLMConfigEntry"),
    "GeminiLLMConfigEntry": ("autogen.oai.gemini", "GeminiLLMConfigEntry"),
    "GroqLLMConfigEntry": ("autogen.oai.groq", "GroqLLMConfigEntry"),
    "MistralLLMConfigEntry": ("autogen.oai.mistral", "MistralLLMConfigEntry"),
    "OllamaLLMConfigEntry": ("autogen.oai.ollama", "OllamaLLMConfigEntry"),
    "TogetherLLMConfigEntry": ("autogen.oai.together", "TogetherLLMConfigEntry"),
}


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _LAZY_ENTRIES:
        from importlib import import_module

        mod_path, cls_name = _LAZY_ENTRIES[name]
        cls = getattr(import_module(mod_path), cls_name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module 'autogen.oai' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ENTRIES))


__all__ = [
    "AnthropicLLMConfigEntry",
    "AzureOpenAILLMConfigEntry",
    "BedrockLLMConfigEntry",
    "Cache",
    "CerebrasLLMConfigEntry",
    "CohereLLMConfigEntry",
    "DeepSeekLLMConfigEntry",
    "GeminiLLMConfigEntry",
    "GroqLLMConfigEntry",
    "MistralLLMConfigEntry",
    "OllamaLLMConfigEntry",
    "OpenAILLMConfigEntry",
    "OpenAIResponsesLLMConfigEntry",
    "OpenAIV2LLMConfigEntry",
    "OpenAIWrapper",
    "TogetherLLMConfigEntry",
    "config_list_from_dotenv",
    "config_list_from_models",
    "config_list_gpt4_gpt35",
    "config_list_openai_aoai",
    "get_config_list",
    "get_first_llm_config",
]
