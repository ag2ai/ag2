# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime dispatcher for LLMConfigEntry subclasses.

At runtime `ConfigEntries` is the LLMConfigEntry base — concrete subclass
dispatch happens in parse_entry() via the lazy registry. Under TYPE_CHECKING
it expands to a Union of every entry class so static type checkers can still
narrow `list[ConfigEntries]` annotations.
"""

from typing import TYPE_CHECKING, Any

from autogen.llm_config.entry import LLMConfigEntry
from autogen.oai.entry_registry import get_entry_class

if TYPE_CHECKING:
    from autogen.llm_clients.openai_responses_v2 import OpenAIResponsesV2LLMConfigEntry
    from autogen.oai.anthropic import AnthropicLLMConfigEntry
    from autogen.oai.bedrock import BedrockLLMConfigEntry
    from autogen.oai.cerebras import CerebrasLLMConfigEntry
    from autogen.oai.client import (
        AzureOpenAILLMConfigEntry,
        DeepSeekLLMConfigEntry,
        OpenAILLMConfigEntry,
        OpenAIResponsesLLMConfigEntry,
        OpenAIV2LLMConfigEntry,
    )
    from autogen.oai.cohere import CohereLLMConfigEntry
    from autogen.oai.gemini import GeminiLLMConfigEntry
    from autogen.oai.groq import GroqLLMConfigEntry
    from autogen.oai.mistral import MistralLLMConfigEntry
    from autogen.oai.ollama import OllamaLLMConfigEntry
    from autogen.oai.together import TogetherLLMConfigEntry

    ConfigEntries = (
        AnthropicLLMConfigEntry
        | CerebrasLLMConfigEntry
        | BedrockLLMConfigEntry
        | AzureOpenAILLMConfigEntry
        | DeepSeekLLMConfigEntry
        | OpenAILLMConfigEntry
        | OpenAIResponsesLLMConfigEntry
        | OpenAIV2LLMConfigEntry
        | CohereLLMConfigEntry
        | GeminiLLMConfigEntry
        | GroqLLMConfigEntry
        | MistralLLMConfigEntry
        | OllamaLLMConfigEntry
        | TogetherLLMConfigEntry
        | OpenAIResponsesV2LLMConfigEntry
    )
else:
    ConfigEntries = LLMConfigEntry


def parse_entry(value: Any) -> LLMConfigEntry:
    """Coerce dict | LLMConfigEntry -> the right LLMConfigEntry subclass.

    Looks up the provider class via the lazy registry, so the provider module
    is only imported when an LLMConfig with that api_type is constructed.
    """
    if isinstance(value, LLMConfigEntry):
        # Pass through unchanged: re-validating would lose provider-specific
        # attributes that LLMConfigEntry's extra='allow' preserves but the
        # base class doesn't declare.
        return value
    if not isinstance(value, dict):
        raise TypeError(
            f"config_list entries must be dict or LLMConfigEntry, "
            f"got {type(value).__name__}"
        )
    api_type = value.get("api_type", "openai")
    return get_entry_class(api_type).model_validate(value)


__all__ = ["ConfigEntries", "parse_entry"]
