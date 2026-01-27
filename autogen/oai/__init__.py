# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
from ..cache.cache import Cache
from ..import_utils import optional_import_block
from .anthropic import AnthropicLLMConfigEntry
from .bedrock import BedrockLLMConfigEntry
from .cerebras import CerebrasLLMConfigEntry
from .client import (
    AzureOpenAILLMConfigEntry,
    DeepSeekLLMConfigEntry,
    OpenAILLMConfigEntry,
    OpenAIResponsesLLMConfigEntry,
    OpenAIV2LLMConfigEntry,
    OpenAIWrapper,
)
from .cohere import CohereLLMConfigEntry

# Wrap GeminiLLMConfigEntry import in optional_import_block since gemini.py
with optional_import_block() as gemini_result:
    from .gemini import GeminiLLMConfigEntry

if gemini_result.is_successful:
    _GeminiLLMConfigEntry = GeminiLLMConfigEntry
else:
    from ..llm_config.entry import LLMConfigEntry

    _GeminiLLMConfigEntry = LLMConfigEntry  # type: ignore[assignment,misc]

GeminiLLMConfigEntry = _GeminiLLMConfigEntry

from .groq import GroqLLMConfigEntry
from .mistral import MistralLLMConfigEntry
from .ollama import OllamaLLMConfigEntry
from .openai_utils import (
    config_list_from_dotenv,
    config_list_from_json,
    config_list_from_models,
    config_list_gpt4_gpt35,
    config_list_openai_aoai,
    filter_config,
    get_config_list,
    get_first_llm_config,
)
from .shared_utils import normalize_pydantic_schema_to_dict
from .together import TogetherLLMConfigEntry

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
    "config_list_from_json",
    "config_list_from_models",
    "config_list_gpt4_gpt35",
    "config_list_openai_aoai",
    "filter_config",
    "get_config_list",
    "get_first_llm_config",
    "normalize_pydantic_schema_to_dict",
]
