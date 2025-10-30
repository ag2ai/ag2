# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

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
)
