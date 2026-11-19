# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag2._lazy import LazyExport, lazy_exports

from .client import LLMClient
from .config import ModelConfig, ModelProvider

if TYPE_CHECKING:
    from .anthropic import AnthropicConfig
    from .bedrock import BedrockConfig
    from .dashscope import DashScopeConfig
    from .gemini import GeminiConfig, VertexAIConfig
    from .mistral import MistralConfig
    from .ollama import OllamaConfig
    from .openai import (
        ContainerInfo,
        ContainerManager,
        ExpiresAfter,
        OpenAICompletionsConfig,
        OpenAIConfig,
        OpenAIDecisionsConfig,
        OpenAIResponsesConfig,
    )
    from .typesafe import TypeSafeConfig
    from .xai import XAIConfig
    from .zai import ZAIConfig

# Provider SDKs load on first use; see docs/adr/0036-provider-exports-load-on-first-use.md.
_OPENAI = LazyExport(".openai", "openai")
_GEMINI = LazyExport(".gemini", "gemini")
_LAZY_EXPORTS = {
    "AnthropicConfig": LazyExport(".anthropic", "anthropic"),
    "BedrockConfig": LazyExport(".bedrock", "bedrock"),
    "ContainerInfo": _OPENAI,
    "ContainerManager": _OPENAI,
    "DashScopeConfig": LazyExport(".dashscope", "dashscope"),
    "ExpiresAfter": _OPENAI,
    "GeminiConfig": _GEMINI,
    "MistralConfig": LazyExport(".mistral", "mistral"),
    "OllamaConfig": LazyExport(".ollama", "ollama"),
    "OpenAICompletionsConfig": _OPENAI,
    "OpenAIConfig": _OPENAI,
    "OpenAIDecisionsConfig": _OPENAI,
    "OpenAIResponsesConfig": _OPENAI,
    "TypeSafeConfig": LazyExport(".typesafe", "typesafe"),
    "VertexAIConfig": _GEMINI,
    "XAIConfig": LazyExport(".xai", "xai"),
    "ZAIConfig": LazyExport(".zai", "zai"),
}

__getattr__ = lazy_exports(__name__, globals(), _LAZY_EXPORTS)

__all__ = (
    "AnthropicConfig",
    "BedrockConfig",
    "ContainerInfo",
    "ContainerManager",
    "DashScopeConfig",
    "ExpiresAfter",
    "GeminiConfig",
    "LLMClient",
    "MistralConfig",
    "ModelConfig",
    "ModelProvider",
    "OllamaConfig",
    "OpenAICompletionsConfig",
    "OpenAIConfig",
    "OpenAIDecisionsConfig",
    "OpenAIResponsesConfig",
    "TypeSafeConfig",
    "VertexAIConfig",
    "XAIConfig",
    "ZAIConfig",
)
