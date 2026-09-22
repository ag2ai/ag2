# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag2.exceptions import missing_optional_dependency

from .client import LLMClient
from .config import ModelConfig, ModelProvider

# Each provider is imported twice on purpose: the checker is shown only the real
# import, because rebinding a name it has already bound to a class is an error, and
# the install hint is runtime behaviour. See
# website/docs/contributor-guide/type-checking.mdx.
if TYPE_CHECKING:
    from .anthropic import AnthropicConfig
    from .bedrock import BedrockConfig
    from .dashscope import DashScopeConfig
    from .gemini import GeminiConfig, VertexAIConfig
    from .mistral import MistralConfig
    from .ollama import OllamaConfig
    from .openai import ContainerInfo, ContainerManager, ExpiresAfter, OpenAIConfig, OpenAIResponsesConfig
    from .xai import XAIConfig
    from .zai import ZAIConfig
else:
    try:
        from .openai import ContainerInfo, ContainerManager, ExpiresAfter, OpenAIConfig, OpenAIResponsesConfig
    except ImportError as e:
        OpenAIConfig = missing_optional_dependency("OpenAIConfig", "openai", e)
        OpenAIResponsesConfig = missing_optional_dependency("OpenAIResponsesConfig", "openai", e)
        ContainerManager = missing_optional_dependency("ContainerManager", "openai", e)
        ContainerInfo = missing_optional_dependency("ContainerInfo", "openai", e)
        ExpiresAfter = missing_optional_dependency("ExpiresAfter", "openai", e)

    try:
        from .anthropic import AnthropicConfig
    except ImportError as e:
        AnthropicConfig = missing_optional_dependency("AnthropicConfig", "anthropic", e)

    try:
        from .bedrock import BedrockConfig
    except ImportError as e:
        BedrockConfig = missing_optional_dependency("BedrockConfig", "bedrock", e)

    try:
        from .dashscope import DashScopeConfig
    except ImportError as e:
        DashScopeConfig = missing_optional_dependency("DashScopeConfig", "dashscope", e)

    try:
        from .gemini import GeminiConfig, VertexAIConfig
    except ImportError as e:
        GeminiConfig = missing_optional_dependency("GeminiConfig", "gemini", e)
        VertexAIConfig = missing_optional_dependency("VertexAIConfig", "gemini", e)

    try:
        from .mistral import MistralConfig
    except ImportError as e:
        MistralConfig = missing_optional_dependency("MistralConfig", "mistral", e)

    try:
        from .ollama import OllamaConfig
    except ImportError as e:
        OllamaConfig = missing_optional_dependency("OllamaConfig", "ollama", e)

    try:
        from .xai import XAIConfig
    except ImportError as e:
        XAIConfig = missing_optional_dependency("XAIConfig", "xai", e)

    try:
        from .zai import ZAIConfig
    except ImportError as e:
        ZAIConfig = missing_optional_dependency("ZAIConfig", "zai", e)

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
    "OpenAIConfig",
    "OpenAIResponsesConfig",
    "VertexAIConfig",
    "XAIConfig",
    "ZAIConfig",
)
