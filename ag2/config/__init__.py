# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.exceptions import missing_optional_dependency

from .client import LLMClient
from .config import ModelConfig

try:
    from .openai import ContainerInfo, ContainerManager, ExpiresAfter, OpenAIConfig, OpenAIResponsesConfig
except ImportError as e:
    OpenAIConfig = missing_optional_dependency("OpenAIConfig", "openai", e)  # type: ignore[misc]
    OpenAIResponsesConfig = missing_optional_dependency("OpenAIResponsesConfig", "openai", e)  # type: ignore[misc]
    ContainerManager = missing_optional_dependency("ContainerManager", "openai", e)  # type: ignore[misc]
    ContainerInfo = missing_optional_dependency("ContainerInfo", "openai", e)  # type: ignore[misc]
    ExpiresAfter = missing_optional_dependency("ExpiresAfter", "openai", e)  # type: ignore[misc]

try:
    from .anthropic import AnthropicConfig
except ImportError as e:
    AnthropicConfig = missing_optional_dependency("AnthropicConfig", "anthropic", e)  # type: ignore[misc]

try:
    from .bedrock import BedrockConfig
except ImportError as e:
    BedrockConfig = missing_optional_dependency("BedrockConfig", "bedrock", e)  # type: ignore[misc]

try:
    from .dashscope import DashScopeConfig
except ImportError as e:
    DashScopeConfig = missing_optional_dependency("DashScopeConfig", "dashscope", e)  # type: ignore[misc]

try:
    from .gemini import GeminiConfig, VertexAIConfig
except ImportError as e:
    GeminiConfig = missing_optional_dependency("GeminiConfig", "gemini", e)  # type: ignore[misc]
    VertexAIConfig = missing_optional_dependency("VertexAIConfig", "gemini", e)  # type: ignore[misc]

try:
    from .ollama import OllamaConfig
except ImportError as e:
    OllamaConfig = missing_optional_dependency("OllamaConfig", "ollama", e)  # type: ignore[misc]

try:
    from .xai import XAIConfig
except ImportError as e:
    XAIConfig = missing_optional_dependency("XAIConfig", "xai", e)  # type: ignore[misc]

try:
    from .zai import ZAIConfig
except ImportError as e:
    ZAIConfig = missing_optional_dependency("ZAIConfig", "zai", e)  # type: ignore[misc]

__all__ = (
    "AnthropicConfig",
    "BedrockConfig",
    "ContainerInfo",
    "ContainerManager",
    "DashScopeConfig",
    "ExpiresAfter",
    "GeminiConfig",
    "LLMClient",
    "ModelConfig",
    "OllamaConfig",
    "OpenAIConfig",
    "OpenAIResponsesConfig",
    "VertexAIConfig",
    "XAIConfig",
    "ZAIConfig",
)
