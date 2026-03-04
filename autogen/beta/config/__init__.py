# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from .config import ModelConfig
from .llms import LLMClient


def _missing_optional_dependency_config(config_name: str, extra: str, error: ImportError) -> Mock:
    def _raise_helpful_import_error(*args: object, **kwargs: object) -> None:
        raise ImportError(
            f'{config_name} requires optional provider dependencies. Install with `pip install "ag2[{extra}]"`'
        ) from error

    return Mock(side_effect=_raise_helpful_import_error)


try:
    from .openai import OpenAIConfig
except ImportError as e:
    OpenAIConfig = _missing_optional_dependency_config("OpenAIConfig", "openai", e)

try:
    from .openai_responses import OpenAIResponsesConfig
except ImportError as e:
    OpenAIResponsesConfig = _missing_optional_dependency_config("OpenAIResponsesConfig", "openai", e)

try:
    from .anthropic import AnthropicConfig
except ImportError as e:
    AnthropicConfig = _missing_optional_dependency_config("AnthropicConfig", "anthropic", e)

try:
    from .dashscope import DashScopeConfig
except ImportError as e:
    DashScopeConfig = _missing_optional_dependency_config("DashScopeConfig", "dashscope", e)

try:
    from .gemini import GeminiConfig
except ImportError as e:
    GeminiConfig = _missing_optional_dependency_config("GeminiConfig", "gemini", e)

try:
    from .ollama import OllamaConfig
except ImportError as e:
    OllamaConfig = _missing_optional_dependency_config("OllamaConfig", "ollama", e)

__all__ = (
    "AnthropicConfig",
    "DashScopeConfig",
    "GeminiConfig",
    "LLMClient",
    "ModelConfig",
    "OllamaConfig",
    "OpenAIConfig",
    "OpenAIResponsesConfig",
)
