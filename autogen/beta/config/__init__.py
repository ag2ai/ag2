# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from .config import ModelConfig
from .llms import LLMClient

try:
    from .openai import OpenAIConfig
except ImportError as e:
    from unittest.mock import Mock

    OpenAIConfig = Mock(side_effect=e)

try:
    from .openai_responses import OpenAIResponsesConfig
except ImportError as e:
    from unittest.mock import Mock

    OpenAIResponsesConfig = Mock(side_effect=e)

try:
    from .anthropic import AnthropicConfig
except ImportError as e:
    from unittest.mock import Mock

    AnthropicConfig = Mock(side_effect=e)

try:
    from .dashscope import DashScopeConfig
except ImportError as e:
    from unittest.mock import Mock

    DashScopeConfig = Mock(side_effect=e)

try:
    from .gemini import GeminiConfig
except ImportError as e:
    from unittest.mock import Mock

    GeminiConfig = Mock(side_effect=e)

try:
    from .ollama import OllamaConfig
except ImportError as e:
    from unittest.mock import Mock

    OllamaConfig = Mock(side_effect=e)

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
