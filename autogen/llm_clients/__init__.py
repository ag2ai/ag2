# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
LLM Clients package for AG2.

This package provides the next-generation LLM client interface (ModelClientV2)
and unified response models that support rich content blocks from all providers.

Key Features:
- Provider-agnostic response format (UnifiedResponse)
- Rich content blocks (reasoning, thinking, citations, etc.)
- Forward compatibility with unknown content types via GenericContent
- Backward compatibility with existing ChatCompletion-based interface
- Extensible content type registry

Usage:
    from autogen.llm_clients import OpenAICompletionsClient, GeminiStatelessClient, UnifiedResponse
    from autogen.llm_clients.models import ContentParser, ReasoningContent, ThinkingContent

    # Use OpenAI Chat Completions Client
    client = OpenAICompletionsClient(api_key="...")
    response = client.create({
        "model": "o1-preview",
        "messages": [{"role": "user", "content": "Explain quantum computing"}]
    })

    # Access reasoning blocks
    for reasoning in response.reasoning:
        print(reasoning.reasoning)

    # Use Gemini Stateless Client
    gemini = GeminiStatelessClient(api_key="...")
    response = gemini.create({
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Explain quantum computing"}],
        "thinking_config": {"include_thoughts": True}
    })

    # Access thinking blocks (Gemini 2.5+)
    for thinking in response.thinking:
        print(thinking.thinking)

    # Register custom content types
    ContentParser.register("custom_type", CustomContent)
"""

from .client_v2 import ModelClientV2
from .gemini_stateless_client import GeminiStatelessClient
from .models import (
    AudioContent,
    BaseContent,
    CitationContent,
    ContentBlock,
    ContentParser,
    GenericContent,
    ImageContent,
    ReasoningContent,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    ToolResultContent,
    UnifiedMessage,
    UnifiedResponse,
    VideoContent,
)
from .openai_completions_client import OpenAICompletionsClient

__all__ = [
    # Content blocks
    "AudioContent",
    "BaseContent",
    "CitationContent",
    "ContentBlock",
    "ContentParser",
    # Clients
    "GeminiStatelessClient",
    "GenericContent",
    "ImageContent",
    # Protocol
    "ModelClientV2",
    "OpenAICompletionsClient",
    "ReasoningContent",
    "TextContent",
    "ThinkingContent",
    "ToolCallContent",
    "ToolResultContent",
    # Unified formats
    "UnifiedMessage",
    "UnifiedResponse",
    "VideoContent",
]
