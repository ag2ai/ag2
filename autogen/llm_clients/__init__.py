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
    from autogen.llm_clients import OpenAIResponsesClient, UnifiedResponse
    from autogen.llm_clients.models import ContentParser, ReasoningContent

    # Use OpenAI Responses Client
    client = OpenAIResponsesClient(api_key="...")
    response = client.create({
        "model": "o1-preview",
        "messages": [{"role": "user", "content": "Explain quantum computing"}]
    })

    # Access reasoning blocks
    for reasoning in response.reasoning:
        print(reasoning.reasoning)

    # Register custom content types
    ContentParser.register("custom_type", CustomContent)
"""

from .client_v2 import ModelClientV2
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
from .openai_responses_client import OpenAIResponsesClient

__all__ = [
    # Protocol
    "ModelClientV2",
    # Clients
    "OpenAIResponsesClient",
    # Content blocks
    "AudioContent",
    "BaseContent",
    "CitationContent",
    "ContentBlock",
    "ContentParser",
    "GenericContent",
    "ImageContent",
    "ReasoningContent",
    "TextContent",
    "ThinkingContent",
    "ToolCallContent",
    "ToolResultContent",
    "VideoContent",
    # Unified formats
    "UnifiedMessage",
    "UnifiedResponse",
]
