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
    from autogen.llm_clients import ModelClientV2, UnifiedResponse
    from autogen.llm_clients.models import ContentParser, ReasoningContent

    # Implement ModelClientV2
    class MyClient:
        def create(self, params: dict) -> UnifiedResponse:
            # Return rich responses with reasoning, thinking, etc.
            ...

        def create_v1_compatible(self, params: dict) -> ChatCompletionExtended:
            # Backward compatible legacy format
            ...

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

__all__ = [
    # Protocol
    "ModelClientV2",
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
