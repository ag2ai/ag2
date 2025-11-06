# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Unified response models for LLM clients.

This package provides:
- Content block system with extensibility via GenericContent
- UnifiedMessage format supporting all provider features
- UnifiedResponse provider-agnostic format
- ContentParser for registry-based content type handling
"""

from .content_blocks import (
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
    VideoContent,
)
from .unified_message import UnifiedMessage
from .unified_response import UnifiedResponse

__all__ = [
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
    # Unified formats
    "UnifiedMessage",
    "UnifiedResponse",
    "VideoContent",
]
