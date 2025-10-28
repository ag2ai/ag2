# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Content block system for unified LLM responses.

This module provides an extensible content block architecture that supports:
- Known content types (text, image, audio, video, reasoning, thinking, citations, tool calls)
- Unknown content types via GenericContent (forward compatibility)
- Registry-based parsing for extensibility
"""

import warnings
from typing import Any, Literal, Union

from pydantic import BaseModel, Field

# ============================================================================
# Base Content Block
# ============================================================================


class BaseContent(BaseModel):
    """Base class for all content blocks with extension points.

    This serves as the foundation for all content types, providing:
    - Common type field for discriminated unions
    - Extra field for storing unknown provider-specific data
    - Pydantic configuration for flexible field handling
    """

    type: str  # Not Literal - allows any string for unknown types!

    # Extension point for unknown fields
    extra: dict[str, Any] = Field(default_factory=dict)

    class Config:
        # Allow extra fields to be stored in model
        extra = "allow"


# ============================================================================
# Known Content Types
# ============================================================================


class TextContent(BaseContent):
    """Plain text content block."""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseContent):
    """Image content with optional detail level."""

    type: Literal["image"] = "image"
    image_url: str
    detail: Literal["auto", "low", "high"] | None = None


class AudioContent(BaseContent):
    """Audio content with optional transcript."""

    type: Literal["audio"] = "audio"
    audio_url: str
    transcript: str | None = None


class VideoContent(BaseContent):
    """Video content block."""

    type: Literal["video"] = "video"
    video_url: str


class ReasoningContent(BaseContent):
    """Reasoning/chain-of-thought content (e.g., OpenAI o1/o3 models)."""

    type: Literal["reasoning"] = "reasoning"
    reasoning: str
    summary: str | None = None


class ThinkingContent(BaseContent):
    """Thinking process content (e.g., Anthropic thinking mode)."""

    type: Literal["thinking"] = "thinking"
    thinking: str


class CitationContent(BaseContent):
    """Web search citation or reference."""

    type: Literal["citation"] = "citation"
    url: str
    title: str
    snippet: str
    relevance_score: float | None = None


class ToolCallContent(BaseContent):
    """Tool/function call request."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: str


class ToolResultContent(BaseContent):
    """Tool/function execution result."""

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    output: str


# ============================================================================
# Generic Content Block (Handles Unknown Types) - KEY FOR EXTENSIBILITY!
# ============================================================================


class GenericContent(BaseContent):
    """Handles content blocks we don't have specific types for yet.

    This is the KEY to forward compatibility:
    - When a provider adds a new content type (e.g., "reflection", "video_analysis")
    - We don't have a specific class defined yet
    - GenericContent catches it and preserves ALL fields
    - Users can access fields via .data dict or attribute access
    - Later we can add a specific typed class without breaking anything

    Example:
        # Provider returns new "reflection" type
        reflection = GenericContent(
            type="reflection",
            reflection="Upon reviewing...",
            confidence=0.87,
            corrections=["fix1", "fix2"]
        )

        # Access fields immediately
        print(reflection.type)           # "reflection"
        print(reflection.data)           # {"reflection": "...", "confidence": 0.87, ...}
        print(reflection.reflection)     # "Upon reviewing..." (attribute access)
        print(reflection.confidence)     # 0.87 (attribute access)
    """

    type: str  # Any string allowed - not restricted to Literal
    data: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        """Store all unknown fields in .data for easy access."""
        type_val = kwargs.pop("type", "unknown")
        extra_val = kwargs.pop("extra", {})

        # All remaining fields go into .data
        remaining = {k: v for k, v in kwargs.items() if k != "data"}

        # If data was explicitly provided, merge with remaining
        if "data" in kwargs:
            remaining.update(kwargs["data"])

        super().__init__(type=type_val, data=remaining, extra=extra_val)

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to data fields.

        Example:
            content.reflection  # Instead of content.data["reflection"]
        """
        if name.startswith("_"):
            # Don't intercept private attributes
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if name in self.__dict__.get("data", {}):
            return self.data[name]

        raise AttributeError(f"'{self.type}' content has no attribute '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style get for convenience."""
        return self.data.get(key, default)


# ============================================================================
# Smart Content Parser - Routes to Specific or Generic Types
# ============================================================================


class ContentParser:
    """Parses content blocks with automatic fallback to GenericContent.

    This enables extensibility:
    1. Try to parse as known type (TextContent, ReasoningContent, etc.)
    2. If unknown type or parsing fails → GenericContent (preserves data)
    3. Later add new types to registry without breaking existing code
    """

    # Registry of known types
    _registry: dict[str, type[BaseContent]] = {
        "text": TextContent,
        "image": ImageContent,
        "audio": AudioContent,
        "video": VideoContent,
        "reasoning": ReasoningContent,
        "thinking": ThinkingContent,
        "citation": CitationContent,
        "tool_call": ToolCallContent,
        "tool_result": ToolResultContent,
        # Registry grows as we add types - no code changes elsewhere!
    }

    @classmethod
    def register(cls, content_type: str, content_class: type[BaseContent]) -> None:
        """Register a new content type.

        Example:
            # Add support for new "reflection" type
            class ReflectionContent(BaseContent):
                type: Literal["reflection"] = "reflection"
                reflection: str
                confidence: float

            ContentParser.register("reflection", ReflectionContent)
        """
        cls._registry[content_type] = content_class

    @classmethod
    def parse(cls, data: dict[str, Any]) -> BaseContent:
        """Parse content block data to appropriate type.

        Returns:
            - Specific type (TextContent, ReasoningContent, etc.) if known
            - GenericContent if unknown type or parsing fails
            - Always succeeds - never raises for unknown types!
        """
        content_type = data.get("type", "unknown")

        # Try known type
        if content_type in cls._registry:
            content_class = cls._registry[content_type]
            try:
                return content_class(**data)
            except Exception as e:
                # Parsing failed - fall back to generic
                # This ensures we never lose data due to validation errors
                warnings.warn(
                    f"Failed to parse {content_type} as {content_class.__name__}: {e}. Using GenericContent instead.",
                    UserWarning,
                    stacklevel=2,
                )
                return GenericContent(**data)

        # Unknown type - use generic (KEY FOR FORWARD COMPATIBILITY!)
        return GenericContent(**data)


# ============================================================================
# Union of all content types - easily extensible
# ============================================================================

# Union of all content types
ContentBlock = Union[
    TextContent,
    ImageContent,
    AudioContent,
    VideoContent,
    ReasoningContent,
    ThinkingContent,
    CitationContent,
    ToolCallContent,
    ToolResultContent,
    GenericContent,  # Always last - catches unknown types!
]

# Note: GenericContent must be last in the Union so that specific types
# are matched first during isinstance checks
