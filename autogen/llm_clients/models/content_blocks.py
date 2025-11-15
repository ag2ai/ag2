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
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ============================================================================
# Content Type Enum
# ============================================================================


class ContentType(str, Enum):
    """Known content types supported by the system.

    This enum provides type safety for known content types while still allowing
    unknown types via str union in BaseContent.
    """

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    REASONING = "reasoning"
    CITATION = "citation"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


# ============================================================================
# Base Content Block
# ============================================================================


class BaseContent(BaseModel):
    """Base class for all content blocks with extension points.

    This serves as the foundation for all content types, providing:
    - Common type field for discriminated unions (ContentType enum or str for unknown types)
    - Extra field for storing unknown provider-specific data
    - Pydantic configuration for flexible field handling
    - text property for extracting text representation
    """

    type: ContentType | str  # Enum for known types, str for forward compatibility

    # Extension point for unknown fields
    extra: dict[str, Any] = Field(default_factory=dict)

    class Config:
        # Allow extra fields to be stored in model
        extra = "allow"

    def get_text(self) -> str:
        """Extract text representation of content block.

        Override in subclasses to provide specific text extraction logic.
        Default returns empty string for content blocks without text.
        """
        return ""


# ============================================================================
# Known Content Types
# ============================================================================


class TextContent(BaseContent):
    """Plain text content block."""

    type: Literal[ContentType.TEXT] = ContentType.TEXT
    text: str

    def get_text(self) -> str:
        """Get text content."""
        return self.text


class ImageContent(BaseContent):
    """Image content with optional detail level.

    Supports both URLs and data URIs (base64-encoded blob data):
    - image_url: Remote HTTP(S) URL to the image
    - data_uri: Base64-encoded image data (e.g., "data:image/png;base64,...")

    Note: Provide either image_url OR data_uri, not both.
    """

    type: Literal[ContentType.IMAGE] = ContentType.IMAGE
    image_url: str | None = None
    data_uri: str | None = None
    detail: Literal["auto", "low", "high"] | None = None

    # Inherits get_text() -> "" from BaseContent


class AudioContent(BaseContent):
    """Audio content with optional transcript.

    Supports both URLs and data URIs (base64-encoded blob data):
    - audio_url: Remote HTTP(S) URL to the audio file
    - data_uri: Base64-encoded audio data (e.g., "data:audio/mp3;base64,...")

    Note: Provide either audio_url OR data_uri, not both.
    """

    type: Literal[ContentType.AUDIO] = ContentType.AUDIO
    audio_url: str | None = None
    data_uri: str | None = None
    transcript: str | None = None

    def get_text(self) -> str:
        """Get audio transcript as text."""
        if self.transcript:
            return f"audio transcript:{self.transcript}"
        return ""


class VideoContent(BaseContent):
    """Video content block.

    Supports both URLs and data URIs (base64-encoded blob data):
    - video_url: Remote HTTP(S) URL to the video file
    - data_uri: Base64-encoded video data (e.g., "data:video/mp4;base64,...")

    Note: Provide either video_url OR data_uri, not both.
    """

    type: Literal[ContentType.VIDEO] = ContentType.VIDEO
    video_url: str | None = None
    data_uri: str | None = None

    # Inherits get_text() -> "" from BaseContent


class ReasoningContent(BaseContent):
    """Reasoning/chain-of-thought content (e.g., OpenAI o1/o3 models)."""

    type: Literal[ContentType.REASONING] = ContentType.REASONING
    reasoning: str
    summary: str | None = None

    def get_text(self) -> str:
        """Get reasoning text."""
        return self.reasoning


class CitationContent(BaseContent):
    """Web search citation or reference."""

    type: Literal[ContentType.CITATION] = ContentType.CITATION
    url: str
    title: str
    snippet: str
    relevance_score: float | None = None

    def get_text(self) -> str:
        """Get citation title as text."""
        if self.title:
            return f"citation: {self.title}"
        return ""


class ToolCallContent(BaseContent):
    """Tool/function call request."""

    type: Literal[ContentType.TOOL_CALL] = ContentType.TOOL_CALL
    id: str
    name: str
    arguments: str

    def get_text(self) -> str:
        """Get tool call as text."""
        return f"tool call name: {self.name} tool call arguments: {self.arguments}"


class ToolResultContent(BaseContent):
    """Tool/function execution result."""

    type: Literal[ContentType.TOOL_RESULT] = ContentType.TOOL_RESULT
    tool_call_id: str
    output: str

    def get_text(self) -> str:
        """Get tool result as text."""
        return f"tool result: {self.output}"


# ============================================================================
# Generic Content Block (Handles Unknown Types) - KEY FOR EXTENSIBILITY!
# ============================================================================


class GenericContent(BaseContent):
    """Handles content blocks we don't have specific types for yet.

    This is the KEY to forward compatibility:
    - When a provider adds a new content type (e.g., "reflection", "video_analysis")
    - We don't have a specific class defined yet
    - GenericContent catches it and preserves ALL fields using Pydantic's native extra='allow'
    - Users can access fields via attribute access or helper methods
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
        print(reflection.reflection)     # "Upon reviewing..." (attribute access)
        print(reflection.confidence)     # 0.87 (attribute access)

        # Extract all fields
        print(reflection.get_all_fields())   # All fields as dict
        print(reflection.get_extra_fields()) # Only unknown fields
    """

    type: ContentType | str  # Inherits from BaseContent, can be enum or any string

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style get for any field (defined or extra).

        Example:
            content.get("reflection", "N/A")
            content.get("confidence", 0.0)
        """
        # Try model_extra first (unknown fields)
        if self.model_extra and key in self.model_extra:
            return self.model_extra[key]
        # Fall back to getattr for defined fields
        return getattr(self, key, default)

    def get_all_fields(self) -> dict[str, Any]:
        """Get all fields (defined + extra) as a single dict.

        This is equivalent to model_dump() but more explicitly named.

        Example:
            all_data = content.get_all_fields()
        """
        return self.model_dump()

    def get_extra_fields(self) -> dict[str, Any]:
        """Get only the extra (unknown) fields.

        Example:
            extras = content.get_extra_fields()
            for key, value in extras.items():
                print(f"{key}: {value}")
        """
        return self.model_extra if self.model_extra else {}

    def has_field(self, key: str) -> bool:
        """Check if field exists (defined or extra).

        Example:
            if content.has_field("reflection"):
                print(content.reflection)
        """
        return hasattr(self, key)

    # Backward compatibility property for migration
    @property
    def data(self) -> dict[str, Any]:
        """Backward compatibility: access extra fields as .data

        Deprecated: Use get_extra_fields() or model_extra instead.
        """
        return self.get_extra_fields()


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

    # Registry of known types (maps both enum values and string values)
    _registry: dict[ContentType | str, type[BaseContent]] = {
        ContentType.TEXT: TextContent,
        "text": TextContent,
        ContentType.IMAGE: ImageContent,
        "image": ImageContent,
        ContentType.AUDIO: AudioContent,
        "audio": AudioContent,
        ContentType.VIDEO: VideoContent,
        "video": VideoContent,
        ContentType.REASONING: ReasoningContent,
        "reasoning": ReasoningContent,
        ContentType.CITATION: CitationContent,
        "citation": CitationContent,
        ContentType.TOOL_CALL: ToolCallContent,
        "tool_call": ToolCallContent,
        ContentType.TOOL_RESULT: ToolResultContent,
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
        # Ensure 'type' field is present for GenericContent validation
        data_with_type = {"type": content_type, **data}
        return GenericContent(**data_with_type)


# ============================================================================
# Union of all content types - easily extensible
# ============================================================================

# Union of all content types
ContentBlock = (
    TextContent
    | ImageContent
    | AudioContent
    | VideoContent
    | ReasoningContent
    | CitationContent
    | ToolCallContent
    | ToolResultContent
    | GenericContent
)  # Always last - catches unknown types!

# Note: GenericContent must be last in the Union so that specific types
# are matched first during isinstance checks
