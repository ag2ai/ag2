# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Unified message format supporting all provider features.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from .content_blocks import (
    BaseContent,
    CitationContent,
    ContentBlock,
    ReasoningContent,
    TextContent,
    ThinkingContent,
    ToolCallContent,
)


class UnifiedMessage(BaseModel):
    """Unified message format supporting all provider features.

    This message format can represent:
    - Text, images, audio, video
    - Reasoning blocks (OpenAI o1/o3)
    - Thinking blocks (Anthropic)
    - Citations (web search results)
    - Tool calls and results
    - Any future content types via GenericContent
    - Any future role types via extensible role field

    The role field is extensible to support future provider-specific roles.
    Standard roles are: "user", "assistant", "system", "tool"
    But any string value is accepted for forward compatibility.
    """

    # Known standard roles (for reference and validation)
    STANDARD_ROLES: ClassVar[list[str]] = ["user", "assistant", "system", "tool"]

    role: str  # Extensible - accepts any string, not just standard roles
    content: list[ContentBlock]  # Rich, typed content blocks

    # Metadata
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)  # Provider-specific extras

    def get_text(self) -> str:
        """Extract all text content as string."""
        text_parts = []
        for block in self.content:
            if isinstance(block, TextContent):
                text_parts.append(block.text)
            elif isinstance(block, ReasoningContent):
                text_parts.append(block.reasoning)
            elif isinstance(block, ThinkingContent):
                text_parts.append(block.thinking)
        return " ".join(text_parts)

    def get_reasoning(self) -> list[ReasoningContent]:
        """Extract reasoning blocks."""
        return [b for b in self.content if isinstance(b, ReasoningContent)]

    def get_thinking(self) -> list[ThinkingContent]:
        """Extract thinking blocks."""
        return [b for b in self.content if isinstance(b, ThinkingContent)]

    def get_citations(self) -> list[CitationContent]:
        """Extract citations."""
        return [b for b in self.content if isinstance(b, CitationContent)]

    def get_tool_calls(self) -> list[ToolCallContent]:
        """Extract tool calls."""
        return [b for b in self.content if isinstance(b, ToolCallContent)]

    def get_content_by_type(self, content_type: str) -> list[BaseContent]:
        """Get all content blocks of a specific type.

        This is especially useful for unknown types handled by GenericContent.

        Args:
            content_type: The type string to filter by (e.g., "text", "reasoning", "reflection")

        Returns:
            List of content blocks matching the type
        """
        return [b for b in self.content if b.type == content_type]

    def is_standard_role(self) -> bool:
        """Check if this message uses a standard role.

        Returns:
            True if role is one of the standard roles (user, assistant, system, tool),
            False if it's a custom/future role
        """
        return self.role in self.STANDARD_ROLES
