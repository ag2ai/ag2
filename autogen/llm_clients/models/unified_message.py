# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Unified message format supporting all provider features.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .content_blocks import (
    BaseContent,
    CitationContent,
    ContentBlock,
    ReasoningContent,
    ToolCallContent,
)


class UserRoleEnum(str, Enum):
    """Standard message roles with strict typing."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


# Union type: strict typing for known roles, flexible string for unknown
UserRoleType = UserRoleEnum | str


def normalize_role(role: str | None) -> UserRoleType:
    """
    Normalize role string to UserRoleEnum for known roles, or return as-is for unknown roles.

    This function converts standard role strings to type-safe UserRoleEnum values while
    preserving unknown/custom roles as plain strings for forward compatibility.

    Args:
        role: Role string from API response (e.g., "user", "assistant", "system", "tool")

    Returns:
        UserRoleEnum for known roles, or original string for unknown/custom roles

    Examples:
        >>> normalize_role("user")
        UserRoleEnum.USER
        >>> normalize_role("assistant")
        UserRoleEnum.ASSISTANT
        >>> normalize_role("custom_role")
        "custom_role"
    """
    if not role:
        return UserRoleEnum.ASSISTANT  # Default fallback

    # Map string roles to enum values
    role_mapping = {
        "user": UserRoleEnum.USER,
        "assistant": UserRoleEnum.ASSISTANT,
        "system": UserRoleEnum.SYSTEM,
        "tool": UserRoleEnum.TOOL,
    }

    # Return enum for known roles, original string for unknown roles
    return role_mapping.get(role.lower(), role)


class UnifiedMessage(BaseModel):
    """Unified message format supporting all provider features.

    This message format can represent:
    - Text, images, audio, video
    - Reasoning blocks (OpenAI o1/o3, Anthropic)
    - Citations (web search results)
    - Tool calls and results
    - Any future content types via GenericContent
    - Any future role types via extensible role field

    The role field uses UserRoleType which provides:
    - Type-safe enum values for standard roles (UserRoleEnum.USER, etc.)
    - String literal typing for known roles ("user", "assistant", "system", "tool")
    - Flexible string fallback for unknown/future provider-specific roles
    """

    role: UserRoleType  # Type-safe for known roles, flexible for unknown
    content: list[ContentBlock]  # Rich, typed content blocks

    # Metadata
    name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)  # Provider-specific extras

    def get_text(self) -> str:
        """Extract all text content as string.

        Uses the get_text() method of each content block for unified text extraction.
        """
        text_parts = []
        for block in self.content:
            block_text = block.get_text()
            if block_text:  # Only include non-empty text
                text_parts.append(block_text)

        return " ".join(text_parts)

    def get_reasoning(self) -> list[ReasoningContent]:
        """Extract reasoning blocks."""
        return [b for b in self.content if isinstance(b, ReasoningContent)]

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
        # Handle both UserRoleEnum and string types
        if isinstance(self.role, UserRoleEnum):
            return True  # All enum values are standard roles
        # Check if string role matches any enum value
        return self.role in [e.value for e in UserRoleEnum]
