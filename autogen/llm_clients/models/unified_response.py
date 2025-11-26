# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Unified response format for all LLM providers.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, Field

from .content_blocks import BaseContent, ReasoningContent
from .unified_message import UnifiedMessage


class UnifiedResponse(BaseModel):
    """Provider-agnostic response format.

    This response format can represent responses from any LLM provider while
    preserving all provider-specific features (reasoning, citations, etc.).

    Features:
    - Provider agnostic (OpenAI, Anthropic, Gemini, etc.)
    - Rich content blocks (text, images, reasoning, citations)
    - Usage tracking and cost calculation
    - Provider-specific metadata preservation
    - Serializable (no attached functions)
    - Extensible status field for provider-specific statuses
    """

    # Known standard status values (for reference)
    STANDARD_STATUSES: ClassVar[list[str]] = ["completed", "in_progress", "failed"]

    id: str
    model: str
    messages: list[UnifiedMessage]

    # Usage tracking
    usage: dict[str, int] = Field(default_factory=dict)  # prompt_tokens, completion_tokens, etc.
    cost: float | None = None

    # Provider-specific
    provider: str  # "openai", "anthropic", "gemini", etc.
    provider_metadata: dict[str, Any] = Field(default_factory=dict)  # Raw provider data if needed

    # Status - extensible to support provider-specific status values
    finish_reason: str | None = None
    status: str | None = None  # Extensible - accepts any string, standard: "completed", "in_progress", "failed"

    @property
    def text(self) -> str:
        """Quick access to text content from all messages."""
        if self.messages:
            return " ".join([msg.get_text() for msg in self.messages])
        return ""

    @property
    def reasoning(self) -> list[ReasoningContent]:
        """Quick access to reasoning blocks from all messages."""
        return [block for msg in self.messages for block in msg.get_reasoning()]

    def get_content_by_type(self, content_type: str) -> list[BaseContent]:
        """Get all content blocks of a specific type across all messages.

        This is especially useful for unknown types handled by GenericContent.

        Args:
            content_type: The type string to filter by (e.g., "text", "reasoning", "reflection")

        Returns:
            List of content blocks matching the type across all messages
        """
        return [block for msg in self.messages for block in msg.get_content_by_type(content_type)]

    def is_standard_status(self) -> bool:
        """Check if this response uses a standard status value.

        Returns:
            True if status is one of the standard statuses (completed, in_progress, failed),
            False if it's a custom/future status or None
        """
        return self.status in self.STANDARD_STATUSES if self.status else False
