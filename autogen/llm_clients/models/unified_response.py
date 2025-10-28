# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Unified response format for all LLM providers.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from .content_blocks import BaseContent, ReasoningContent, ThinkingContent
from .unified_message import UnifiedMessage


class UnifiedResponse(BaseModel):
    """Provider-agnostic response format.

    This response format can represent responses from any LLM provider while
    preserving all provider-specific features (reasoning, thinking, citations, etc.).

    Features:
    - Provider agnostic (OpenAI, Anthropic, Gemini, etc.)
    - Rich content blocks (text, images, reasoning, thinking, citations)
    - Usage tracking and cost calculation
    - Provider-specific metadata preservation
    - Serializable (no attached functions)
    """

    id: str
    model: str
    messages: list[UnifiedMessage]

    # Usage tracking
    usage: dict[str, int] = Field(default_factory=dict)  # prompt_tokens, completion_tokens, etc.
    cost: float | None = None

    # Provider-specific
    provider: str  # "openai", "anthropic", "gemini", etc.
    provider_metadata: dict[str, Any] = Field(default_factory=dict)  # Raw provider data if needed

    # Status
    finish_reason: str | None = None
    status: Literal["completed", "in_progress", "failed"] | None = None

    @property
    def text(self) -> str:
        """Quick access to text content from first message."""
        if self.messages:
            return self.messages[0].get_text()
        return ""

    @property
    def reasoning(self) -> list[ReasoningContent]:
        """Quick access to reasoning blocks from all messages."""
        return [block for msg in self.messages for block in msg.get_reasoning()]

    @property
    def thinking(self) -> list[ThinkingContent]:
        """Quick access to thinking blocks from all messages."""
        return [block for msg in self.messages for block in msg.get_thinking()]

    def get_content_by_type(self, content_type: str) -> list[BaseContent]:
        """Get all content blocks of a specific type across all messages.

        This is especially useful for unknown types handled by GenericContent.

        Args:
            content_type: The type string to filter by (e.g., "text", "reasoning", "reflection")

        Returns:
            List of content blocks matching the type across all messages
        """
        return [block for msg in self.messages for block in msg.get_content_by_type(content_type)]
