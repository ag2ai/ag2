# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
ModelClientV2 protocol for next-generation LLM client interface.

This protocol defines the interface for LLM clients that return rich,
provider-agnostic responses (UnifiedResponse) while maintaining backward
compatibility with the existing ChatCompletion-based interface.
"""

from typing import Any, Protocol

from .models.unified_response import UnifiedResponse


class ModelClientV2(Protocol):
    """Next-generation client interface with rich unified responses.

    This protocol extends the current ModelClient interface to support:
    - Rich content blocks (reasoning, thinking, citations, etc.)
    - Provider-agnostic response format
    - Forward compatibility with unknown content types
    - Backward compatibility via create_v1_compatible()

    Migration Path:
    1. Implement create() to return UnifiedResponse
    2. Implement create_v1_compatible() for backward compatibility
    3. Gradually migrate agents to use UnifiedResponse directly
    4. Eventually deprecate v1 compatibility layer

    Example Implementation:
        class OpenAIClientV2:
            def create(self, params: dict[str, Any]) -> UnifiedResponse:
                # Call OpenAI API
                raw_response = openai.chat.completions.create(**params)

                # Transform to UnifiedResponse with rich content blocks
                return self._to_unified_response(raw_response)

            def create_v1_compatible(self, params: dict[str, Any]) -> ChatCompletionExtended:
                # Get rich response
                response = self.create(params)

                # Convert to legacy format
                return self._to_v1(response)
    """

    RESPONSE_USAGE_KEYS: list[str] = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    def create(self, params: dict[str, Any]) -> UnifiedResponse:
        """Create a completion and return unified response.

        Args:
            params: Request parameters (messages, model, temperature, etc.)

        Returns:
            UnifiedResponse with rich content blocks preserving all provider features
        """
        ...

    def create_v1_compatible(self, params: dict[str, Any]) -> Any:
        """Backward compatible - returns ChatCompletionExtended.

        This method provides backward compatibility during migration by:
        1. Calling create() to get UnifiedResponse
        2. Converting to ChatCompletionExtended format
        3. Flattening rich content to legacy format

        Args:
            params: Request parameters (same as create())

        Returns:
            ChatCompletionExtended for compatibility with existing agents

        Note:
            This method may lose information (reasoning blocks, citations, etc.)
            when converting to the legacy format. Prefer create() for new code.
        """
        ...

    def cost(self, response: UnifiedResponse) -> float:
        """Calculate cost from response.

        Args:
            response: UnifiedResponse from create()

        Returns:
            Cost in USD for the API call
        """
        ...

    @staticmethod
    def get_usage(response: UnifiedResponse) -> dict[str, Any]:
        """Extract usage statistics from response.

        Args:
            response: UnifiedResponse from create()

        Returns:
            Dict with keys from RESPONSE_USAGE_KEYS
        """
        ...

    def message_retrieval(self, response: UnifiedResponse) -> list[str]:
        """Retrieve text content from response messages.

        This is a convenience method for extracting simple text content
        from UnifiedResponse. For rich content (reasoning, citations, etc.),
        access response.messages directly.

        Args:
            response: UnifiedResponse from create()

        Returns:
            List of text strings from message content blocks
        """
        ...
