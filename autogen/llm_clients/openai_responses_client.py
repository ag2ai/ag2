# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
OpenAI Responses API Client implementing ModelClientV2 and ModelClient protocols.

This client handles the OpenAI Responses API which returns rich responses with:
- Reasoning blocks (o1, o3 models)
- Web search citations
- File search results
- Standard chat messages

The client preserves all provider-specific features in UnifiedResponse format
and is compatible with AG2's agent system through ModelClient protocol.
"""

from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # Will raise error in __init__ if not installed

from ..llm_config.client import ModelClient
from .models import (
    CitationContent,
    ReasoningContent,
    TextContent,
    ToolCallContent,
    UnifiedMessage,
    UnifiedResponse,
)


class OpenAIResponsesClient(ModelClient):
    """
    OpenAI Responses API client implementing ModelClientV2 protocol.

    This client works with OpenAI's Responses API (used by o1, o3 models) which
    returns structured output with reasoning blocks, web search results, and more.

    Key Features:
    - Preserves reasoning blocks as ReasoningContent
    - Extracts web search citations as CitationContent
    - Handles tool calls and results
    - Provides backward compatibility via create_v1_compatible()

    Example:
        client = OpenAIResponsesClient(api_key="...")

        # Get rich response with reasoning
        response = client.create({
            "model": "o1-preview",
            "messages": [{"role": "user", "content": "Explain quantum computing"}]
        })

        # Access reasoning blocks
        for reasoning in response.reasoning:
            print(f"Reasoning: {reasoning.reasoning}")

        # Get text response
        print(f"Answer: {response.text}")
    """

    RESPONSE_USAGE_KEYS: list[str] = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        """
        Initialize OpenAI Responses API client.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            base_url: Custom base URL for OpenAI API
            timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to OpenAI client
        """
        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)
        self._cost_per_token = {
            # o1 series pricing (example - update with actual pricing)
            "o1-preview": {"prompt": 0.015 / 1000, "completion": 0.060 / 1000},
            "o1-mini": {"prompt": 0.003 / 1000, "completion": 0.012 / 1000},
            "o3-mini": {"prompt": 0.003 / 1000, "completion": 0.012 / 1000},
            # GPT-4 series
            "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
            "gpt-4-turbo": {"prompt": 0.01 / 1000, "completion": 0.03 / 1000},
        }

    def create(self, params: dict[str, Any]) -> UnifiedResponse:  # type: ignore[override]
        """
        Create a completion and return UnifiedResponse with all features preserved.

        This method implements ModelClient.create() but returns UnifiedResponse instead
        of ModelClientResponseProtocol. The rich UnifiedResponse structure is compatible
        via duck typing - it has .model attribute and works with message_retrieval().

        Args:
            params: Request parameters including:
                - model: Model name (e.g., "o1-preview")
                - messages: List of message dicts
                - temperature: Optional temperature (not supported by o1 models)
                - max_tokens: Optional max completion tokens
                - tools: Optional tool definitions
                - **other OpenAI parameters

        Returns:
            UnifiedResponse with reasoning blocks, citations, and all content preserved
        """
        # Call OpenAI API
        response = self.client.chat.completions.create(**params)

        # Transform to UnifiedResponse
        return self._transform_response(response, params.get("model", "unknown"))

    def _transform_response(self, openai_response: Any, model: str) -> UnifiedResponse:
        """
        Transform OpenAI response to UnifiedResponse.

        This handles both the standard ChatCompletion format and the Responses API
        format with reasoning blocks and web search results.

        Args:
            openai_response: Raw OpenAI API response
            model: Model name

        Returns:
            UnifiedResponse with all content blocks properly typed
        """
        messages = []

        # Process each choice
        for choice in openai_response.choices:
            content_blocks = []
            message_obj = choice.message

            # Extract reasoning if present (o1/o3 models)
            if hasattr(message_obj, "reasoning") and message_obj.reasoning:
                content_blocks.append(
                    ReasoningContent(
                        type="reasoning",
                        reasoning=message_obj.reasoning,
                        summary=None,
                    )
                )

            # Extract text content
            if message_obj.content:
                if isinstance(message_obj.content, str):
                    content_blocks.append(TextContent(type="text", text=message_obj.content))
                elif isinstance(message_obj.content, list):
                    # Multimodal content
                    for item in message_obj.content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                content_blocks.append(TextContent(type="text", text=item.get("text", "")))
                            # Add other multimodal types as needed

            # Extract tool calls
            if hasattr(message_obj, "tool_calls") and message_obj.tool_calls:
                for tool_call in message_obj.tool_calls:
                    content_blocks.append(
                        ToolCallContent(
                            type="tool_call",
                            id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        )
                    )

            # Extract web search citations if present
            # Note: This is a placeholder for when OpenAI adds web search
            if hasattr(message_obj, "citations") and message_obj.citations:
                for citation in message_obj.citations:
                    content_blocks.append(
                        CitationContent(
                            type="citation",
                            url=citation.get("url", ""),
                            title=citation.get("title", ""),
                            snippet=citation.get("snippet", ""),
                            relevance_score=citation.get("relevance_score"),
                        )
                    )

            # Create unified message
            messages.append(
                UnifiedMessage(
                    role=message_obj.role or "assistant",
                    content=content_blocks,
                    name=getattr(message_obj, "name", None),
                )
            )

        # Extract usage information
        usage = {}
        if hasattr(openai_response, "usage") and openai_response.usage:
            usage = {
                "prompt_tokens": openai_response.usage.prompt_tokens,
                "completion_tokens": openai_response.usage.completion_tokens,
                "total_tokens": openai_response.usage.total_tokens,
            }

        # Build UnifiedResponse
        unified_response = UnifiedResponse(
            id=openai_response.id,
            model=openai_response.model,
            provider="openai",
            messages=messages,
            usage=usage,
            finish_reason=openai_response.choices[0].finish_reason if openai_response.choices else None,
            status="completed",
            provider_metadata={
                "created": getattr(openai_response, "created", None),
                "system_fingerprint": getattr(openai_response, "system_fingerprint", None),
                "service_tier": getattr(openai_response, "service_tier", None),
            },
        )

        # Calculate cost
        unified_response.cost = self.cost(unified_response)

        return unified_response

    def create_v1_compatible(self, params: dict[str, Any]) -> Any:
        """
        Create completion in backward-compatible ChatCompletionExtended format.

        This method provides compatibility with existing AG2 code that expects
        ChatCompletionExtended format. Note that reasoning blocks and citations
        will be lost in this format.

        Args:
            params: Same parameters as create()

        Returns:
            ChatCompletionExtended-compatible dict (flattened response)

        Warning:
            This method loses information (reasoning blocks, citations) when
            converting to the legacy format. Prefer create() for new code.
        """
        # Get rich response
        unified_response = self.create(params)

        # Convert to legacy format (simplified - would need full ChatCompletionExtended in practice)
        return {
            "id": unified_response.id,
            "model": unified_response.model,
            "created": unified_response.provider_metadata.get("created"),
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": unified_response.messages[0].role if unified_response.messages else "assistant",
                        "content": unified_response.text,
                    },
                    "finish_reason": unified_response.finish_reason,
                }
            ],
            "usage": unified_response.usage,
            "cost": unified_response.cost,
        }

    def cost(self, response: UnifiedResponse) -> float:  # type: ignore[override]
        """
        Calculate cost from response usage.

        Implements ModelClient.cost() but accepts UnifiedResponse via duck typing.

        Args:
            response: UnifiedResponse with usage information

        Returns:
            Cost in USD for the API call
        """
        if not response.usage:
            return 0.0

        model = response.model
        prompt_tokens = response.usage.get("prompt_tokens", 0)
        completion_tokens = response.usage.get("completion_tokens", 0)

        # Find pricing for model (exact match or prefix)
        pricing = None
        for model_key in self._cost_per_token:
            if model.startswith(model_key):
                pricing = self._cost_per_token[model_key]
                break

        if not pricing:
            # Unknown model - use default pricing (gpt-4 level)
            pricing = {"prompt": 0.03 / 1000, "completion": 0.06 / 1000}

        return (prompt_tokens * pricing["prompt"]) + (completion_tokens * pricing["completion"])

    @staticmethod
    def get_usage(response: UnifiedResponse) -> dict[str, Any]:  # type: ignore[override]
        """
        Extract usage statistics from response.

        Implements ModelClient.get_usage() but accepts UnifiedResponse via duck typing.

        Args:
            response: UnifiedResponse from create()

        Returns:
            Dict with keys from RESPONSE_USAGE_KEYS
        """
        return {
            "prompt_tokens": response.usage.get("prompt_tokens", 0),
            "completion_tokens": response.usage.get("completion_tokens", 0),
            "total_tokens": response.usage.get("total_tokens", 0),
            "cost": response.cost or 0.0,
            "model": response.model,
        }

    def message_retrieval(self, response: UnifiedResponse) -> list[str]:  # type: ignore[override]
        """
        Retrieve text content from response messages.

        Implements ModelClient.message_retrieval() but accepts UnifiedResponse via duck typing.

        Args:
            response: UnifiedResponse from create()

        Returns:
            List of text strings from message content blocks
        """
        return [msg.get_text() for msg in response.messages]
