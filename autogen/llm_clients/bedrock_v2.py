# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
AWS Bedrock Converse API Client implementing ModelClientV2 and ModelClient protocols.

This client handles the AWS Bedrock Converse API (bedrock_runtime.converse)
which returns rich responses with:
- Text content
- Image content (multimodal)
- Tool calls and function execution
- Structured outputs via response_format

The client preserves all provider-specific features in UnifiedResponse format
and is compatible with AG2's agent system through ModelClient protocol.
"""

from __future__ import annotations

from typing import Any, Literal

from autogen.import_utils import optional_import_block, require_optional_import
from autogen.llm_config.client import ModelClient
from autogen.llm_clients.models import (
    UnifiedMessage,
    UnifiedResponse,
    UserRoleEnum,
)

# Import Bedrock-specific utilities from existing client
from autogen.oai.bedrock import (
    format_tools,
    oai_messages_to_bedrock_messages,
)

with optional_import_block():
    import boto3
    from botocore.config import Config

if optional_import_block().is_successful:
    boto3_import_exception: ImportError | None = None
else:
    boto3_import_exception = ImportError(
        "Please install boto3 to use BedrockV2Client. Install with: pip install boto3"
    )


@require_optional_import("boto3", "bedrock")
class BedrockV2Client(ModelClient):
    """
    AWS Bedrock Converse API client implementing ModelClientV2 protocol.

    This client works with AWS Bedrock's Converse API (bedrock_runtime.converse)
    which returns structured output with tool calls, multimodal content, and more.

    Key Features:
    - Preserves text and image content as typed content blocks
    - Handles tool calls and structured outputs
    - Supports system prompts (model-dependent)
    - Provides backward compatibility via create_v1_compatible()
    - Supports additional model request fields for model-specific features
    """

    RESPONSE_USAGE_KEYS: list[str] = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    _retries = 5

    def __init__(
        self,
        aws_region: str | None = None,
        aws_access_key: str | None = None,
        aws_secret_key: str | None = None,
        aws_session_token: str | None = None,
        aws_profile_name: str | None = None,
        timeout: int | None = None,
        total_max_attempts: int = 5,
        max_attempts: int = 5,
        mode: Literal["standard", "adaptive", "legacy"] = "standard",
        response_format: Any = None,
        **kwargs: Any,
    ):
        """
        Initialize AWS Bedrock Converse API client.

        Args:
            aws_region: AWS region (required, or set AWS_REGION env var)
            aws_access_key: AWS access key (or set AWS_ACCESS_KEY env var)
            aws_secret_key: AWS secret key (or set AWS_SECRET_KEY env var)
            aws_session_token: AWS session token (or set AWS_SESSION_TOKEN env var)
            aws_profile_name: AWS profile name for credentials
            timeout: Request timeout in seconds (default: 60)
            total_max_attempts: Total max retry attempts (default: 5)
            max_attempts: Max attempts per retry (default: 5)
            mode: Retry mode - "standard", "adaptive", or "legacy" (default: "standard")
            response_format: Optional response format (Pydantic model or JSON schema) for structured outputs
            **kwargs: Additional arguments passed to boto3 client
        """
        raise NotImplementedError("Initialization not yet implemented")

    def create(self, params: dict[str, Any]) -> UnifiedResponse:  # type: ignore[override]
        """
        Create a completion and return UnifiedResponse with all features preserved.

        This method implements ModelClient.create() but returns UnifiedResponse instead
        of ModelClientResponseProtocol. The rich UnifiedResponse structure is compatible
        via duck typing - it has .model attribute and works with message_retrieval().

        Args:
            params: Request parameters including:
                - model: Model ID (e.g., "anthropic.claude-sonnet-4-5-20250929-v1:0")
                - messages: List of message dicts
                - temperature: Optional temperature
                - max_tokens: Optional max completion tokens
                - tools: Optional tool definitions
                - response_format: Optional Pydantic BaseModel or JSON schema dict
                - supports_system_prompts: Whether model supports system prompts (default: True)
                - price: Optional [input_price_per_1k, output_price_per_1k] for cost calculation
                - additional_model_request_fields: Optional model-specific fields
                - **other Bedrock parameters

        Returns:
            UnifiedResponse with text, images, tool calls, and all content preserved
        """
        raise NotImplementedError("create() not yet implemented")

    def create_v1_compatible(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Create completion in backward-compatible ChatCompletionExtended format.

        This method provides compatibility with existing AG2 code that expects
        ChatCompletionExtended format.

        Args:
            params: Same parameters as create()

        Returns:
            ChatCompletionExtended-compatible dict (flattened response)

        Warning:
            This method loses information (images, rich content) when converting
            to the legacy format. Prefer create() for new code.
        """
        raise NotImplementedError("create_v1_compatible() not yet implemented")

    def cost(self, response: UnifiedResponse) -> float:  # type: ignore[override]
        """
        Calculate cost from response usage.

        Implements ModelClient.cost() but accepts UnifiedResponse via duck typing.

        Args:
            response: UnifiedResponse with usage information

        Returns:
            Cost in USD for the API call
        """
        raise NotImplementedError("cost() not yet implemented")

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
        raise NotImplementedError("get_usage() not yet implemented")

    def message_retrieval(self, response: UnifiedResponse) -> list[str] | list[dict[str, Any]]:  # type: ignore[override]
        """
        Retrieve messages from response in OpenAI-compatible format.

        Returns list of strings for text-only messages, or list of dicts when
        tool calls or complex content is present.

        Args:
            response: UnifiedResponse from create()

        Returns:
            List of strings (for text-only) OR list of message dicts (for tool calls/complex content)
        """
        raise NotImplementedError("message_retrieval() not yet implemented")
