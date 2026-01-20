# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Google Gemini V2 Client implementing ModelClientV2 protocol.

This client handles Google's Gemini API which supports:
- Multimodal content (text, images, audio, video)
- Function/tool calling
- Structured outputs
- Thinking/reasoning tokens (Gemini 3 models)
- Grounding metadata and citations

The client preserves all provider-specific features in UnifiedResponse format
and is compatible with AG2's agent system through ModelClient protocol.
"""

from typing import Any, Literal

from pydantic import Field


from ..import_utils import optional_import_block
from ..llm_config.client import ModelClient
from ..llm_config.entry import LLMConfigEntry, LLMConfigEntryDict
from .models import (
    UnifiedResponse,
)

with optional_import_block() as gemini_result:
    import google.genai as genai
    import vertexai
    from google.auth.credentials import Credentials
    from google.genai import types
    from google.genai.types import (
        Content,
        FinishReason,
        FunctionCall,
        FunctionResponse,
        GenerateContentConfig,
        GenerateContentResponse,
        Part,
        ThinkingConfig,
        Tool,
    )
    from vertexai.generative_models import (
        Content as VertexAIContent,
        FunctionDeclaration as vaiFunctionDeclaration,
        GenerationConfig,
        GenerativeModel,
        GenerationResponse as VertexAIGenerationResponse,
        HarmBlockThreshold as VertexAIHarmBlockThreshold,
        HarmCategory as VertexAIHarmCategory,
        Part as VertexAIPart,
        SafetySetting as VertexAISafetySetting,
        Tool as vaiTool,
    )

if gemini_result.is_successful:
    gemini_import_exception: ImportError | None = None
else:
    gemini_import_exception = ImportError(
        "Please install google-genai and vertexai to use GeminiV2Client. Install with: pip install google-genai vertexai"
    )


class GeminiV2EntryDict(LLMConfigEntryDict, total=False):
    """Entry dict for Gemini V2 client configuration."""

    api_type: Literal["gemini_v2"]
    project_id: str | None
    location: str | None
    google_application_credentials: str | None
    credentials: Any | str | None
    stream: bool
    safety_settings: list[dict[str, Any]] | dict[str, Any] | None
    price: list[float] | None
    tool_config: dict[str, Any] | None
    proxy: str | None
    include_thoughts: bool | None
    thinking_budget: int | None
    thinking_level: Literal["High", "Medium", "Low", "Minimal"] | None


class GeminiV2LLMConfigEntry(LLMConfigEntry):
    """LLM config entry for Gemini V2 client."""

    api_type: Literal["gemini_v2"] = "gemini_v2"
    project_id: str | None = None
    location: str | None = None
    google_application_credentials: str | None = None
    credentials: Any | str | None = None
    stream: bool = False
    safety_settings: list[dict[str, Any]] | dict[str, Any] | None = None
    price: list[float] | None = Field(default=None, min_length=2, max_length=2)
    tool_config: dict[str, Any] | None = None
    proxy: str | None = None
    include_thoughts: bool | None = None
    thinking_budget: int | None = None
    thinking_level: Literal["High", "Medium", "Low", "Minimal"] | None = None

    def create_client(self) -> ModelClient:  # pragma: no cover
        """Create GeminiV2Client instance."""
        raise NotImplementedError("GeminiV2LLMConfigEntry.create_client() is not implemented.")


class GeminiV2Client(ModelClient):
    """
    Google Gemini V2 client implementing ModelClientV2 protocol.

    This client works with Google's Gemini API which supports multimodal content,
    function calling, structured outputs, and thinking tokens.

    Key Features:
    - Preserves multimodal content (text, images, audio, video)
    - Handles function/tool calls
    - Supports structured outputs via response_format
    - Extracts thinking/reasoning tokens (Gemini 3 models)
    - Preserves grounding metadata and citations

    Example:
        client = GeminiV2Client(api_key="...")

        # Get rich response with multimodal content
        response = client.create({
            "model": "gemini-2.5-pro",
            "messages": [{"role": "user", "content": "Explain quantum computing"}]
        })

        # Access text content
        print(f"Answer: {response.text}")

        # Access images if present
        images = response.get_content_by_type("image")
        for image in images:
            print(f"Image: {image.data_uri}")
    """

    RESPONSE_USAGE_KEYS: list[str] = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    # Mapping from AG2 parameter names to Gemini parameter names
    PARAMS_MAPPING = {
        "max_tokens": "max_output_tokens",
        "seed": "seed",
        "stop_sequences": "stop_sequences",
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "max_output_tokens": "max_output_tokens",
    }

    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        location: str | None = None,
        google_application_credentials: str | None = None,
        credentials: Any | None = None,
        proxy: str | None = None,
        api_version: str | None = None,
        response_format: Any = None,
        **kwargs: Any,
    ):
        """
        Initialize Google Gemini API client.

        Args:
            api_key: Gemini API key (or set GOOGLE_GEMINI_API_KEY env var)
            project_id: Google Cloud project ID (for Vertex AI)
            location: Google Cloud location (for Vertex AI)
            google_application_credentials: Path to service account JSON keyfile
            credentials: Google auth credentials object
            proxy: HTTP proxy URL
            api_version: API version override
            response_format: Optional response format (Pydantic model or JSON schema)
            **kwargs: Additional arguments
        """
        if gemini_import_exception is not None:
            raise gemini_import_exception

        # TODO: Implement initialization logic
        raise NotImplementedError("GeminiV2Client.__init__() not yet implemented")

    def create(self, params: dict[str, Any]) -> UnifiedResponse:  # type: ignore[override]
        """
        Create a completion and return UnifiedResponse with all features preserved.

        Args:
            params: Request parameters including:
                - model: Model name (e.g., "gemini-2.5-pro")
                - messages: List of message dicts
                - temperature: Optional temperature
                - max_tokens: Optional max completion tokens
                - tools: Optional tool definitions
                - response_format: Optional Pydantic BaseModel or JSON schema dict
                - safety_settings: Optional safety settings
                - include_thoughts: Optional thinking tokens flag
                - thinking_budget: Optional thinking budget in tokens
                - **other Gemini parameters

        Returns:
            UnifiedResponse with text, images, tool calls, and all content preserved
        """
        # TODO: Implement create method
        raise NotImplementedError("GeminiV2Client.create() not yet implemented")

    def create_v1_compatible(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Create a completion and return V1-compatible ChatCompletion format.

        Args:
            params: Request parameters (same as create())

        Returns:
            ChatCompletion-like dict for backward compatibility
        """
        # TODO: Implement create_v1_compatible method
        raise NotImplementedError("GeminiV2Client.create_v1_compatible() not yet implemented")

    def cost(self, response: UnifiedResponse) -> float:
        """
        Calculate cost from UnifiedResponse.

        Args:
            response: UnifiedResponse object

        Returns:
            Cost in USD
        """
        # TODO: Implement cost calculation
        raise NotImplementedError("GeminiV2Client.cost() not yet implemented")

    @staticmethod
    def get_usage(response: UnifiedResponse) -> dict[str, Any]:
        """
        Extract usage information from UnifiedResponse.

        Args:
            response: UnifiedResponse object

        Returns:
            Dictionary with usage keys from RESPONSE_USAGE_KEYS
        """
        # TODO: Implement get_usage method
        raise NotImplementedError("GeminiV2Client.get_usage() not yet implemented")

    def message_retrieval(self, response: UnifiedResponse) -> list[str]:  # type: ignore[override]
        """
        Extract text messages from UnifiedResponse for V1 compatibility.

        Args:
            response: UnifiedResponse object

        Returns:
            List of text strings from messages
        """
        # TODO: Implement message_retrieval method
        raise NotImplementedError("GeminiV2Client.message_retrieval() not yet implemented")
