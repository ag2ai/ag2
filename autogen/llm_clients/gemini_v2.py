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


from __future__ import annotations

import base64
import copy
import json
import os
import random
import time
import warnings
from typing import Any, Literal

from pydantic import Field, field_validator

from autogen.oai.shared_utils import normalize_pydantic_schema_to_dict

from ..import_utils import optional_import_block
from ..json_utils import resolve_json_references
from ..llm_config.client import ModelClient
from ..llm_config.entry import LLMConfigEntry, LLMConfigEntryDict
from .models import (
    AudioContent,
    GenericContent,
    ImageContent,
    ReasoningContent,
    TextContent,
    ToolCallContent,
    UnifiedMessage,
    UnifiedResponse,
    UserRoleEnum,
    VideoContent,
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
        GoogleSearch,
        Part,
        ThinkingConfig,
        Tool,
    )
    from vertexai.generative_models import (
        Content as VertexAIContent,
    )
    from vertexai.generative_models import (
        FunctionDeclaration as vaiFunctionDeclaration,
    )
    from vertexai.generative_models import (
        GenerationConfig,
        GenerativeModel,
    )
    from vertexai.generative_models import (
        GenerationResponse as VertexAIGenerationResponse,
    )
    from vertexai.generative_models import (
        HarmBlockThreshold as VertexAIHarmBlockThreshold,
    )
    from vertexai.generative_models import (
        HarmCategory as VertexAIHarmCategory,
    )
    from vertexai.generative_models import (
        Part as VertexAIPart,
    )
    from vertexai.generative_models import (
        SafetySetting as VertexAISafetySetting,
    )
    from vertexai.generative_models import (
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

    @field_validator("tool_config", mode="before")
    @classmethod
    def _coerce_tool_config(cls, v: Any) -> dict[str, Any] | None:
        """Accept ToolConfig (google.genai.types or autogen.oai.gemini_types) and coerce to dict."""
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        if hasattr(v, "model_dump"):
            return v.model_dump(exclude_none=True, mode="json")
        if hasattr(v, "dict"):  # Pydantic v1 fallback
            return v.dict(exclude_none=True)
        raise ValueError(f"tool_config must be dict or ToolConfig, got {type(v).__name__}")

    def create_client(self) -> ModelClient:  # pragma: no cover
        """Create GeminiV2Client instance."""
        GeminiV2Client = globals()["GeminiV2Client"]  # noqa: N806

        return GeminiV2Client(
            api_key=None,  # Will use env var or credentials
            project_id=self.project_id,
            location=self.location,
            google_application_credentials=self.google_application_credentials,
            credentials=self.credentials,
            proxy=self.proxy,
            response_format=None,
        )


@require_optional_import(["google", "vertexai", "PIL", "jsonschema"], "gemini")
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

        self.api_key = api_key or os.getenv("GOOGLE_GEMINI_API_KEY")
        self.project_id = project_id
        self.location = location
        self.google_application_credentials = google_application_credentials
        self.credentials = credentials
        self.proxy = proxy
        self.api_version = api_version
        self._response_format = response_format

        # Determine if using Vertex AI or API key
        if not self.api_key:
            self.use_vertexai = True
            self._initialize_vertexai()
        else:
            self.use_vertexai = False
            if self.project_id or self.location:
                raise ValueError("Google Cloud project and location cannot be set when using an API Key!")

        # Maps function call ids to function names
        self.tool_call_function_map: dict[str, str] = {}
        # Maps function call ids to thought signatures (for Gemini 3 models)
        self.tool_call_thought_signatures: dict[str, bytes] = {}

        # Store pricing for cost calculation
        self._price_per_1k_tokens: tuple[float, float] | None = None

    def _initialize_vertexai(self) -> None:
        """Initialize Vertex AI if using service account credentials."""
        if self.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.google_application_credentials

        vertexai_init_args = {}
        if self.project_id:
            vertexai_init_args["project"] = self.project_id
        if self.location:
            vertexai_init_args["location"] = self.location
        if self.credentials:
            if not isinstance(self.credentials, Credentials):
                raise TypeError("Object type google.auth.credentials.Credentials is expected!")
            vertexai_init_args["credentials"] = self.credentials

        if vertexai_init_args:
            vertexai.init(**vertexai_init_args)

    def _normalize_pydantic_schema_to_dict(
        self, schema: dict[str, Any] | type[BaseModel], for_genai_api: bool = False
    ) -> dict[str, Any]:
        """
        Convert a Pydantic model's JSON schema to a flat dict schema by resolving $ref references.

        Similar to bedrock.py's _normalize_pydantic_schema_to_dict, but also handles
        additionalProperties conversion for Gemini GenAI API compatibility.

        Args:
            schema: Either a Pydantic model class or a dict containing the JSON schema
            for_genai_api: If True, convert additionalProperties to regular properties
                          for Gemini GenAI API compatibility (not needed for Vertex AI)

        Returns:
            A normalized dict schema with all $ref references resolved, $defs removed,
            and additionalProperties converted to properties if for_genai_api is True
        """
        from pydantic import BaseModel

        # If it's a Pydantic model, get its JSON schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_dict = schema.model_json_schema()
        elif isinstance(schema, dict):
            schema_dict = schema.copy()
        else:
            raise ValueError(f"Schema must be a Pydantic model class or dict, got {type(schema)}")

        # Extract $defs if present
        defs = schema_dict.get("$defs", {}).copy()

        def resolve_ref(ref: str, definitions: dict[str, Any]) -> dict[str, Any]:
            """Resolve a $ref to its actual schema definition."""
            if not ref.startswith("#/$defs/"):
                raise ValueError(f"Unsupported $ref format: {ref}. Only '#/$defs/...' is supported.")
            # Extract the definition name from "#/$defs/Name"
            def_name = ref.split("/")[-1]
            if def_name not in definitions:
                raise ValueError(f"Definition '{def_name}' not found in $defs")
            return definitions[def_name].copy()

        def resolve_refs_recursive(obj: Any, definitions: dict[str, Any]) -> Any:
            """Recursively resolve all $ref references in the schema."""
            if isinstance(obj, dict):
                # If this dict has a $ref, replace it with the actual definition
                if "$ref" in obj:
                    ref_def = resolve_ref(obj["$ref"], definitions)
                    # Merge any additional properties from the current object (except $ref)
                    merged = {**ref_def, **{k: v for k, v in obj.items() if k != "$ref"}}
                    # Recursively resolve any refs in the merged definition
                    return resolve_refs_recursive(merged, definitions)
                else:
                    # Process all values recursively
                    return {k: resolve_refs_recursive(v, definitions) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_refs_recursive(item, definitions) for item in obj]
            else:
                return obj

        # Resolve all references
        normalized_schema = resolve_refs_recursive(schema_dict, defs)

        # Remove $defs section as it's no longer needed
        if "$defs" in normalized_schema:
            normalized_schema.pop("$defs")

        # Convert additionalProperties to regular properties for Gemini GenAI API
        if for_genai_api:

            def convert_additional_properties_to_properties(schema: dict) -> dict:
                """Recursively convert additionalProperties to regular properties.

                For objects with only additionalProperties (like dict[str, T]),
                we convert the additionalProperties value into a regular property
                to satisfy Gemini's requirement that objects must have non-empty properties.
                """
                if isinstance(schema, dict):
                    # Process nested schemas first
                    if "properties" in schema:
                        for prop_schema in schema["properties"].values():
                            convert_additional_properties_to_properties(prop_schema)
                    if "items" in schema:
                        convert_additional_properties_to_properties(schema["items"])
                    if "anyOf" in schema:
                        for any_of_schema in schema["anyOf"]:
                            convert_additional_properties_to_properties(any_of_schema)
                    if "oneOf" in schema:
                        for one_of_schema in schema["oneOf"]:
                            convert_additional_properties_to_properties(one_of_schema)
                    if "allOf" in schema:
                        for all_of_schema in schema["allOf"]:
                            convert_additional_properties_to_properties(all_of_schema)

                    # Convert additionalProperties to a regular property if object has no properties
                    if (
                        schema.get("type") == "object"
                        and "additionalProperties" in schema
                        and not schema.get("properties")
                    ):
                        additional_props_value = schema["additionalProperties"]
                        # Recursively process the value schema
                        if isinstance(additional_props_value, dict):
                            processed_value = convert_additional_properties_to_properties(additional_props_value)
                            # Convert to a regular property (preserving the type information)
                            schema["properties"] = {"value": processed_value}
                            # Remove additionalProperties since we've converted it
                            schema.pop("additionalProperties", None)
                    else:
                        # Remove additionalProperties if object already has properties
                        schema.pop("additionalProperties", None)

                return schema

            normalized_schema = convert_additional_properties_to_properties(normalized_schema)

        return normalized_schema

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
                - price: Optional [input_price_per_1k, output_price_per_1k] for cost calculation
                - **other Gemini parameters

        Returns:
            UnifiedResponse with text, images, tool calls, and all content preserved
        """
        params = params.copy()

        # Parse custom parameters
        self._parse_custom_params(params)

        # Get model name
        model_name = params.get("model", "gemini-pro")
        if not model_name:
            raise ValueError("Please provide a model name for the Gemini Client.")

        # Handle response format for structured outputs
        response_format = params.get("response_format") or self._response_format
        has_response_format = response_format is not None

        # Convert messages to Gemini format
        gemini_messages = self._oai_messages_to_gemini_messages(params.get("messages", []))

        # Extract system instruction
        system_instruction = self._extract_system_instruction(params.get("messages", []))

        # Convert tools to Gemini format
        tools = self._tools_to_gemini_tools(params["tools"]) if "tools" in params else None

        # Build generation config
        generation_config = {
            gemini_term: params[autogen_term]
            for autogen_term, gemini_term in self.PARAMS_MAPPING.items()
            if autogen_term in params
        }

        # Handle structured outputs
        if has_response_format:
            generation_config["response_mime_type"] = "application/json"
            # Normalize schema: resolve $ref references and handle additionalProperties
            # For GenAI API, convert additionalProperties to properties
            # For Vertex AI, keep as-is (may support additionalProperties)
            response_schema = normalize_pydantic_schema_to_dict(response_format, for_genai_api=not self.use_vertexai)
            generation_config["response_schema"] = response_schema

        # Handle safety settings
        if self.use_vertexai:
            safety_settings = self._to_vertexai_safety_settings(params.get("safety_settings", []))
        else:
            safety_settings = params.get("safety_settings", [])

        # Handle thinking config (Gemini 3 models)
        thinking_config = None
        if params.get("include_thoughts") is not None or params.get("thinking_budget") is not None:
            thinking_config = ThinkingConfig(
                include_thoughts=params.get("include_thoughts"),
                thinking_budget=params.get("thinking_budget"),
            )

        # Build HTTP options
        http_options = types.HttpOptions()
        if self.proxy:
            http_options.client_args = {"proxy": self.proxy}
            http_options.async_client_args = {"proxy": self.proxy}
        if self.api_version:
            http_options.api_version = self.api_version

        # Call Gemini API
        if self.use_vertexai:
            model = GenerativeModel(
                model_name,
                generation_config=GenerationConfig(**generation_config),
                safety_settings=safety_settings,
                system_instruction=system_instruction,
                tools=tools,
            )
            chat = model.start_chat(history=gemini_messages[:-1])
            response = chat.send_message(gemini_messages[-1].parts, safety_settings=safety_settings)
        else:
            client = genai.Client(api_key=self.api_key, http_options=http_options)
            generate_content_config = GenerateContentConfig(
                safety_settings=safety_settings,
                system_instruction=system_instruction,
                tools=tools,
                thinking_config=thinking_config if thinking_config is not None else None,
                **generation_config,
            )
            chat = client.chats.create(model=model_name, config=generate_content_config, history=gemini_messages[:-1])
            response = chat.send_message(message=gemini_messages[-1].parts)

        # Transform to UnifiedResponse
        return self._transform_response(response, model_name, has_response_format)

    def _transform_response(
        self,
        gemini_response: GenerateContentResponse | VertexAIGenerationResponse,
        model: str,
        has_response_format: bool = False,
    ) -> UnifiedResponse:
        """
        Transform Gemini API response to UnifiedResponse.

        Content handling:
        - Text parts → TextContent
        - Function calls → ToolCallContent
        - Inline data (images) → ImageContent
        - Thinking/reasoning → ReasoningContent (Gemini 3 models)
        - Unknown content types → GenericContent (forward compatibility)

        Args:
            gemini_response: Raw Gemini API response
            model: Model name
            has_response_format: Whether structured output was requested

        Returns:
            UnifiedResponse with all content blocks properly typed
        """
        # Extract candidate (Gemini always returns one candidate)
        # Guard isinstance checks to handle optional imports
        try:
            is_generate_content = isinstance(gemini_response, GenerateContentResponse)
        except NameError:
            is_generate_content = type(gemini_response).__name__ == "GenerateContentResponse"

        try:
            is_vertexai = isinstance(gemini_response, VertexAIGenerationResponse)
        except NameError:
            is_vertexai = type(gemini_response).__name__ == "GenerationResponse"

        if is_generate_content or is_vertexai:
            if len(gemini_response.candidates) != 1:
                raise ValueError(f"Unexpected number of candidates. Expected 1, got {len(gemini_response.candidates)}")
            candidate = gemini_response.candidates[0]
            parts = candidate.content.parts if candidate.content and candidate.content.parts else []
            finish_reason = self._convert_finish_reason(candidate.finish_reason)
            usage_metadata = gemini_response.usage_metadata
        else:
            raise ValueError(f"Unexpected response type: {type(gemini_response)}")

        # Build content blocks from parts
        content_blocks = []
        structured_text = None

        for part in parts:
            # Text content
            if hasattr(part, "text") and part.text:
                if has_response_format and finish_reason == "stop":
                    # Structured output - parse JSON
                    try:
                        structured_text = json.loads(part.text)
                    except json.JSONDecodeError:
                        structured_text = part.text
                else:
                    content_blocks.append(TextContent(text=part.text))

            # Function calls (tool calls)
            if hasattr(part, "function_call") and part.function_call:
                fn_call = part.function_call
                tool_call_id = f"call-{random.randint(0, 10000)}"
                self.tool_call_function_map[tool_call_id] = fn_call.name

                # Store thought signature if present (Gemini 3 models)
                if hasattr(part, "thought_signature") and part.thought_signature:
                    self.tool_call_thought_signatures[tool_call_id] = part.thought_signature

                # Safely convert args to dict and then to JSON string
                # Handle different types of args objects (dict, Mapping, etc.)
                if fn_call.args:
                    try:
                        # Try dict() conversion first (works for dict, Mapping, etc.)
                        if hasattr(fn_call.args, "items"):
                            args_dict = dict(fn_call.args.items())
                        elif isinstance(fn_call.args, dict):
                            args_dict = fn_call.args
                        else:
                            # Fallback: try direct conversion
                            args_dict = dict(fn_call.args) if fn_call.args else {}
                        arguments = json.dumps(args_dict)
                    except (TypeError, ValueError, AttributeError):
                        # If all else fails, use empty dict
                        arguments = "{}"
                else:
                    arguments = "{}"

                content_blocks.append(
                    ToolCallContent(
                        id=tool_call_id,
                        name=fn_call.name,
                        arguments=arguments,
                    )
                )

            # Inline data (images, audio, video)
            if hasattr(part, "inline_data") and part.inline_data:
                inline_data = part.inline_data
                mime_type = inline_data.mime_type
                data = inline_data.data

                if mime_type.startswith("image/"):
                    data_uri = f"data:{mime_type};base64,{data}"
                    content_blocks.append(ImageContent(data_uri=data_uri))
                elif mime_type.startswith("audio/"):
                    data_uri = f"data:{mime_type};base64,{data}"
                    content_blocks.append(AudioContent(data_uri=data_uri))
                elif mime_type.startswith("video/"):
                    data_uri = f"data:{mime_type};base64,{data}"
                    content_blocks.append(VideoContent(data_uri=data_uri))
                else:
                    # Unknown media type - use GenericContent
                    content_blocks.append(
                        GenericContent(
                            type="media",
                            mime_type=mime_type,
                            data=data,
                        )
                    )

            # Thinking/reasoning (Gemini 3 models)
            # Note: Gemini 3 thinking is typically in function_call parts with thought_signature
            # We extract it separately if available
            if hasattr(part, "thought") and part.thought:
                content_blocks.append(ReasoningContent(reasoning=part.thought, summary=None))

        # If structured output, add it as text
        if structured_text and has_response_format:
            if isinstance(structured_text, dict):
                formatted_text = json.dumps(structured_text, indent=2)
            else:
                formatted_text = str(structured_text)
            content_blocks.insert(0, TextContent(text=formatted_text))

        # Create unified message
        messages = [
            UnifiedMessage(
                role=UserRoleEnum.ASSISTANT,
                content=content_blocks if content_blocks else [TextContent(text="")],
            )
        ]

        # Extract usage
        usage = {
            "prompt_tokens": usage_metadata.prompt_token_count,
            "completion_tokens": usage_metadata.candidates_token_count or 0,
            "total_tokens": usage_metadata.prompt_token_count + (usage_metadata.candidates_token_count or 0),
        }

        # Build UnifiedResponse
        unified_response = UnifiedResponse(
            id=f"gemini-{random.randint(0, 10000)}",
            model=model,
            provider="gemini",
            messages=messages,
            usage=usage,
            finish_reason=finish_reason,
            status="completed",
            provider_metadata={
                "finish_reason_raw": str(candidate.finish_reason) if hasattr(candidate, "finish_reason") else None,
            },
        )

        # Calculate cost
        unified_response.cost = self.cost(unified_response)

        return unified_response

    def _convert_finish_reason(self, finish_reason: FinishReason| Any | None) -> str:
        """Convert Gemini finish reason to standard finish reason."""
        if finish_reason is None:
            return "stop"

        # Handle both FinishReason enum and string inputs
        if isinstance(finish_reason, FinishReason):
            # Use enum.name property for type-safe access to enum value name
            finish_reason_str = finish_reason.name
        elif isinstance(finish_reason, str):
            finish_reason_str = finish_reason
        else:
            raise ValueError(f"Unexpected finish reason type: {type(finish_reason)}")

        mapping = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
            "OTHER": "stop",
        }

        return mapping.get(finish_reason_str.upper(), "stop")

    def _extract_system_instruction(self, messages: list[dict[str, Any]]) -> str | None:
        """Extract system instruction from messages."""
        if not messages or messages[0].get("role") != "system":
            return None

        message = messages[0]
        content = message["content"]

        # Handle multimodal content (list of dicts)
        if isinstance(content, list):
            content = content[0].get("text", "").strip() if content else ""
        else:
            content = content.strip() if content else ""

        return content if len(content) > 0 else None

    def _oai_messages_to_gemini_messages(self, messages: list[dict[str, Any]]) -> list[Content | VertexAIContent]:
        """Convert OAI format messages to Gemini format."""
        # Import helper from gemini.py or implement here
        # For now, use a simplified version
        rst = []
        for message in messages:
            parts, part_type = self._oai_content_to_gemini_content(message)
            role = "user" if message["role"] in ["user", "system"] else "model"

            if self.use_vertexai:
                rst.append(VertexAIContent(parts=parts, role=role))
            else:
                rst.append(Content(parts=parts, role=role))

        # Ensure first and last messages are from user
        if rst and rst[0].role != "user":
            text_part = VertexAIPart.from_text("start chat") if self.use_vertexai else Part(text="start chat")
            rst.insert(
                0,
                VertexAIContent(parts=[text_part], role="user")
                if self.use_vertexai
                else Content(parts=[text_part], role="user"),
            )

        if rst and rst[-1].role != "user":
            text_part = VertexAIPart.from_text("continue") if self.use_vertexai else Part(text="continue")
            rst.append(
                VertexAIContent(parts=[text_part], role="user")
                if self.use_vertexai
                else Content(parts=[text_part], role="user")
            )

        return rst

    def _oai_content_to_gemini_content(self, message: dict[str, Any]) -> tuple[list[Part | VertexAIPart], str]:
        """Convert OAI content to Gemini parts."""
        parts = []

        # Handle tool responses
        if message.get("role") == "tool" and "tool_call_id" in message:
            function_name = self.tool_call_function_map.get(message["tool_call_id"], "unknown")
            content = message["content"]
            if isinstance(content, str):
                with contextlib.suppress(json.JSONDecodeError):
                    content = json.loads(content)
                except json.JSONDecodeError:
                    warnings.warn(
                        f"Tool response content for function '{function_name}' is not valid JSON. "
                        f"Sending as string wrapped in {{'result': content}} format. "
                        f"Content preview: {content[:100] if len(content) > 100 else content}",
                        UserWarning,
                        stacklevel=2,
                    )

            if self.use_vertexai:
                parts.append(VertexAIPart.from_function_response(name=function_name, response={"result": content}))
            else:
                parts.append(Part(function_response=FunctionResponse(name=function_name, response={"result": content})))

            return parts, "tool"

        # Handle tool calls
        if "tool_calls" in message and message["tool_calls"]:
            for tool_call in message["tool_calls"]:
                function_id = tool_call["id"]
                function_name = tool_call["function"]["name"]
                self.tool_call_function_map[function_id] = function_name

                args = json.loads(tool_call["function"]["arguments"])

                if self.use_vertexai:
                    # Vertex AI supports thoughtSignature at the Part level (not inside FunctionCall)
                    # Include it when available for Gemini 3 models
                    thought_sig = self.tool_call_thought_signatures.get(function_id)
                    function_call_dict = {"functionCall": {"name": function_name, "args": args}}
                    # Include thoughtSignature as base64-encoded string if available
                    if thought_sig:
                        function_call_dict["thoughtSignature"] = base64.b64encode(thought_sig).decode("utf-8")
                    parts.append(VertexAIPart.from_dict(function_call_dict))
                else:
                    thought_sig = self.tool_call_thought_signatures.get(function_id)
                    parts.append(
                        Part(
                            function_call=FunctionCall(name=function_name, args=args),
                            thought_signature=thought_sig,
                        )
                    )

            return parts, "tool_call"

        # Handle text content
        if isinstance(message.get("content"), str):
            content = message["content"]
            if content == "":
                content = "empty"  # Empty content not allowed

            if self.use_vertexai:
                parts.append(VertexAIPart.from_text(content))
            else:
                parts.append(Part(text=content))

            return parts, "text"

        # Handle multimodal content (list)
        if isinstance(message.get("content"), list):
            has_image = False
            for msg in message["content"]:
                if isinstance(msg, dict):
                    if msg.get("type") == "text":
                        text = msg.get("text", "")
                        if self.use_vertexai:
                            parts.append(VertexAIPart.from_text(text))
                        else:
                            parts.append(Part(text=text))
                    elif msg.get("type") == "image_url":
                        img_url = msg["image_url"]["url"]
                        if self.use_vertexai:
                            parts.append(VertexAIPart.from_uri(img_url, mime_type="image/png"))
                        else:
                            # Extract base64 data from data URI
                            if img_url.startswith("data:image"):
                                b64_data = img_url.split(",", 1)[1]
                                parts.append(Part(inline_data={"mime_type": "image/png", "data": b64_data}))
                            else:
                                # URL - would need to fetch, simplified for now
                                parts.append(Part(inline_data={"mime_type": "image/png", "data": ""}))

                        has_image = True

            return parts, "image" if has_image else "text"

        raise ValueError(f"Unable to convert content to Gemini format: {message}")

    def _tools_to_gemini_tools(self, tools: list[dict[str, Any]]) -> list[Tool | vaiTool]:
        """Convert OAI tools to Gemini tools format."""
        # Check for prebuilt Google Search tool (only for GenAI API, not Vertex AI)
        if self._check_if_prebuilt_google_search_tool_exists(tools) and not self.use_vertexai:
            return [Tool(google_search=GoogleSearch())]

        functions = []
        for tool in tools:
            if self.use_vertexai:
                function_parameters = copy.deepcopy(tool["function"]["parameters"])
                function_parameters = self._convert_type_null_to_nullable(function_parameters)
                function_parameters = self._unwrap_references(function_parameters)

                function = vaiFunctionDeclaration(
                    name=tool["function"]["name"],
                    description=tool["function"]["description"],
                    parameters=function_parameters,
                )
            else:
                function = self._create_gemini_function_declaration(tool)

            functions.append(function)

        if self.use_vertexai:
            return [vaiTool(function_declarations=functions)]
        else:
            return [Tool(function_declarations=functions)]

    def _create_gemini_function_declaration(self, tool: dict[str, Any]) -> Any:
        """Create Gemini function declaration from tool dict."""
        # Simplified - would need full schema conversion
        from google.genai.types import FunctionDeclaration

        function_declaration = FunctionDeclaration()
        function_declaration.name = tool["function"]["name"]
        function_declaration.description = tool["function"]["description"]

        params = tool["function"]["parameters"]
        if "properties" in params and len(params["properties"]) > 0:
            # Convert JSON schema to Gemini Schema
            function_declaration.parameters = self._create_gemini_schema(params)

        return function_declaration

    def _create_gemini_schema(self, json_schema: dict[str, Any]) -> Any:
        """Convert JSON schema to Gemini Schema object."""
        from google.genai.types import Schema, Type

        # Resolve references
        json_schema = resolve_json_references(json_schema)
        if "$defs" in json_schema:
            json_schema = copy.deepcopy(json_schema)
            json_schema.pop("$defs")

        schema = Schema()
        if "type" not in json_schema:
            schema.type = Type.STRING
            return schema

        type_mapping = {
            "integer": Type.INTEGER,
            "number": Type.NUMBER,
            "string": Type.STRING,
            "boolean": Type.BOOLEAN,
            "array": Type.ARRAY,
            "object": Type.OBJECT,
        }

        schema.type = type_mapping.get(json_schema["type"], Type.STRING)

        if json_schema["type"] == "object" and "properties" in json_schema:
            schema.properties = {}
            for prop_name, prop_data in json_schema["properties"].items():
                schema.properties[prop_name] = self._create_gemini_schema(prop_data)

        if json_schema["type"] == "array" and "items" in json_schema:
            schema.items = self._create_gemini_schema(json_schema["items"])

        if "description" in json_schema:
            schema.description = json_schema["description"]

        if "required" in json_schema:
            schema.required = json_schema["required"]

        if "enum" in json_schema:
            schema.enum = json_schema["enum"]

        return schema

    @staticmethod
    def _convert_type_null_to_nullable(schema: Any) -> Any:
        """Convert {"type": "null"} to {"nullable": True}."""
        if isinstance(schema, dict):
            if schema == {"type": "null"}:
                return {"nullable": True}
            return {key: GeminiV2Client._convert_type_null_to_nullable(value) for key, value in schema.items()}
        elif isinstance(schema, list):
            return [GeminiV2Client._convert_type_null_to_nullable(item) for item in schema]
        return schema

    @staticmethod
    def _unwrap_references(function_parameters: dict[str, Any]) -> dict[str, Any]:
        """Unwrap $ref references in function parameters."""
        if "properties" not in function_parameters:
            return function_parameters

        function_parameters_copy = copy.deepcopy(function_parameters)
        for property_name, property_value in function_parameters["properties"].items():
            if "$defs" in property_value:
                function_parameters_copy["properties"][property_name] = resolve_json_references(property_value)
                function_parameters_copy["properties"][property_name].pop("$defs", None)

        return function_parameters_copy

    @staticmethod
    def _check_if_prebuilt_google_search_tool_exists(tools: list[dict[str, Any]]) -> bool:
        """Check if the Google Search tool is present in the tools list."""
        exists = False
        for tool in tools:
            if tool["function"]["name"] == "prebuilt_google_search":
                exists = True
                break

        if exists and len(tools) > 1:
            raise ValueError(
                "Google Search tool can be used only by itself. Please remove other tools from the tools list."
            )

        return exists

    @staticmethod
    def _to_vertexai_safety_settings(safety_settings: list[dict[str, Any]] | None) -> list[Any]:
        """Convert safety settings to VertexAI format."""
        if not isinstance(safety_settings, list):
            return safety_settings or []

        vertexai_safety_settings = []
        for safety_setting in safety_settings:
            if (
                isinstance(safety_setting, dict)
                and safety_setting["category"] in VertexAIHarmCategory.__members__
                and safety_setting["threshold"] in VertexAIHarmBlockThreshold.__members__
            ):
                vertexai_safety_settings.append(
                    VertexAISafetySetting(
                        category=safety_setting["category"],
                        threshold=safety_setting["threshold"],
                    )
                )

        return vertexai_safety_settings

    def _parse_custom_params(self, params: dict[str, Any]) -> None:
        """Parse custom parameters for this client."""
        if "price" in params and isinstance(params["price"], list) and len(params["price"]) == 2:
            self._price_per_1k_tokens = (params["price"][0], params["price"][1])

    def create_v1_compatible(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Create a completion and return V1-compatible ChatCompletion format.

        Args:
            params: Request parameters (same as create())

        Returns:
            ChatCompletion-like dict for backward compatibility
        """
        response = self.create(params)

        # Convert UnifiedResponse to ChatCompletion format
        choices = []
        for idx, message in enumerate(response.messages):
            # Extract text content
            text_content = message.get_text()

            # Extract tool calls
            tool_calls = None
            tool_call_objects = message.get_tool_calls()
            if tool_call_objects:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in tool_call_objects
                ]

            choices.append({
                "index": idx,
                "message": {
                    "role": message.role.value if hasattr(message.role, "value") else str(message.role),
                    "content": text_content,
                    "tool_calls": tool_calls,
                },
                "finish_reason": response.finish_reason or "stop",
            })

        return {
            "id": response.id,
            "model": response.model,
            "created": int(time.time()),
            "object": "chat.completion",
            "choices": choices,
            "usage": response.usage,
            "cost": response.cost,
        }

    def cost(self, response: UnifiedResponse) -> float:
        """
        Calculate cost from UnifiedResponse.

        Args:
            response: UnifiedResponse object

        Returns:
            Cost in USD
        """
        # Use custom price if set
        if self._price_per_1k_tokens:
            input_cost = (response.usage.get("prompt_tokens", 0) / 1000) * self._price_per_1k_tokens[0]
            output_cost = (response.usage.get("completion_tokens", 0) / 1000) * self._price_per_1k_tokens[1]
            return input_cost + output_cost

        # Use default Gemini pricing (import from gemini.py)
        # Use optional_import_block to safely import calculate_gemini_cost
        # since gemini.py module may have optional dependencies
        with optional_import_block() as gemini_cost_result:
            from ..oai.gemini import calculate_gemini_cost

        if gemini_cost_result.is_successful:
            return calculate_gemini_cost(
                self.use_vertexai,
                response.usage.get("prompt_tokens", 0),
                response.usage.get("completion_tokens", 0),
                response.model,
            )
        else:
            # Fallback: return 0.0 if calculate_gemini_cost cannot be imported
            # This can happen if gemini.py optional dependencies are not available
            return 0.0

    @staticmethod
    def get_usage(response: UnifiedResponse) -> dict[str, Any]:
        """
        Extract usage information from UnifiedResponse.

        Args:
            response: UnifiedResponse object

        Returns:
            Dictionary with usage keys from RESPONSE_USAGE_KEYS
        """
        return {
            "prompt_tokens": response.usage.get("prompt_tokens", 0),
            "completion_tokens": response.usage.get("completion_tokens", 0),
            "total_tokens": response.usage.get("total_tokens", 0),
            "cost": response.cost or 0.0,
            "model": response.model,
        }

    def message_retrieval(self, response: UnifiedResponse) -> list[str] | list[dict[str, Any]]:  # type: ignore[override]
        """
        Extract text messages or OpenAI-style message dicts from UnifiedResponse for V1 compatibility.

        Returns message dicts with tool_calls when present so _Group_Tool_Executor can execute tools.

        Args:
            response: UnifiedResponse object

        Returns:
            List of text strings or message dicts (with tool_calls when present)
        """
        if not response.messages:
            return []
        result: list[str] | list[dict[str, Any]] = []
        for msg in response.messages:
            tool_calls_list = msg.get_tool_calls()
            if tool_calls_list:
                result.append({
                    "role": "assistant",
                    "content": msg.get_text() or None,
                    "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                        for tc in tool_calls_list
                    ],
                })
            else:
                text = msg.get_text()
                result.append({"role": "assistant", "content": text or ""})
        return result
