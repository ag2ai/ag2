# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Anthropic Messages API Client implementing ModelClientV2 and ModelClient protocols.

This client handles the Anthropic Messages API (client.messages.create) which returns
rich responses with:
- Thinking blocks (extended thinking feature)
- Tool calls and function execution
- Native structured outputs (beta API)
- JSON Mode structured outputs (fallback)
- Standard chat messages

The client preserves all provider-specific features in UnifiedResponse format
and is compatible with AG2's agent system through ModelClient protocol.

Note: This uses the Messages API, supporting both standard and beta structured outputs.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import warnings
from typing import Any, Literal

from pydantic import BaseModel

from autogen.import_utils import optional_import_block

logger = logging.getLogger(__name__)

# Import Anthropic SDK with optional import handling
with optional_import_block() as anthropic_result:
    from anthropic import Anthropic, BadRequestError
    from anthropic.types import Message

    # Beta imports for structured outputs
    try:
        from anthropic import transform_schema
    except ImportError:
        transform_schema = None  # type: ignore[misc, assignment]


if anthropic_result.is_successful:
    anthropic_import_exception: ImportError | None = None
else:
    Anthropic = None  # type: ignore[assignment]
    BadRequestError = None  # type: ignore[assignment]
    Message = None  # type: ignore[assignment]
    anthropic_import_exception = ImportError(
        "Please install anthropic to use AnthropicCompletionsClient. Install with: pip install anthropic"
    )

# Import helper functions and constants from existing anthropic.py
from autogen.oai.anthropic import (
    AnthropicEntryDict,
    AnthropicLLMConfigEntry,
    _calculate_cost,
    _is_text_block,
    _is_thinking_block,
    _is_tool_use_block,
    has_beta_messages_api,
    oai_messages_to_anthropic_messages,
    supports_native_structured_outputs,
    transform_schema_for_anthropic,
    validate_structured_outputs_version,
)
from autogen.oai.client_utils import FormatterProtocol, validate_parameter

# Import for backward compatibility
from autogen.oai.oai_models import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Choice,
    CompletionUsage,
)

# Import ModelClient protocol
from ..llm_config.client import ModelClient

# Import UnifiedResponse models
from .models import (
    AudioContent,
    GenericContent,
    ImageContent,
    ReasoningContent,
    TextContent,
    ToolCallContent,
    UnifiedMessage,
    UnifiedResponse,
    VideoContent,
    normalize_role,
)


class AnthropicV2LLMConfigDict(AnthropicEntryDict, total=False):
    api_type: Literal["anthropic_v2"]


class AnthropicV2LLMConfigEntry(AnthropicLLMConfigEntry):
    """
    LLMConfig entry for Anthropic V2 Client with ModelClientV2 architecture.

    This uses the new AnthropicV2Client from autogen.llm_clients which returns
    rich UnifiedResponse objects with typed content blocks (ReasoningContent,
    CitationContent, ToolCallContent, etc.).
    """

    api_type: Literal["anthropic_v2"] = "anthropic_v2"


class AnthropicV2Client(ModelClient):
    """
    Anthropic Messages API client implementing ModelClientV2 protocol.

    This client works with Anthropic's Messages API (client.messages.create)
    which returns structured output with thinking blocks, tool calls, and more.

    Key Features:
    - Preserves thinking blocks as ReasoningContent (extended thinking feature)
    - Handles tool calls and results
    - Supports native structured outputs (beta API) and JSON Mode fallback
    - Provides backward compatibility via create_v1_compatible()
    - Supports multiple authentication methods (API key, AWS Bedrock, GCP Vertex)

    Example:
        client = AnthropicCompletionsClient(api_key="...")

        # Get rich response with thinking
        response = client.create({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Explain quantum computing"}]
        })

        # Access thinking blocks
        for reasoning in response.reasoning:
            print(f"Thinking: {reasoning.reasoning}")

        # Get text response
        print(f"Answer: {response.text}")
    """

    RESPONSE_USAGE_KEYS: list[str] = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
        response_format: type[BaseModel] | dict | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Anthropic Messages API client.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            base_url: Optional base URL for the API
            timeout: Optional timeout in seconds
            response_format: Optional response format for structured outputs
            **kwargs: Additional arguments passed to Anthropic client
        """
        if anthropic_import_exception is not None:
            raise anthropic_import_exception

        # Store credentials
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        # Validate credentials
        if self._api_key is None:
            raise ValueError(
                "API key is required to use the Anthropic API. Set api_key parameter or ANTHROPIC_API_KEY environment variable."
            )

        # Initialize Anthropic client
        client_kwargs = {"api_key": self._api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if timeout:
            client_kwargs["timeout"] = timeout
        self._client = Anthropic(**client_kwargs, **kwargs)  # type: ignore[misc]

        # Store response format for structured outputs
        self._response_format: type[BaseModel] | dict | None = response_format

    def create(self, params: dict[str, Any]) -> UnifiedResponse:  # type: ignore[override]
        """
        Create a completion and return UnifiedResponse with all features preserved.

        This method implements ModelClient.create() but returns UnifiedResponse instead
        of ModelClientResponseProtocol. The rich UnifiedResponse structure is compatible
        via duck typing - it has .model attribute and works with message_retrieval().

        Automatically selects the best structured output method:
        - Native structured outputs for Claude Sonnet 4.5+ (guaranteed schema compliance)
        - JSON Mode for older models (prompt-based with <json_response> tags)
        - Standard completion for requests without response_format

        Args:
            params: Request parameters including:
                - model: Model name (e.g., "claude-3-5-sonnet-20241022")
                - messages: List of message dicts
                - temperature: Optional temperature
                - max_tokens: Optional max completion tokens
                - tools: Optional tool definitions
                - response_format: Optional Pydantic BaseModel or JSON schema dict
                - **other Anthropic parameters

        Returns:
            UnifiedResponse with thinking blocks, tool calls, and all content preserved
        """
        model = params.get("model")
        response_format = params.get("response_format") or self._response_format

        # Route to appropriate implementation based on model and response_format
        if response_format:
            self._response_format = response_format
            params["response_format"] = response_format

            # Try native structured outputs if model supports it
            if supports_native_structured_outputs(model) and has_beta_messages_api():
                try:
                    return self._create_with_native_structured_output(params)
                except (BadRequestError, AttributeError, ValueError) as e:  # type: ignore[misc]
                    # Fallback to JSON Mode if native API not supported or schema invalid
                    self._log_structured_output_fallback(e, model, response_format, params)
                    return self._create_with_json_mode(params)
            else:
                # Use JSON Mode for older models or when beta API unavailable
                return self._create_with_json_mode(params)
        else:
            # Standard completion without structured outputs
            return self._create_standard(params)

    def _create_standard(self, params: dict[str, Any]) -> UnifiedResponse:
        """
        Create a standard completion without structured outputs.

        Args:
            params: Request parameters

        Returns:
            UnifiedResponse with all content blocks properly typed
        """
        # Convert OAI messages to Anthropic format
        anthropic_messages = oai_messages_to_anthropic_messages(params)

        # Prepare Anthropic API parameters using helper (handles tool conversion, None removal, etc.)
        anthropic_params = self._prepare_anthropic_params(params, anthropic_messages)

        # Check if any tools use strict mode (requires beta API)
        has_strict_tools = any(tool.get("strict") for tool in anthropic_params.get("tools", []))

        if has_strict_tools:
            # Validate SDK version supports structured outputs beta
            validate_structured_outputs_version()
            # Use beta API for strict tools
            anthropic_params["betas"] = ["structured-outputs-2025-11-13"]
            response = self._client.beta.messages.create(**anthropic_params)  # type: ignore[misc]
        else:
            # Standard API for legacy tools
            response = self._client.messages.create(**anthropic_params)  # type: ignore[misc]

        # Transform to UnifiedResponse
        return self._transform_response(response, anthropic_params["model"], anthropic_params)

    def _create_with_native_structured_output(self, params: dict[str, Any]) -> UnifiedResponse:
        """
        Create completion using native structured outputs (beta API).

        This method uses Anthropic's beta structured outputs feature for guaranteed
        schema compliance via constrained decoding.

        Args:
            params: Request parameters

        Returns:
            UnifiedResponse with structured JSON output

        Raises:
            AttributeError: If SDK doesn't support beta API
            Exception: If native structured output fails
        """
        # Check if Anthropic's transform_schema is available
        if transform_schema is None:
            raise ImportError("Anthropic transform_schema not available. Please upgrade to anthropic>=0.74.1")

        # Get schema from response_format and transform it using Anthropic's function
        if isinstance(self._response_format, type) and issubclass(self._response_format, BaseModel):
            # For Pydantic models, use Anthropic's transform_schema directly
            transformed_schema = transform_schema(self._response_format)
        elif isinstance(self._response_format, dict):
            # For dict schemas, use as-is (already in correct format)
            schema = self._response_format
            # Still apply our transformation for additionalProperties
            transformed_schema = transform_schema_for_anthropic(schema)
        else:
            raise ValueError(f"Invalid response format: {self._response_format}")

        # Convert AG2 messages to Anthropic messages
        anthropic_messages = oai_messages_to_anthropic_messages(params)

        # Prepare Anthropic API parameters using helper
        anthropic_params = self._prepare_anthropic_params(params, anthropic_messages)

        # Validate SDK version supports structured outputs beta
        validate_structured_outputs_version()

        # Add native structured output parameters
        anthropic_params["betas"] = ["structured-outputs-2025-11-13"]

        # Use beta API
        if not hasattr(self._client, "beta"):
            raise AttributeError(
                "Anthropic SDK does not support beta.messages API. Please upgrade to anthropic>=0.39.0"
            )

        # When both tools and structured output are configured, must use create() (not parse())
        # parse() doesn't support tools, so we convert Pydantic models to dict schemas
        has_tools = "tools" in anthropic_params and anthropic_params["tools"]

        if has_tools or isinstance(self._response_format, dict):
            # Use create() with output_format for:
            # 1. Dict schemas (always)
            # 2. Pydantic models when tools are present (parse() doesn't support tools)
            anthropic_params["output_format"] = {
                "type": "json_schema",
                "schema": transformed_schema,
            }
            response = self._client.beta.messages.create(**anthropic_params)  # type: ignore[misc]
        else:
            # Pydantic model without tools - use parse() for automatic validation
            # parse() provides parsed_output attribute for direct model access
            anthropic_params["output_format"] = self._response_format
            response = self._client.beta.messages.parse(**anthropic_params)  # type: ignore[misc]

        # Transform to UnifiedResponse with is_native_structured_output=True
        return self._transform_response(
            response, anthropic_params["model"], anthropic_params, is_native_structured_output=True
        )

    def _create_with_json_mode(self, params: dict[str, Any]) -> UnifiedResponse:
        """
        Create completion using legacy JSON Mode with <json_response> tags.

        This method uses prompt-based structured outputs for older Claude models
        that don't support native structured outputs.

        Args:
            params: Request parameters

        Returns:
            UnifiedResponse with JSON output extracted from tags
        """
        # Add response format instructions to system message before message conversion
        self._add_response_format_to_system(params)

        # Convert AG2 messages to Anthropic messages
        anthropic_messages = oai_messages_to_anthropic_messages(params)

        # Prepare Anthropic API parameters using helper
        anthropic_params = self._prepare_anthropic_params(params, anthropic_messages)

        # Call Anthropic API
        response = self._client.messages.create(**anthropic_params)  # type: ignore[misc]

        # Extract JSON from <json_response> tags
        parsed_response = self._extract_json_response(response)

        # Transform to UnifiedResponse
        unified_response = self._transform_response(response, anthropic_params["model"], anthropic_params)

        # Replace text content with parsed JSON if structured output
        if self._response_format:
            # Find and replace TextContent with parsed JSON
            for msg in unified_response.messages:
                for i, block in enumerate(msg.content):
                    if isinstance(block, TextContent):
                        # Replace with parsed JSON text
                        json_text = (
                            parsed_response.model_dump_json()
                            if hasattr(parsed_response, "model_dump_json")
                            else str(parsed_response)
                        )
                        msg.content[i] = TextContent(type="text", text=json_text)
                        break

        return unified_response

    def _transform_response(
        self,
        anthropic_response: Message,  # type: ignore[valid-type]
        model: str,
        anthropic_params: dict[str, Any],
        is_native_structured_output: bool = False,
    ) -> UnifiedResponse:
        """
        Transform Anthropic Message response to UnifiedResponse.

        Handles all Anthropic content types:
        - Text blocks → TextContent
        - Thinking blocks → ReasoningContent
        - Tool use blocks → ToolCallContent
        - Structured outputs (parsed_output) → GenericContent with 'parsed' type
        - Structured outputs from .create() → Parsed JSON into Pydantic model
        - Unknown fields → GenericContent (forward compatibility)

        Args:
            anthropic_response: Raw Anthropic Message response
            model: Model name
            anthropic_params: Original request parameters (needed for response_format access)
            is_native_structured_output: Whether this is a native structured output response

        Returns:
            UnifiedResponse with all content blocks properly typed
        """
        content_blocks = []

        # Process all content blocks from Anthropic response
        for block in anthropic_response.content:
            # Extract thinking content (extended thinking feature)
            if _is_thinking_block(block):
                content_blocks.append(
                    ReasoningContent(
                        type="reasoning",
                        reasoning=block.thinking,
                        summary=None,
                    )
                )
            # Extract tool calls (handles both ToolUseBlock and BetaToolUseBlock)
            elif _is_tool_use_block(block):
                content_blocks.append(
                    ToolCallContent(
                        type="tool_call",
                        id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.input),
                    )
                )
            # Extract text content (handles both TextBlock and BetaTextBlock)
            elif _is_text_block(block):
                # For native structured output, handle both .parse() and .create() responses
                if is_native_structured_output:
                    # Check if we have parsed_output (from .parse())
                    if (
                        hasattr(anthropic_response, "parsed_output")
                        and anthropic_response.parsed_output is not None
                    ):
                        parsed_response = anthropic_response.parsed_output
                        # Store parsed object as GenericContent to preserve it
                        if hasattr(parsed_response, "model_dump"):
                            parsed_dict = parsed_response.model_dump()
                        elif hasattr(parsed_response, "dict"):
                            parsed_dict = parsed_response.dict()
                        else:
                            parsed_dict = {"value": str(parsed_response)}

                        content_blocks.append(GenericContent(type="parsed", parsed=parsed_dict))

                        # Also add text representation
                        text_content = (
                            parsed_response.model_dump_json()
                            if hasattr(parsed_response, "model_dump_json")
                            else str(parsed_response)
                        )
                        content_blocks.append(TextContent(type="text", text=text_content))
                    else:
                        # Using .create() - parse JSON text into Pydantic model if available
                        # Check if we have a Pydantic model to parse into
                        if (
                            self._response_format
                            and isinstance(self._response_format, type)
                            and issubclass(self._response_format, BaseModel)
                        ):
                            try:
                                # Parse JSON string into Pydantic model
                                json_data = json.loads(block.text)
                                parsed_response = self._response_format.model_validate(json_data)

                                # Store parsed object as GenericContent
                                parsed_dict = parsed_response.model_dump()
                                content_blocks.append(GenericContent(type="parsed", parsed=parsed_dict))

                                # Add text representation
                                text_content = parsed_response.model_dump_json()
                                content_blocks.append(TextContent(type="text", text=text_content))
                            except (json.JSONDecodeError, ValueError) as e:
                                # If parsing fails, log warning and use text as-is
                                logger.warning(f"Failed to parse structured output JSON: {e}")
                                content_blocks.append(TextContent(type="text", text=block.text))
                        else:
                            # Dict schema or no model - just use text as-is
                            content_blocks.append(TextContent(type="text", text=block.text))
                else:
                    # Regular text content (not structured output)
                    content_blocks.append(TextContent(type="text", text=block.text))

        # Fallback: If using native SO parse() and no content blocks were found,
        # extract from parsed_output directly
        if (
            not content_blocks
            and is_native_structured_output
            and hasattr(anthropic_response, "parsed_output")
            and anthropic_response.parsed_output is not None
        ):
            parsed_response = anthropic_response.parsed_output
            # Store parsed object as GenericContent
            if hasattr(parsed_response, "model_dump"):
                parsed_dict = parsed_response.model_dump()
            elif hasattr(parsed_response, "dict"):
                parsed_dict = parsed_response.dict()
            else:
                parsed_dict = {"value": str(parsed_response)}

            content_blocks.append(GenericContent(type="parsed", parsed=parsed_dict))

            # Add text representation
            text_content = (
                parsed_response.model_dump_json()
                if hasattr(parsed_response, "model_dump_json")
                else str(parsed_response)
            )
            content_blocks.append(TextContent(type="text", text=text_content))

        # Create unified message with normalized role (Anthropic responses are always assistant)
        messages = [
            UnifiedMessage(
                role=normalize_role("assistant"),
                content=content_blocks,
            )
        ]

        # Extract usage information
        usage = {
            "prompt_tokens": anthropic_response.usage.input_tokens,
            "completion_tokens": anthropic_response.usage.output_tokens,
            "total_tokens": anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens,
        }

        # Determine finish reason
        finish_reason = "stop"
        if anthropic_response.stop_reason == "tool_use":
            finish_reason = "tool_calls"

        # Build UnifiedResponse
        unified_response = UnifiedResponse(
            id=anthropic_response.id,
            model=model,
            provider="anthropic",
            messages=messages,
            usage=usage,
            finish_reason=finish_reason,
            status="completed",
            provider_metadata={
                "stop_reason": anthropic_response.stop_reason,
                "stop_sequence": getattr(anthropic_response, "stop_sequence", None),
            },
        )

        # Calculate cost
        unified_response.cost = self.cost(unified_response)

        return unified_response

    def load_config(self, params: dict[str, Any]) -> dict[str, Any]:
        """Load the configuration for the Anthropic API client."""
        anthropic_params = {}

        anthropic_params["model"] = params.get("model")
        assert anthropic_params["model"], "Please provide a `model` in the config_list to use the Anthropic API."

        anthropic_params["temperature"] = validate_parameter(
            params, "temperature", (float, int), False, 1.0, (0.0, 1.0), None
        )
        anthropic_params["max_tokens"] = validate_parameter(params, "max_tokens", int, False, 4096, (1, None), None)
        anthropic_params["timeout"] = validate_parameter(params, "timeout", int, True, None, (1, None), None)
        anthropic_params["top_k"] = validate_parameter(params, "top_k", int, True, None, (1, None), None)
        anthropic_params["top_p"] = validate_parameter(params, "top_p", (float, int), True, None, (0.0, 1.0), None)
        anthropic_params["stop_sequences"] = validate_parameter(params, "stop_sequences", list, True, None, None, None)
        anthropic_params["stream"] = validate_parameter(params, "stream", bool, False, False, None, None)
        if "thinking" in params:
            anthropic_params["thinking"] = params["thinking"]

        if anthropic_params["stream"]:
            warnings.warn(
                "Streaming is not currently supported, streaming will be disabled.",
                UserWarning,
            )
            anthropic_params["stream"] = False

        # Note the Anthropic API supports "tool" for tool_choice but you must specify the tool name so we will ignore that here
        # Dictionary, see options here: https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview#controlling-claudes-output
        # type = auto, any, tool, none | name = the name of the tool if type=tool
        anthropic_params["tool_choice"] = validate_parameter(params, "tool_choice", dict, True, None, None, None)

        return anthropic_params

    def _remove_none_params(self, params: dict[str, Any]) -> None:
        """Remove parameters with None values from the params dict.

        Anthropic API doesn't accept None values, so we remove them before making requests.
        This method modifies the params dict in-place.

        Args:
            params: Dictionary of API parameters
        """
        keys_to_remove = [key for key, value in params.items() if value is None]
        for key in keys_to_remove:
            del params[key]

    def _prepare_anthropic_params(
        self, params: dict[str, Any], anthropic_messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Prepare parameters for Anthropic API call.

        Consolidates common parameter preparation logic used across all create methods:
        - Loads base configuration
        - Converts tools format if needed
        - Assigns messages, system, and tools
        - Removes None values

        Args:
            params: Original request parameters
            anthropic_messages: Converted messages in Anthropic format

        Returns:
            Dictionary of Anthropic API parameters ready for use
        """
        # Load base configuration
        anthropic_params = self.load_config(params)

        # Convert tools to functions if needed (make a copy to avoid modifying original)
        params_copy = params.copy()
        if "functions" in params_copy:
            tools_configs = params_copy.pop("functions")
            tools_configs = [self.openai_func_to_anthropic(tool) for tool in tools_configs]
            params_copy["tools"] = tools_configs
        elif "tools" in params_copy:
            # Convert OpenAI tool format to Anthropic format
            # OpenAI format: {"type": "function", "function": {...}}
            # Anthropic format: {"name": "...", "description": "...", "input_schema": {...}}
            tools_configs = self.convert_tools_to_functions(params_copy.pop("tools"))
            tools_configs = [self.openai_func_to_anthropic(tool) for tool in tools_configs]
            params_copy["tools"] = tools_configs

        # Assign messages and optional parameters
        anthropic_params["messages"] = anthropic_messages
        if "system" in params_copy:
            anthropic_params["system"] = params_copy["system"]
        if "tools" in params_copy:
            anthropic_params["tools"] = params_copy["tools"]

        # Remove None values
        self._remove_none_params(anthropic_params)

        return anthropic_params

    def _extract_json_response(self, response: Message) -> Any:  # type: ignore[valid-type]
        """
        Extract and validate JSON response from the output for structured outputs.

        Args:
            response: The response from the API

        Returns:
            The parsed JSON response
        """
        if not self._response_format:
            return response

        # Extract content from response - check both thinking and text blocks
        content = ""
        if response.content:
            for block in response.content:
                if _is_thinking_block(block):
                    content = block.thinking
                    break
                elif _is_text_block(block):
                    content = block.text
                    break

        # Try to extract JSON from tags first
        json_match = re.search(r"<json_response>(.*?)</json_response>", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Fallback to finding first JSON object
            json_start = content.find("{")
            json_end = content.rfind("}")
            if json_start == -1 or json_end == -1:
                raise ValueError("No valid JSON found in response for Structured Output.")
            json_str = content[json_start : json_end + 1]

        try:
            # Parse JSON and validate against the Pydantic model if Pydantic model was provided
            json_data = json.loads(json_str)
            if isinstance(self._response_format, dict):
                return json_str
            else:
                return self._response_format.model_validate(json_data)

        except Exception as e:
            raise ValueError(f"Failed to parse response as valid JSON matching the schema for Structured Output: {e!s}")

    def _resolve_schema_refs(self, schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
        """Recursively resolve $ref references in a JSON schema.

        Args:
            schema: The schema to resolve
            defs: The definitions dict from $defs

        Returns:
            Schema with all $ref references resolved inline
        """
        if isinstance(schema, dict):
            if "$ref" in schema:
                # Extract the reference name (e.g., "#/$defs/Step" -> "Step")
                ref_name = schema["$ref"].split("/")[-1]
                # Replace with the actual definition
                return self._resolve_schema_refs(defs[ref_name].copy(), defs)
            else:
                # Recursively resolve all nested schemas
                return {k: self._resolve_schema_refs(v, defs) for k, v in schema.items()}
        elif isinstance(schema, list):
            return [self._resolve_schema_refs(item, defs) for item in schema]
        else:
            return schema

    def _add_response_format_to_system(self, params: dict[str, Any]) -> None:
        """
        Add prompt that will generate properly formatted JSON for structured outputs to system parameter.

        Based on Anthropic's JSON Mode cookbook, we ask the LLM to put the JSON within <json_response> tags.

        Args:
            params: The client parameters (modified in place)
        """
        # Get the schema of the Pydantic model
        if isinstance(self._response_format, dict):
            schema = self._response_format
        else:
            # Use mode='serialization' and ref_template='{model}' to get a flatter, more LLM-friendly schema
            schema = self._response_format.model_json_schema(mode="serialization", ref_template="{model}")

            # Resolve $ref references for simpler schema
            if "$defs" in schema:
                defs = schema.pop("$defs")
                schema = self._resolve_schema_refs(schema, defs)

        # Add instructions for JSON formatting
        # Generate an example based on the actual schema
        def generate_example(schema_dict: dict[str, Any]) -> dict[str, Any]:
            """Generate example data from schema."""
            example = {}
            properties = schema_dict.get("properties", {})
            for prop_name, prop_schema in properties.items():
                prop_type = prop_schema.get("type", "string")
                if prop_type == "string":
                    example[prop_name] = f"example {prop_name}"
                elif prop_type == "integer":
                    example[prop_name] = 42
                elif prop_type == "number":
                    example[prop_name] = 42.0
                elif prop_type == "boolean":
                    example[prop_name] = True
                elif prop_type == "array":
                    items_schema = prop_schema.get("items", {})
                    items_type = items_schema.get("type", "string")
                    if items_type == "string":
                        example[prop_name] = ["item1", "item2"]
                    elif items_type == "object":
                        example[prop_name] = [generate_example(items_schema)]
                    else:
                        example[prop_name] = []
                elif prop_type == "object":
                    example[prop_name] = generate_example(prop_schema)
                else:
                    example[prop_name] = f"example {prop_name}"
            return example

        example_data = generate_example(schema)
        example_json = json.dumps(example_data, indent=2)

        format_content = f"""You must respond with a valid JSON object that matches this structure (do NOT return the schema itself):
{json.dumps(schema, indent=2)}

IMPORTANT: Put your actual response data (not the schema) inside <json_response> tags.

Correct example format:
<json_response>
{example_json}
</json_response>

WRONG: Do not return the schema definition itself.

Your JSON must:
1. Match the schema structure above
2. Contain actual data values, not schema descriptions
3. Be valid, parseable JSON"""

        # Add formatting to system message (create one if it doesn't exist)
        if "system" in params:
            params["system"] = params["system"] + "\n\n" + format_content
        else:
            params["system"] = format_content

    def _log_structured_output_fallback(
        self,
        exception: Exception,
        model: str | None,
        response_format: Any,
        params: dict[str, Any],
    ) -> None:
        """
        Log detailed error information when native structured output fails and we fallback to JSON Mode.

        Args:
            exception: The exception that triggered the fallback
            model: Model name/identifier
            response_format: Response format specification (Pydantic model or dict)
            params: Original request parameters
        """
        # Build error details dictionary
        error_details = {
            "model": model,
            "response_format": str(
                type(response_format).__name__ if isinstance(response_format, type) else type(response_format)
            ),
            "error_type": type(exception).__name__,
            "error_message": str(exception),
        }

        # Add BadRequestError-specific details if available
        if isinstance(exception, BadRequestError):  # type: ignore[misc]
            if hasattr(exception, "status_code"):
                error_details["status_code"] = exception.status_code
            if hasattr(exception, "response"):
                error_details["response_body"] = str(
                    exception.response.text if hasattr(exception.response, "text") else exception.response
                )
            if hasattr(exception, "body"):
                error_details["error_body"] = str(exception.body)

        # Log sanitized params (remove sensitive data like API keys, message content)
        sanitized_params = {
            "model": params.get("model"),
            "max_tokens": params.get("max_tokens"),
            "temperature": params.get("temperature"),
            "has_tools": "tools" in params,
            "num_messages": len(params.get("messages", [])),
        }
        error_details["params"] = sanitized_params

        # Log warning with full error context
        logger.warning(
            f"Native structured output failed for {model}. Error: {error_details}. Falling back to JSON Mode."
        )

    def create_v1_compatible(self, params: dict[str, Any]) -> ChatCompletion:
        """
        Create completion in backward-compatible ChatCompletion format.

        This method provides compatibility with existing AG2 code that expects
        ChatCompletion format. Note that thinking blocks will be preserved in
        the content string with [Thinking] tags, matching V1 behavior.

        Args:
            params: Same parameters as create()

        Returns:
            ChatCompletion object compatible with OpenAI format

        Warning:
            This method may lose some information when converting to the legacy format.
            Prefer create() for new code.
        """
        # Get rich response
        unified_response = self.create(params)

        # Build message text with proper thinking block formatting (matching V1 behavior)
        message_text = ""
        for msg in unified_response.messages:
            # Extract reasoning blocks (thinking content)
            reasoning_blocks = msg.get_reasoning()
            # Extract text content blocks
            text_blocks = [b for b in msg.content if isinstance(b, TextContent)]

            # Combine thinking content (multiple blocks joined with \n\n)
            thinking_content = "\n\n".join([r.reasoning for r in reasoning_blocks])
            # Combine text content (multiple blocks joined with \n\n)
            text_content = "\n\n".join([t.text for t in text_blocks])

            # Format like V1: [Thinking]\n{thinking}\n\n{text}
            if thinking_content and text_content:
                message_text = f"[Thinking]\n{thinking_content}\n\n{text_content}"
            elif thinking_content:
                message_text = f"[Thinking]\n{thinking_content}"
            elif text_content:
                message_text = text_content
            break  # Anthropic responses have single message

        # Extract tool calls if present
        tool_calls = None
        for msg in unified_response.messages:
            tool_call_blocks = msg.get_tool_calls()
            if tool_call_blocks:
                tool_calls = [
                    ChatCompletionMessageToolCall(
                        id=tc.id,
                        function={"name": tc.name, "arguments": tc.arguments},
                        type="function",
                    )
                    for tc in tool_call_blocks
                ]
                break

        # Build ChatCompletion
        message = ChatCompletionMessage(
            role="assistant",
            content=message_text,
            function_call=None,
            tool_calls=tool_calls,
        )

        choices = [Choice(finish_reason=unified_response.finish_reason or "stop", index=0, message=message)]

        return ChatCompletion(
            id=unified_response.id,
            model=unified_response.model,
            created=int(time.time()),
            object="chat.completion",
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=unified_response.usage.get("prompt_tokens", 0),
                completion_tokens=unified_response.usage.get("completion_tokens", 0),
                total_tokens=unified_response.usage.get("total_tokens", 0),
            ),
            cost=unified_response.cost or 0.0,
        )

    def message_retrieval(self, response: UnifiedResponse) -> list[str] | list[ChatCompletionMessage]:  # type: ignore[override]
        """
        Retrieve messages from response in OpenAI-compatible format.

        Returns list of strings for text-only messages, or list of dicts when
        tool calls or complex content is present.

        Args:
            response: UnifiedResponse from create()

        Returns:
            List of strings (for text-only) OR list of message dicts (for tool calls/complex content)
        """
        result: list[str] | list[ChatCompletionMessage] = []

        for msg in response.messages:
            # Check for tool calls
            tool_calls = msg.get_tool_calls()

            # Check for complex/multimodal content that needs dict format
            has_complex_content = any(
                isinstance(block, (ImageContent, AudioContent, VideoContent)) for block in msg.content
            )

            if tool_calls or has_complex_content:
                # Return OpenAI-compatible dict format
                message_dict = ChatCompletionMessage(
                    role=msg.role.value if hasattr(msg.role, "value") else msg.role,
                    content=msg.get_text() or None,
                )

                # Add tool calls in OpenAI format
                if tool_calls:
                    message_dict.tool_calls = [
                        ChatCompletionMessageToolCall(
                            id=tc.id,
                            type="function",
                            function={"name": tc.name, "arguments": tc.arguments},
                        )
                        for tc in tool_calls
                    ]

                result.append(message_dict)
            else:
                # Simple text content - apply FormatterProtocol if available
                content = msg.get_text()

                # If response_format implements FormatterProtocol (has format() method), use it
                if isinstance(self._response_format, FormatterProtocol):
                    try:
                        # Try to parse and format
                        parsed = self._response_format.model_validate_json(content)  # type: ignore[union-attr]
                        content = parsed.format()  # type: ignore[union-attr]
                    except Exception:
                        # If parsing fails, return as-is
                        pass

                result.append(content)

        return result

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

        return _calculate_cost(prompt_tokens, completion_tokens, model)

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

    @staticmethod
    def openai_func_to_anthropic(openai_func: dict) -> dict:
        """Convert OpenAI function format to Anthropic format.

        Args:
            openai_func: OpenAI function definition

        Returns:
            Anthropic function definition
        """
        res = openai_func.copy()
        res["input_schema"] = res.pop("parameters")

        # Preserve strict field if present (for Anthropic structured outputs)
        # strict=True enables guaranteed schema validation for tool inputs
        if "strict" in openai_func:
            res["strict"] = openai_func["strict"]
            # Transform schema to add required additionalProperties: false for all objects
            # Anthropic requires this for strict tools
            res["input_schema"] = transform_schema_for_anthropic(res["input_schema"])

        return res

    @staticmethod
    def convert_tools_to_functions(tools: list) -> list:
        """Convert tool definitions into Anthropic-compatible functions,
        updating nested $ref paths in property schemas.

        Args:
            tools: List of tool definitions

        Returns:
            List of functions with updated $ref paths
        """

        def update_refs(obj: Any, defs_keys: set[str], prop_name: str) -> None:
            """Recursively update $ref values that start with "#/$defs/"."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "$ref" and isinstance(value, str) and value.startswith("#/$defs/"):
                        ref_key = value[len("#/$defs/") :]
                        if ref_key in defs_keys:
                            obj[key] = f"#/properties/{prop_name}/$defs/{ref_key}"
                    else:
                        update_refs(value, defs_keys, prop_name)
            elif isinstance(obj, list):
                for item in obj:
                    update_refs(item, defs_keys, prop_name)

        functions = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                function = tool["function"]
                parameters = function.get("parameters", {})
                properties = parameters.get("properties", {})
                for prop_name, prop_schema in properties.items():
                    if "$defs" in prop_schema:
                        defs_keys = set(prop_schema["$defs"].keys())
                        update_refs(prop_schema, defs_keys, prop_name)
                functions.append(function)
        return functions
