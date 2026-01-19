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

Example:
    client = BedrockV2Client(
        aws_region="us-east-1",
        aws_access_key="...",
        aws_secret_key="..."
    )

    # Get rich response
    response = client.create({
        "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "messages": [{"role": "user", "content": "Hello"}]
    })

    # Access text content
    print(f"Response: {response.text}")

    # Access tool calls
    for tool_call in response.get_content_by_type("tool_call"):
        print(f"Tool: {tool_call.name}")
"""

from __future__ import annotations

import base64
import json
import os
import time
import warnings
from typing import Any, Literal

from autogen.import_utils import optional_import_block, require_optional_import
from autogen.llm_config.client import ModelClient
from autogen.llm_clients.models import (
    GenericContent,
    ImageContent,
    TextContent,
    ToolCallContent,
    UnifiedMessage,
    UnifiedResponse,
    UserRoleEnum,
    normalize_role,
)

# Import Bedrock-specific utilities from existing client
from autogen.oai.bedrock import (
    calculate_cost,
    convert_stop_reason_to_finish_reason,
    extract_system_messages,
    format_tool_calls,
    format_tools,
    oai_messages_to_bedrock_messages,
)
from autogen.oai.client_utils import validate_parameter

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

    Example:
        client = BedrockV2Client(
            aws_region="us-east-1",
            aws_access_key="...",
            aws_secret_key="..."
        )

        # Get rich response
        response = client.create({
            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "messages": [{"role": "user", "content": "Hello"}]
        })

        # Access text content
        print(f"Response: {response.text}")

        # Access tool calls
        for tool_call in response.get_content_by_type("tool_call"):
            print(f"Tool: {tool_call.name}")
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
        if boto3_import_exception is not None:
            raise boto3_import_exception

        self._aws_access_key = aws_access_key or os.getenv("AWS_ACCESS_KEY")
        self._aws_secret_key = aws_secret_key or os.getenv("AWS_SECRET_KEY")
        self._aws_session_token = aws_session_token or os.getenv("AWS_SESSION_TOKEN")
        self._aws_region = aws_region or os.getenv("AWS_REGION")
        self._aws_profile_name = aws_profile_name
        self._timeout = timeout or 60
        self._total_max_attempts = total_max_attempts
        self._max_attempts = max_attempts
        self._mode = mode
        self._response_format = response_format

        if self._aws_region is None:
            raise ValueError("Region is required to use the Amazon Bedrock API. Set aws_region or AWS_REGION env var.")

        # Initialize retry configuration
        self._retry_config = {
            "total_max_attempts": self._total_max_attempts,
            "max_attempts": self._max_attempts,
            "mode": self._mode,
        }

        # Initialize Bedrock client configuration
        bedrock_config = Config(
            region_name=self._aws_region,
            signature_version="v4",
            retries=self._retry_config,
            read_timeout=self._timeout,
        )

        # Initialize Bedrock runtime client
        if (
            self._aws_access_key is None
            or self._aws_access_key == ""
            or self._aws_secret_key is None
            or self._aws_secret_key == ""
        ):
            # Use attached role (Lambda, EC2, ECS, etc.)
            self.bedrock_runtime = boto3.client(service_name="bedrock-runtime", config=bedrock_config)
        else:
            session = boto3.Session(
                aws_access_key_id=self._aws_access_key,
                aws_secret_access_key=self._aws_secret_key,
                aws_session_token=self._aws_session_token,
                profile_name=self._aws_profile_name,
            )
            self.bedrock_runtime = session.client(service_name="bedrock-runtime", config=bedrock_config)

        # Store model-specific pricing (can be overridden via price parameter)
        self._price_per_1k_tokens: tuple[float, float] | None = None

    def _get_response_format_schema(self, response_format: Any) -> dict[str, Any]:
        """Extract and normalize JSON schema from response_format."""
        from pydantic import BaseModel

        schema = response_format.copy() if isinstance(response_format, dict) else response_format.model_json_schema()

        if "type" not in schema:
            schema["type"] = "object"
        elif schema.get("type") != "object":
            schema = {"type": "object", "properties": {"data": schema}, "required": ["data"]}

        if "properties" not in schema:
            schema["properties"] = {}
        if "required" not in schema:
            schema["required"] = []

        return schema

    def _normalize_pydantic_schema_to_dict(self, schema: dict[str, Any] | type) -> dict[str, Any]:
        """Convert a Pydantic model's JSON schema to a flat dict schema by resolving $ref references."""
        from pydantic import BaseModel

        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_dict = schema.model_json_schema()
        elif isinstance(schema, dict):
            schema_dict = schema.copy()
        else:
            raise ValueError(f"Schema must be a Pydantic model class or dict, got {type(schema)}")

        defs = schema_dict.get("$defs", {}).copy()

        def resolve_ref(ref: str, definitions: dict[str, Any]) -> dict[str, Any]:
            if not ref.startswith("#/$defs/"):
                raise ValueError(f"Unsupported $ref format: {ref}. Only '#/$defs/...' is supported.")
            def_name = ref.split("/")[-1]
            if def_name not in definitions:
                raise ValueError(f"Definition '{def_name}' not found in $defs")
            return definitions[def_name].copy()

        def resolve_refs_recursive(obj: Any, definitions: dict[str, Any]) -> Any:
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref_def = resolve_ref(obj["$ref"], definitions)
                    merged = {**ref_def, **{k: v for k, v in obj.items() if k != "$ref"}}
                    return resolve_refs_recursive(merged, definitions)
                else:
                    return {k: resolve_refs_recursive(v, definitions) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_refs_recursive(item, definitions) for item in obj]
            else:
                return obj

        normalized_schema = resolve_refs_recursive(schema_dict, defs)
        if "$defs" in normalized_schema:
            del normalized_schema["$defs"]

        return normalized_schema

    def _create_structured_output_tool(self, response_format: Any) -> dict[str, Any]:
        """Convert response_format into a Bedrock tool definition for structured outputs."""
        schema = self._get_response_format_schema(response_format)
        schema = self._normalize_pydantic_schema_to_dict(schema)

        return {
            "type": "function",
            "function": {
                "name": "__structured_output",
                "description": "Generate structured output matching the specified schema",
                "parameters": schema,
            },
        }

    def _merge_tools_with_structured_output(
        self, user_tools: list[dict[str, Any]], structured_output_tool: dict[str, Any]
    ) -> dict[Literal["tools"], list[dict[str, Any]]]:
        """Merge user tools with structured output tool."""
        all_tools = list(user_tools) if user_tools else []
        all_tools.append(structured_output_tool)
        return format_tools(all_tools)

    def _extract_structured_output_from_tool_call(self, tool_calls: list[Any]) -> dict[str, Any] | None:
        """Extract structured output data from tool call response."""
        for tool_call in tool_calls:
            if hasattr(tool_call, "function") and tool_call.function.name == "__structured_output":
                try:
                    return json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, AttributeError) as e:
                    raise ValueError(f"Failed to parse structured output from tool call: {e!s}") from e
        return None

    def _validate_and_format_structured_output(self, structured_data: dict[str, Any]) -> str:
        """Validate structured data against schema and format for response message."""
        if self._response_format:
            try:
                from pydantic import BaseModel

                if isinstance(self._response_format, dict):
                    validated_data = structured_data
                else:
                    validated_data = self._response_format.model_validate(structured_data)

                from autogen.oai.client_utils import FormatterProtocol

                if isinstance(validated_data, FormatterProtocol):
                    return validated_data.format()
                elif hasattr(validated_data, "model_dump_json"):
                    return validated_data.model_dump_json()
                else:
                    return json.dumps(structured_data)
            except Exception as e:
                raise ValueError(f"Failed to validate structured output against schema: {e!s}") from e

        return json.dumps(structured_data)

    def parse_custom_params(self, params: dict[str, Any]) -> None:
        """Parses custom parameters for logic in this client class."""
        self._supports_system_prompts = params.get("supports_system_prompts", True)

        if "price" in params and isinstance(params["price"], list) and len(params["price"]) == 2:
            self._price_per_1k_tokens = (params["price"][0], params["price"][1])

    def parse_params(self, params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Loads the valid parameters required to invoke Bedrock Converse."""
        self._model_id = params.get("model")
        if not self._model_id:
            raise ValueError("Please provide the 'model' in the config_list to use Amazon Bedrock")

        config_only_fields = {
            "api_type",
            "model",
            "aws_region",
            "aws_access_key",
            "aws_secret_key",
            "aws_session_token",
            "aws_profile_name",
            "supports_system_prompts",
            "price",
            "timeout",
            "api_key",
            "messages",
            "tools",
            "response_format",
        }

        base_params = {}
        if "temperature" in params:
            base_params["temperature"] = validate_parameter(
                params, "temperature", (float, int), False, None, None, None
            )
        if "top_p" in params:
            base_params["topP"] = validate_parameter(params, "top_p", (float, int), False, None, None, None)
        if "max_tokens" in params:
            base_params["maxTokens"] = validate_parameter(params, "max_tokens", (int,), False, None, None, None)

        additional_params = {}
        for param_name, suitable_types in (
            ("top_k", (int,)),
            ("k", (int,)),
            ("seed", (int,)),
            ("cache_seed", (int,)),
        ):
            if param_name in params and param_name not in config_only_fields:
                additional_params[param_name] = validate_parameter(
                    params, param_name, suitable_types, False, None, None, None
                )

        if "additional_model_request_fields" in params and isinstance(params["additional_model_request_fields"], dict):
            additional_model_fields = params["additional_model_request_fields"]
            for key, value in additional_model_fields.items():
                if key not in config_only_fields:
                    additional_params[key] = value

        if params.get("stream", False):
            warnings.warn(
                "Streaming is not currently supported, streaming will be disabled.",
                UserWarning,
            )

        return base_params, additional_params

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
        params = params.copy()

        self.parse_custom_params(params)
        base_params, additional_params = self.parse_params(params)

        has_response_format = self._response_format is not None
        if has_response_format:
            structured_output_tool = self._create_structured_output_tool(self._response_format)
            user_tools = params.get("tools", [])
            tool_config = self._merge_tools_with_structured_output(user_tools, structured_output_tool)
            has_tools = len(tool_config["tools"]) > 0
        else:
            has_tools = "tools" in params
            tool_config = format_tools(params["tools"] if has_tools else [])
            has_tools = len(tool_config["tools"]) > 0

        messages = oai_messages_to_bedrock_messages(
            params["messages"], has_tools or has_response_format, self._supports_system_prompts
        )

        system_messages = None
        if self._supports_system_prompts:
            system_messages = extract_system_messages(params["messages"])

        request_args: dict[str, Any] = {"messages": messages, "modelId": self._model_id}

        if len(base_params) > 0:
            request_args["inferenceConfig"] = base_params

        if len(additional_params) > 0:
            request_args["additionalModelRequestFields"] = additional_params

        if system_messages:
            request_args["system"] = system_messages

        if len(tool_config["tools"]) > 0:
            request_args["toolConfig"] = tool_config

        response = self.bedrock_runtime.converse(**request_args)
        if response is None:
            raise RuntimeError(f"Failed to get response from Bedrock after retrying {self._retries} times.")

        return self._transform_response(response, has_response_format)

    def _transform_response(self, bedrock_response: dict[str, Any], has_response_format: bool = False) -> UnifiedResponse:
        """
        Transform AWS Bedrock Converse API response to UnifiedResponse.

        Content handling:
        - Text content → TextContent
        - Image content → ImageContent
        - Tool calls → ToolCallContent
        - Unknown content types → GenericContent (forward compatibility)
        """
        finish_reason = convert_stop_reason_to_finish_reason(bedrock_response["stopReason"])
        response_message = bedrock_response["output"]["message"]
        response_content = response_message.get("content", [])

        tool_calls = None
        if finish_reason == "tool_calls":
            tool_calls = format_tool_calls(response_content)

        content_blocks = []

        structured_text = None
        if has_response_format and finish_reason == "tool_calls" and tool_calls:
            structured_data = self._extract_structured_output_from_tool_call(tool_calls)
            if structured_data:
                structured_text = self._validate_and_format_structured_output(structured_data)

        for content_part in response_content:
            if "text" in content_part:
                text = content_part["text"]
                if structured_text is not None and text == "":
                    text = structured_text
                if text:
                    content_blocks.append(TextContent(text=text))

            elif "image" in content_part:
                image_data = content_part["image"]
                image_format = image_data.get("format", "jpeg")
                image_source = image_data.get("source", {})

                if "bytes" in image_source:
                    image_bytes = image_source["bytes"]
                    if isinstance(image_bytes, bytes):
                        base64_data = base64.b64encode(image_bytes).decode("utf-8")
                        data_uri = f"data:image/{image_format};base64,{base64_data}"
                        content_blocks.append(ImageContent(data_uri=data_uri))
                    else:
                        data_uri = f"data:image/{image_format};base64,{image_bytes}"
                        content_blocks.append(ImageContent(data_uri=data_uri))
                elif "url" in image_source:
                    content_blocks.append(ImageContent(image_url=image_source["url"]))

            elif "toolUse" in content_part:
                tool_use = content_part["toolUse"]
                content_blocks.append(
                    ToolCallContent(
                        id=tool_use["toolUseId"],
                        name=tool_use["name"],
                        arguments=json.dumps(tool_use.get("input", {})),
                    )
                )

            else:
                content_blocks.append(GenericContent(type="unknown", **content_part))

        if structured_text and not any(isinstance(b, TextContent) for b in content_blocks):
            content_blocks.insert(0, TextContent(text=structured_text))

        if not content_blocks:
            content_blocks.append(TextContent(text=""))

        messages = [
            UnifiedMessage(
                role=normalize_role("assistant"),
                content=content_blocks,
            )
        ]

        response_usage = bedrock_response.get("usage", {})
        usage = {
            "prompt_tokens": response_usage.get("inputTokens", 0),
            "completion_tokens": response_usage.get("outputTokens", 0),
            "total_tokens": response_usage.get("totalTokens", 0),
        }

        unified_response = UnifiedResponse(
            id=bedrock_response.get("ResponseMetadata", {}).get("RequestId", f"bedrock-{int(time.time())}"),
            model=self._model_id,
            provider="bedrock",
            messages=messages,
            usage=usage,
            finish_reason=finish_reason,
            status="completed",
            provider_metadata={
                "stopReason": bedrock_response.get("stopReason"),
                "amazonBedrockInvocationMetrics": bedrock_response.get("amazonBedrockInvocationMetrics"),
            },
        )

        unified_response.cost = self.cost(unified_response)

        return unified_response

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
        unified_response = self.create(params)

        role = unified_response.messages[0].role if unified_response.messages else UserRoleEnum.ASSISTANT
        role_str = role.value if isinstance(role, UserRoleEnum) else role

        text = unified_response.text

        tool_calls_list = []
        for msg in unified_response.messages:
            for tool_call in msg.get_tool_calls():
                tool_calls_list.append({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                })

        return {
            "id": unified_response.id,
            "model": unified_response.model,
            "created": int(time.time()),
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": role_str,
                        "content": text,
                        **({"tool_calls": tool_calls_list} if tool_calls_list else {}),
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

        prompt_tokens = response.usage.get("prompt_tokens", 0)
        completion_tokens = response.usage.get("completion_tokens", 0)

        if self._price_per_1k_tokens:
            input_cost_per_k, output_cost_per_k = self._price_per_1k_tokens
            input_cost = (prompt_tokens / 1000) * input_cost_per_k
            output_cost = (completion_tokens / 1000) * output_cost_per_k
            return input_cost + output_cost

        from autogen.oai.bedrock import calculate_cost

        return calculate_cost(prompt_tokens, completion_tokens, response.model)

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
        result: list[str] | list[dict[str, Any]] = []

        for msg in response.messages:
            tool_calls = msg.get_tool_calls()

            has_complex_content = any(
                isinstance(block, (ImageContent,)) for block in msg.content
            )

            if tool_calls or has_complex_content:
                message_dict: dict[str, Any] = {
                    "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                    "content": msg.get_text() or None,
                }

                if tool_calls:
                    message_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in tool_calls
                    ]

                result.append(message_dict)
            else:
                result.append(msg.get_text())

        return result