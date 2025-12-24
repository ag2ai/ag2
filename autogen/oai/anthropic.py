# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
"""Create an OpenAI-compatible client for the Anthropic API.

Example usage:
Install the `anthropic` package by running `pip install --upgrade anthropic`.
- https://docs.anthropic.com/en/docs/quickstart-guide

```python
import autogen

config_list = [
    {
        "model": "claude-3-sonnet-20240229",
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "api_type": "anthropic",
    }
]

assistant = autogen.AssistantAgent("assistant", llm_config={"config_list": config_list})
```

Example usage for Anthropic Bedrock:

Install the `anthropic` package by running `pip install --upgrade anthropic`.
- https://docs.anthropic.com/en/docs/quickstart-guide

```python
import autogen

config_list = [
    {
        "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "aws_access_key":<accessKey>,
        "aws_secret_key":<secretKey>,
        "aws_session_token":<sessionTok>,
        "aws_region":"us-east-1",
        "api_type": "anthropic",
    }
]

assistant = autogen.AssistantAgent("assistant", llm_config={"config_list": config_list})
```

Example usage for Anthropic VertexAI:

Install the `anthropic` package by running `pip install anthropic[vertex]`.
- https://docs.anthropic.com/en/docs/quickstart-guide

```python

import autogen
config_list = [
    {
        "model": "claude-3-5-sonnet-20240620-v1:0",
        "gcp_project_id": "dummy_project_id",
        "gcp_region": "us-west-2",
        "gcp_auth_token": "dummy_auth_token",
        "api_type": "anthropic",
    }
]

assistant = autogen.AssistantAgent("assistant", llm_config={"config_list": config_list})
```python
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import time
import warnings
from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import Unpack

from ..code_utils import content_str
from ..import_utils import optional_import_block, require_optional_import
from ..llm_config.entry import LLMConfigEntry, LLMConfigEntryDict

logger = logging.getLogger(__name__)
from .client_utils import FormatterProtocol, validate_parameter
from .oai_models import ChatCompletion, ChatCompletionMessage, ChatCompletionMessageToolCall, Choice, CompletionUsage

with optional_import_block():
    from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex, BadRequestError
    from anthropic import __version__ as anthropic_version
    from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock

    # Import transform_schema for structured outputs (SDK >= 0.74.1)
    try:
        from anthropic import transform_schema
    except ImportError:
        transform_schema = None  # type: ignore[misc, assignment]

    # Beta content block types for structured outputs (SDK >= 0.74.1)
    try:
        from anthropic.types.beta.structured_outputs.beta_text_block import BetaTextBlock
        from anthropic.types.beta.structured_outputs.beta_tool_use_block import BetaToolUseBlock

        BETA_BLOCKS_AVAILABLE = True
    except ImportError:
        # Beta blocks not available in older SDK versions
        BetaTextBlock = None  # type: ignore[misc, assignment]
        BetaToolUseBlock = None  # type: ignore[misc, assignment]
        BETA_BLOCKS_AVAILABLE = False

    TOOL_ENABLED = anthropic_version >= "0.23.1"
    if TOOL_ENABLED:
        pass


ANTHROPIC_PRICING_1k = {
    "claude-3-7-sonnet-20250219": (0.003, 0.015),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-5-haiku-20241022": (0.0008, 0.004),
    "claude-3-5-sonnet-20240620": (0.003, 0.015),
    "claude-3-sonnet-20240229": (0.003, 0.015),
    "claude-3-opus-20240229": (0.015, 0.075),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
    "claude-2.1": (0.008, 0.024),
    "claude-2.0": (0.008, 0.024),
    "claude-instant-1.2": (0.008, 0.024),
}

# Models that support native structured outputs via beta API
# https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs
STRUCTURED_OUTPUT_MODELS = {
    "claude-sonnet-4-5",
    "claude-sonnet-4-5-20250929",  # Versioned Claude Sonnet 4.5
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-opus-4-1",  # Future model
}


def supports_native_structured_outputs(model: str) -> bool:
    """Check if a Claude model supports native structured outputs (beta feature).

    Native structured outputs use constrained decoding to guarantee schema compliance.
    This is more reliable than JSON Mode which relies on prompting.

    Args:
        model: The Claude model name (e.g., "claude-sonnet-4-5")

    Returns:
        True if the model supports native structured outputs, False otherwise.

    Supported models:
        - Claude Sonnet 4.5+ (claude-sonnet-4-5, claude-sonnet-4-5-20250929, claude-3-5-sonnet-20241022+)
        - Claude Sonnet 3.7+ (claude-3-7-sonnet-20250219+)
        - Claude Opus 4.1+ (claude-opus-4-1+)

    NOT supported (will use JSON Mode fallback):
        - Claude Sonnet 4.0 (claude-sonnet-4-20250514) - older version
        - Claude 3 Haiku models
        - Claude 2.x models

    Example:
        >>> supports_native_structured_outputs("claude-sonnet-4-5")
        True
        >>> supports_native_structured_outputs("claude-sonnet-4-20250514")
        False  # Claude Sonnet 4.0 doesn't support it
        >>> supports_native_structured_outputs("claude-3-haiku-20240307")
        False
    """
    # Exact match for known models
    if model in STRUCTURED_OUTPUT_MODELS:
        return True

    # Pattern matching for versioned models
    # Support future Sonnet 3.5+ and 3.7+ versions
    if model.startswith(("claude-3-5-sonnet-", "claude-3-7-sonnet-")):
        return True

    # Support future Sonnet 4.5+ versions (NOT Sonnet 4.0)
    if model.startswith("claude-sonnet-4-5"):
        return True

    # Support future Opus 4.x versions
    if model.startswith("claude-opus-4"):
        return True

    return False


def has_beta_messages_api() -> bool:
    """Check if the current Anthropic SDK version supports beta.messages API.

    The beta.messages API is required for native structured outputs.
    This function performs runtime detection of SDK capabilities.

    Returns:
        True if beta.messages.parse() is available, False otherwise.

    Example:
        >>> has_beta_messages_api()
        True  # If anthropic>=0.39.0 is installed
    """
    try:
        from anthropic.resources.beta.messages import Messages

        return hasattr(Messages, "parse")
    except ImportError:
        return False


def validate_structured_outputs_version() -> None:
    """Validate that the Anthropic SDK version supports structured outputs beta.

    The structured-outputs-2025-11-13 beta header requires anthropic>=0.74.1.

    Raises:
        ImportError: If the Anthropic SDK version is too old

    Example:
        >>> validate_structured_outputs_version()  # Raises if version < 0.74.1
    """
    try:
        from packaging import version

        min_version = "0.74.1"
        current_version = anthropic_version  # Use module-level import

        if version.parse(current_version) < version.parse(min_version):
            raise ImportError(
                f"Anthropic structured outputs require anthropic>={min_version}, "
                f"but found version {current_version}. "
                f"Please upgrade: pip install --upgrade 'anthropic>={min_version}'"
            )
    except ImportError as e:
        if "anthropic" in str(e) or "version" in str(e).lower():
            raise
        # If packaging is not available, try manual version comparison
        current_version = anthropic_version  # Use module-level import

        # Simple version comparison (works for major.minor.patch format)
        current_parts = [int(x) for x in anthropic_version.split(".")[:3]]
        min_parts = [0, 74, 1]

        if current_parts < min_parts:
            raise ImportError(
                f"Anthropic structured outputs require anthropic>=0.74.1, "
                f"but found version {anthropic_version}. "
                f"Please upgrade: pip install --upgrade 'anthropic>=0.74.1'"
            )


def _is_text_block(content: Any) -> bool:
    """Check if a content block is a text block (legacy or beta version).

    Args:
        content: Content block to check

    Returns:
        True if content is a TextBlock or BetaTextBlock
    """
    if type(content) == TextBlock:
        return True
    if BETA_BLOCKS_AVAILABLE and type(content) == BetaTextBlock:
        return True
    return False


def _is_tool_use_block(content: Any) -> bool:
    """Check if a content block is a tool use block (legacy or beta version).

    Args:
        content: Content block to check

    Returns:
        True if content is a ToolUseBlock or BetaToolUseBlock
    """
    content_type = type(content)
    content_type_name = content_type.__name__

    if content_type == ToolUseBlock:
        return True
    if BETA_BLOCKS_AVAILABLE and content_type == BetaToolUseBlock:
        return True

    # Fallback: check by name if type comparison fails
    if content_type_name in ("ToolUseBlock", "BetaToolUseBlock"):
        return True

    return False


def _is_thinking_block(content: Any) -> bool:
    """Check if a content block is a thinking block (extended thinking).

    Args:
        content: Content block to check

    Returns:
        True if content is a ThinkingBlock
    """
    content_type = type(content)
    content_type_name = content_type.__name__

    if content_type == ThinkingBlock:
        return True

    # Fallback: check by name if type comparison fails
    if content_type_name == "ThinkingBlock":
        return True

    return False


def transform_schema_for_anthropic(schema: dict[str, Any]) -> dict[str, Any]:
    """Transform JSON schema to be compatible with Anthropic's structured outputs.

    Anthropic's structured outputs don't support certain JSON Schema features:
    - Numerical constraints (minimum, maximum, multipleOf)
    - String length constraints (minLength, maxLength, pattern with backreferences)
    - Recursive schemas ($ref loops)
    - Complex regex patterns

    This function removes unsupported constraints while preserving the core structure.

    Args:
        schema: A JSON schema dict (typically from Pydantic model_json_schema())

    Returns:
        Transformed schema compatible with Anthropic's requirements

    Example:
        >>> schema = {"type": "object", "properties": {"age": {"type": "integer", "minimum": 0, "maximum": 150}}}
        >>> transformed = transform_schema_for_anthropic(schema)
        >>> "minimum" in transformed["properties"]["age"]
        False
    """
    import copy

    transformed = copy.deepcopy(schema)

    def remove_unsupported_constraints(obj: Any) -> None:
        """Recursively remove unsupported constraints from schema."""
        if isinstance(obj, dict):
            # Remove numerical constraints
            obj.pop("minimum", None)
            obj.pop("maximum", None)
            obj.pop("multipleOf", None)

            # Remove string length constraints
            obj.pop("minLength", None)
            obj.pop("maxLength", None)

            # Remove array length constraints
            obj.pop("minItems", None)
            obj.pop("maxItems", None)

            # Add additionalProperties: false for ALL objects (Anthropic requirement)
            if obj.get("type") == "object" and "additionalProperties" not in obj:
                obj["additionalProperties"] = False

            # Recurse into nested objects
            for value in obj.values():
                remove_unsupported_constraints(value)
        elif isinstance(obj, list):
            for item in obj:
                remove_unsupported_constraints(item)

    # Remove constraints from entire schema
    remove_unsupported_constraints(transformed)

    return transformed


class AnthropicEntryDict(LLMConfigEntryDict, total=False):
    api_type: Literal["anthropic"]
    timeout: int | None
    stop_sequences: list[str] | None
    stream: bool
    price: list[float] | None
    tool_choice: dict | None
    thinking: dict | None
    gcp_project_id: str | None
    gcp_region: str | None
    gcp_auth_token: str | None


class AnthropicLLMConfigEntry(LLMConfigEntry):
    api_type: Literal["anthropic"] = "anthropic"

    # Basic options
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)

    # Anthropic-specific options
    timeout: int | None = Field(default=None, ge=1)
    top_k: int | None = Field(default=None, ge=1)
    stop_sequences: list[str] | None = None
    stream: bool = False
    price: list[float] | None = Field(default=None, min_length=2, max_length=2)
    tool_choice: dict | None = None
    thinking: dict | None = None

    gcp_project_id: str | None = None
    gcp_region: str | None = None
    gcp_auth_token: str | None = None

    def create_client(self):
        raise NotImplementedError("AnthropicLLMConfigEntry.create_client is not implemented.")


@require_optional_import("anthropic", "anthropic")
class AnthropicClient:
    RESPONSE_USAGE_KEYS: list[str] = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    def __init__(self, **kwargs: Unpack[AnthropicEntryDict]):
        """Initialize the Anthropic API client.

        Args:
            **kwargs: The configuration parameters for the client.
        """
        self._api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        self._aws_access_key = kwargs.get("aws_access_key") or os.getenv("AWS_ACCESS_KEY")
        self._aws_secret_key = kwargs.get("aws_secret_key") or os.getenv("AWS_SECRET_KEY")
        self._aws_session_token = kwargs.get("aws_session_token")
        self._aws_region = kwargs.get("aws_region") or os.getenv("AWS_REGION")
        self._gcp_project_id = kwargs.get("gcp_project_id")
        self._gcp_region = kwargs.get("gcp_region") or os.getenv("GCP_REGION")
        self._gcp_auth_token = kwargs.get("gcp_auth_token")
        self._base_url = kwargs.get("base_url")

        if self._api_key is None:
            if self._aws_region:
                if self._aws_access_key is None or self._aws_secret_key is None:
                    raise ValueError("API key or AWS credentials are required to use the Anthropic API.")
            elif self._gcp_region:
                if self._gcp_project_id is None or self._gcp_region is None:
                    raise ValueError("API key or GCP credentials are required to use the Anthropic API.")
            else:
                raise ValueError("API key or AWS credentials or GCP credentials are required to use the Anthropic API.")

        if self._api_key is not None:
            client_kwargs = {"api_key": self._api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            self._client = Anthropic(**client_kwargs)
        elif self._gcp_region is not None:
            kw = {}
            for p in inspect.signature(AnthropicVertex).parameters:
                if hasattr(self, f"_gcp_{p}"):
                    kw[p] = getattr(self, f"_gcp_{p}")
            if self._base_url:
                kw["base_url"] = self._base_url
            self._client = AnthropicVertex(**kw)
        else:
            client_kwargs = {
                "aws_access_key": self._aws_access_key,
                "aws_secret_key": self._aws_secret_key,
                "aws_session_token": self._aws_session_token,
                "aws_region": self._aws_region,
            }
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            self._client = AnthropicBedrock(**client_kwargs)

        self._last_tooluse_status = {}

        # Store the response format, if provided (for structured outputs)
        self._response_format: type[BaseModel] | dict | None = kwargs.get("response_format")

    def load_config(self, params: dict[str, Any]):
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
        """Prepare parameters for Anthropic API call.

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

    def _process_response_content(
        self, response: Message, is_native_structured_output: bool = False
    ) -> tuple[str, list[ChatCompletionMessageToolCall] | None, str]:
        """Process Anthropic response content into OpenAI-compatible format.

        Extracts tool calls, text content, and determines finish reason from the response.
        Handles both standard and native structured output responses.

        Args:
            response: Anthropic Message response object
            is_native_structured_output: Whether this is a native structured output response

        Returns:
            Tuple of (message_text, tool_calls, finish_reason)
        """
        tool_calls: list[ChatCompletionMessageToolCall] = []
        message_text = ""
        finish_reason = "stop"

        if response is None:
            return message_text, None, finish_reason

        # Determine finish reason
        if response.stop_reason == "tool_use":
            finish_reason = "tool_calls"

        # Process all content blocks
        thinking_content = ""
        text_content = ""

        for content in response.content:
            # Extract tool calls (handles both ToolUseBlock and BetaToolUseBlock)
            if _is_tool_use_block(content):
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=content.id,
                        function={"name": content.name, "arguments": json.dumps(content.input)},
                        type="function",
                    )
                )
            # Extract thinking content (extended thinking feature)
            elif _is_thinking_block(content):
                if thinking_content:
                    thinking_content += "\n\n"
                thinking_content += content.thinking
            # Extract text content (handles both TextBlock and BetaTextBlock)
            elif _is_text_block(content):
                # For native structured output, prefer parsed_output from parse() if available
                # Otherwise use the text content from the BetaTextBlock (from create() with dict schema)
                if (
                    is_native_structured_output
                    and hasattr(response, "parsed_output")
                    and response.parsed_output is not None
                ):
                    parsed_response = response.parsed_output
                    text_content = (
                        parsed_response.model_dump_json()
                        if hasattr(parsed_response, "model_dump_json")
                        else str(parsed_response)
                    )
                else:
                    # Use text content from BetaTextBlock (when using create() with dict schema)
                    # or regular TextBlock (non-SO responses)
                    if text_content:
                        text_content += "\n\n"
                    text_content += content.text

        # Combine thinking and text content
        if thinking_content and text_content:
            message_text = f"[Thinking]\n{thinking_content}\n\n{text_content}"
        elif thinking_content:
            message_text = f"[Thinking]\n{thinking_content}"
        elif text_content:
            message_text = text_content

        # Fallback: If using native SO parse() and no text was found in content blocks,
        # extract from parsed_output directly (if it's not None)
        if (
            not message_text
            and is_native_structured_output
            and hasattr(response, "parsed_output")
            and response.parsed_output is not None
        ):
            parsed_response = response.parsed_output
            message_text = (
                parsed_response.model_dump_json()
                if hasattr(parsed_response, "model_dump_json")
                else str(parsed_response)
            )

        return message_text, tool_calls if tool_calls else None, finish_reason

    def _log_structured_output_fallback(
        self,
        exception: Exception,
        model: str | None,
        response_format: Any,
        params: dict[str, Any],
    ) -> None:
        """Log detailed error information when native structured output fails and we fallback to JSON Mode.

        Consolidates error logging logic used in the create() method when native structured
        output encounters errors and needs to fall back to JSON Mode.

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
        if isinstance(exception, BadRequestError):
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

    def cost(self, response) -> float:
        """Calculate the cost of the completion using the Anthropic pricing."""
        return response.cost

    @property
    def api_key(self):
        return self._api_key

    @property
    def aws_access_key(self):
        return self._aws_access_key

    @property
    def aws_secret_key(self):
        return self._aws_secret_key

    @property
    def aws_session_token(self):
        return self._aws_session_token

    @property
    def aws_region(self):
        return self._aws_region

    @property
    def gcp_project_id(self):
        return self._gcp_project_id

    @property
    def gcp_region(self):
        return self._gcp_region

    @property
    def gcp_auth_token(self):
        return self._gcp_auth_token

    def create(self, params: dict[str, Any]) -> ChatCompletion:
        """Creates a completion using the Anthropic API.

        Automatically selects the best structured output method:
        - Native structured outputs for Claude Sonnet 4.5+ (guaranteed schema compliance)
        - JSON Mode for older models (prompt-based with <json_response> tags)
        - Standard completion for requests without response_format

        Args:
            params: Request parameters including model, messages, and optional response_format

        Returns:
            ChatCompletion object compatible with OpenAI format
        """
        model = params.get("model")
        response_format = params.get("response_format") or self._response_format

        # Route to appropriate implementation based on model and response_format
        if response_format:
            self._response_format = response_format
            params["response_format"] = response_format  # Ensure response_format is in params for methods

            # Try native structured outputs if model supports it
            if supports_native_structured_outputs(model) and has_beta_messages_api():
                try:
                    return self._create_with_native_structured_output(params)
                except (BadRequestError, AttributeError, ValueError) as e:
                    # Fallback to JSON Mode if native API not supported or schema invalid
                    # BadRequestError: Model doesn't support output_format
                    # AttributeError: SDK doesn't have beta API
                    # ValueError: Invalid schema format
                    self._log_structured_output_fallback(e, model, response_format, params)
                    return self._create_with_json_mode(params)
            else:
                # Use JSON Mode for older models or when beta API unavailable
                return self._create_with_json_mode(params)
        else:
            # Standard completion without structured outputs
            return self._create_standard(params)

    def _create_standard(self, params: dict[str, Any]) -> ChatCompletion:
        """Create a standard completion without structured outputs."""
        # Convert AG2 messages to Anthropic messages
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
            response = self._client.beta.messages.create(**anthropic_params)
        else:
            # Standard API for legacy tools
            response = self._client.messages.create(**anthropic_params)

        # Process response content using helper (extracts tool calls, text, finish reason)
        message_text, tool_calls, anthropic_finish = self._process_response_content(response)

        # Build and return ChatCompletion
        return self._build_chat_completion(response, message_text, tool_calls, anthropic_finish, anthropic_params)

    def _create_with_native_structured_output(self, params: dict[str, Any]) -> ChatCompletion:
        """Create completion using native structured outputs (beta API).

        This method uses Anthropic's beta structured outputs feature for guaranteed
        schema compliance via constrained decoding.

        Args:
            params: Request parameters

        Returns:
            ChatCompletion with structured JSON output

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
            response = self._client.beta.messages.create(**anthropic_params)
        else:
            # Pydantic model without tools - use parse() for automatic validation
            # parse() provides parsed_output attribute for direct model access
            anthropic_params["output_format"] = self._response_format
            response = self._client.beta.messages.parse(**anthropic_params)

        # Process response content using helper (extracts tool calls, text, finish reason)
        # Pass is_native_structured_output=True to handle parsed_output correctly
        message_text, tool_calls, anthropic_finish = self._process_response_content(
            response, is_native_structured_output=True
        )

        # Build and return ChatCompletion
        return self._build_chat_completion(response, message_text, tool_calls, anthropic_finish, anthropic_params)

    def _create_with_json_mode(self, params: dict[str, Any]) -> ChatCompletion:
        """Create completion using legacy JSON Mode with <json_response> tags.

        This method uses prompt-based structured outputs for older Claude models
        that don't support native structured outputs.

        Args:
            params: Request parameters

        Returns:
            ChatCompletion with JSON output extracted from tags
        """
        # Add response format instructions to system message before message conversion
        self._add_response_format_to_system(params)

        # Convert AG2 messages to Anthropic messages
        anthropic_messages = oai_messages_to_anthropic_messages(params)

        # Prepare Anthropic API parameters using helper
        anthropic_params = self._prepare_anthropic_params(params, anthropic_messages)

        # Call Anthropic API
        response = self._client.messages.create(**anthropic_params)

        # Extract JSON from <json_response> tags
        parsed_response = self._extract_json_response(response)
        # Keep as JSON - FormatterProtocol formatting will be applied in message_retrieval()
        message_text = (
            parsed_response.model_dump_json() if hasattr(parsed_response, "model_dump_json") else str(parsed_response)
        )

        # Build and return ChatCompletion
        return self._build_chat_completion(
            response, message_text, tool_calls=None, finish_reason="stop", anthropic_params=anthropic_params
        )

    def _build_chat_completion(
        self,
        response: Message,
        message_text: str,
        tool_calls: list[ChatCompletionMessageToolCall] | None,
        finish_reason: str,
        anthropic_params: dict[str, Any],
    ) -> ChatCompletion:
        """Build OpenAI-compatible ChatCompletion from Anthropic response.

        Args:
            response: Anthropic Message response
            message_text: Processed message content
            tool_calls: List of tool calls if any
            finish_reason: Completion finish reason
            anthropic_params: Original request parameters

        Returns:
            ChatCompletion object
        """
        # Calculate token usage
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens

        # Build message
        message = ChatCompletionMessage(
            role="assistant",
            content=message_text,
            function_call=None,
            tool_calls=tool_calls,
        )

        choices = [Choice(finish_reason=finish_reason, index=0, message=message)]

        # Build and return ChatCompletion
        return ChatCompletion(
            id=response.id,
            model=anthropic_params["model"],
            created=int(time.time()),
            object="chat.completion",
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            cost=_calculate_cost(prompt_tokens, completion_tokens, anthropic_params["model"]),
        )

    def message_retrieval(self, response) -> list[str] | list[ChatCompletionMessage]:
        """Retrieve and return a list of strings or a list of Choice.Message from the response.

        This method handles structured outputs with FormatterProtocol:
        - If tool/function calls present: returns full message objects
        - If structured output with format(): applies custom formatting
        - Otherwise: returns content as-is

        NOTE: if a list of Choice.Message is returned, it currently needs to contain the fields of OpenAI's ChatCompletion Message object,
        since that is expected for function or tool calling in the rest of the codebase at the moment, unless a custom agent is being used.
        """
        choices = response.choices

        def _format_content(content: str | list[dict[str, Any]] | None) -> str:
            """Format content using FormatterProtocol if available."""
            normalized_content = content_str(content)  # type: ignore [arg-type]
            # If response_format implements FormatterProtocol (has format() method), use it
            if isinstance(self._response_format, FormatterProtocol):
                try:
                    return self._response_format.model_validate_json(normalized_content).format()  # type: ignore [union-attr]
                except Exception:
                    # If parsing fails (e.g., content is error message), return as-is
                    return normalized_content
            else:
                return normalized_content

        # Handle tool/function calls - return full message object
        if TOOL_ENABLED:
            return [  # type: ignore [return-value]
                (choice.message if choice.message.tool_calls is not None else _format_content(choice.message.content))
                for choice in choices
            ]
        else:
            return [_format_content(choice.message.content) for choice in choices]  # type: ignore [return-value]

    @staticmethod
    def openai_func_to_anthropic(openai_func: dict) -> dict:
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
    def get_usage(response: ChatCompletion) -> dict:
        """Get the usage of tokens and their cost information."""
        return {
            "prompt_tokens": response.usage.prompt_tokens if response.usage is not None else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage is not None else 0,
            "total_tokens": response.usage.total_tokens if response.usage is not None else 0,
            "cost": response.cost if hasattr(response, "cost") else 0.0,
            "model": response.model,
        }

    @staticmethod
    def convert_tools_to_functions(tools: list) -> list:
        """Convert tool definitions into Anthropic-compatible functions,
        updating nested $ref paths in property schemas.

        Args:
            tools (list): List of tool definitions.

        Returns:
            list: List of functions with updated $ref paths.
        """

        def update_refs(obj, defs_keys, prop_name):
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

    def _add_response_format_to_system(self, params: dict[str, Any]):
        """Add prompt that will generate properly formatted JSON for structured outputs to system parameter.

        Based on Anthropic's JSON Mode cookbook, we ask the LLM to put the JSON within <json_response> tags.

        Args:
            params (dict): The client parameters
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

    def _extract_json_response(self, response: Message) -> Any:
        """Extract and validate JSON response from the output for structured outputs.

        Args:
            response (Message): The response from the API.

        Returns:
            Any: The parsed JSON response.
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


def _format_json_response(response: Any) -> str:
    """Formats the JSON response for structured outputs using the format method if it exists."""
    if isinstance(response, str):
        return response
    elif isinstance(response, FormatterProtocol):
        return response.format()
    else:
        return response.model_dump_json()


def process_image_content(content_item: dict[str, Any]) -> dict[str, Any]:
    """Process an OpenAI image content item into Claude format."""
    if content_item["type"] != "image_url":
        return content_item

    url = content_item["image_url"]["url"]
    try:
        # Handle data URLs
        if url.startswith("data:"):
            data_url_pattern = r"data:image/([a-zA-Z]+);base64,(.+)"
            match = re.match(data_url_pattern, url)
            if match:
                media_type, base64_data = match.groups()
                return {
                    "type": "image",
                    "source": {"type": "base64", "media_type": f"image/{media_type}", "data": base64_data},
                }

        else:
            print("Error processing image.")
            # Return original content if image processing fails
            return content_item

    except Exception as e:
        print(f"Error processing image image: {e}")
        # Return original content if image processing fails
        return content_item


def process_message_content(message: dict[str, Any]) -> str | list[dict[str, Any]]:
    """Process message content, handling both string and list formats with images."""
    content = message.get("content", "")

    # Handle empty content
    if content == "":
        return content

    # If content is already a string, return as is
    if isinstance(content, str):
        return content

    # Handle list content (mixed text and images)
    if isinstance(content, list):
        processed_content = []
        for item in content:
            if item["type"] == "text":
                processed_content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image_url":
                processed_content.append(process_image_content(item))
        return processed_content

    return content


def _extract_system_message(message: dict[str, Any], params: dict[str, Any]) -> None:
    """Extract system message content and add to params['system'].

    System messages are handled specially in Anthropic API - they're passed as a separate
    'system' parameter rather than in the messages list.

    Args:
        message: Message dict with role='system'
        params: Params dict to update with system content (modified in place)
    """
    content = process_message_content(message)
    if isinstance(content, list):
        # For system messages with images, concatenate only the text portions
        text_content = " ".join(item.get("text", "") for item in content if item.get("type") == "text")
        params["system"] = params.get("system", "") + (" " if "system" in params else "") + text_content
    else:
        params["system"] = params.get("system", "") + ("\n" if "system" in params else "") + content


def _convert_tool_call_message(
    message: dict[str, Any],
    has_tools: bool,
    expected_role: str,
    user_continue_message: dict[str, str],
    processed_messages: list[dict[str, Any]],
) -> tuple[int, int | None]:
    """Convert OpenAI tool_calls format to Anthropic ToolUseBlock format.

    Args:
        message: Message dict containing 'tool_calls'
        has_tools: Whether tools parameter is present in request
        expected_role: Expected role based on message alternation ("user" or "assistant")
        user_continue_message: Standard continue message for role alternation
        processed_messages: List to append converted messages to (modified in place)

    Returns:
        Tuple of (tool_use_messages_count, last_tool_use_index)
        - tool_use_messages_count: Number of tool use messages added (0 or count)
        - last_tool_use_index: Index of last tool use message, or None if not using tools
    """
    # Map the tool call options to Anthropic's ToolUseBlock
    tool_uses = []
    tool_names = []
    tool_use_count = 0

    for tool_call in message["tool_calls"]:
        tool_uses.append(
            ToolUseBlock(
                type="tool_use",
                id=tool_call["id"],
                name=tool_call["function"]["name"],
                input=json.loads(tool_call["function"]["arguments"]),
            )
        )
        if has_tools:
            tool_use_count += 1
        tool_names.append(tool_call["function"]["name"])

    # Ensure role alternation: if we expect user, insert user continue message
    if expected_role == "user":
        processed_messages.append(user_continue_message)

    # Add tool use message (format depends on whether tools are enabled)
    if has_tools:
        processed_messages.append({"role": "assistant", "content": tool_uses})
        last_tool_use_index = len(processed_messages) - 1
        return tool_use_count, last_tool_use_index
    else:
        # Not using tools, so put in a plain text message
        processed_messages.append({
            "role": "assistant",
            "content": f"Some internal function(s) that could be used: [{', '.join(tool_names)}]",
        })
        return 0, None


def _convert_tool_result_message(
    message: dict[str, Any],
    has_tools: bool,
    expected_role: str,
    assistant_continue_message: dict[str, str],
    processed_messages: list[dict[str, Any]],
    last_tool_result_index: int,
) -> tuple[int, int]:
    """Convert OpenAI tool result format to Anthropic tool_result format.

    Args:
        message: Message dict containing 'tool_call_id'
        has_tools: Whether tools parameter is present in request
        expected_role: Expected role based on message alternation ("user" or "assistant")
        assistant_continue_message: Standard continue message for role alternation
        processed_messages: List to append converted messages to (modified in place)
        last_tool_result_index: Index of last tool result message (-1 if none)

    Returns:
        Tuple of (tool_result_messages_count, updated_last_tool_result_index)
        - tool_result_messages_count: 1 if tool result added, 0 otherwise
        - updated_last_tool_result_index: New index of last tool result message
    """
    if has_tools:
        # Map the tool usage call to tool_result for Anthropic
        tool_result = {
            "type": "tool_result",
            "tool_use_id": message["tool_call_id"],
            "content": message["content"],
        }

        # If the previous message also had a tool_result, add it to that
        # Otherwise append a new message
        if last_tool_result_index == len(processed_messages) - 1:
            processed_messages[-1]["content"].append(tool_result)
        else:
            if expected_role == "assistant":
                # Insert an extra assistant message as we will append a user message
                processed_messages.append(assistant_continue_message)

            processed_messages.append({"role": "user", "content": [tool_result]})
            last_tool_result_index = len(processed_messages) - 1

        return 1, last_tool_result_index
    else:
        # Not using tools, so put in a plain text message
        processed_messages.append({
            "role": "user",
            "content": f"Running the function returned: {message['content']}",
        })
        return 0, last_tool_result_index


@require_optional_import("anthropic", "anthropic")
def oai_messages_to_anthropic_messages(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert messages from OAI format to Anthropic format.
    We correct for any specific role orders and types, etc.
    """
    # Track whether we have tools passed in. If not,  tool use / result messages should be converted to text messages.
    # Anthropic requires a tools parameter with the tools listed, if there are other messages with tool use or tool results.
    # This can occur when we don't need tool calling, such as for group chat speaker selection.
    has_tools = "tools" in params

    # Convert messages to Anthropic compliant format
    processed_messages = []

    # Used to interweave user messages to ensure user/assistant alternating
    user_continue_message = {"content": "Please continue.", "role": "user"}
    assistant_continue_message = {"content": "Please continue.", "role": "assistant"}

    tool_use_messages = 0
    tool_result_messages = 0
    last_tool_use_index = -1
    last_tool_result_index = -1
    for message in params["messages"]:
        if message["role"] == "system":
            _extract_system_message(message, params)
        else:
            # New messages will be added here, manage role alternations
            expected_role = "user" if len(processed_messages) % 2 == 0 else "assistant"

            if "tool_calls" in message:
                # Convert OpenAI tool_calls to Anthropic ToolUseBlock format
                count, index = _convert_tool_call_message(
                    message, has_tools, expected_role, user_continue_message, processed_messages
                )
                tool_use_messages += count
                if index is not None:
                    last_tool_use_index = index
            elif "tool_call_id" in message:
                # Convert OpenAI tool result to Anthropic tool_result format
                count, last_tool_result_index = _convert_tool_result_message(
                    message,
                    has_tools,
                    expected_role,
                    assistant_continue_message,
                    processed_messages,
                    last_tool_result_index,
                )
                tool_result_messages += count
            elif message["content"] == "":
                # Ignoring empty messages
                pass
            else:
                if expected_role != message["role"]:
                    # Inserting the alternating continue message
                    processed_messages.append(
                        user_continue_message if expected_role == "user" else assistant_continue_message
                    )
                # Process messages for images
                processed_content = process_message_content(message)
                processed_message = message.copy()
                processed_message["content"] = processed_content
                processed_messages.append(processed_message)

    # We'll replace the last tool_use if there's no tool_result (occurs if we finish the conversation before running the function)
    if has_tools and tool_use_messages != tool_result_messages:
        processed_messages[last_tool_use_index] = assistant_continue_message

    # name is not a valid field on messages
    for message in processed_messages:
        if "name" in message:
            message.pop("name", None)

    # Note: When using reflection_with_llm we may end up with an "assistant" message as the last message and that may cause a blank response
    # So, if the last role is not user, add a 'user' continue message at the end
    if processed_messages[-1]["role"] != "user":
        processed_messages.append(user_continue_message)

    return processed_messages


def _calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate the cost of the completion using the Anthropic pricing."""
    total = 0.0

    if model in ANTHROPIC_PRICING_1k:
        input_cost_per_1k, output_cost_per_1k = ANTHROPIC_PRICING_1k[model]
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        total = input_cost + output_cost
    else:
        warnings.warn(f"Cost calculation not available for model {model}", UserWarning)

    return total
