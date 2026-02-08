# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
"""Create a ModelClientV2-compatible client using Mistral.AI's API.

This client implements the ModelClientV2 protocol, returning rich UnifiedResponse
objects while maintaining backward compatibility with the legacy ModelClient interface.

Example:
    ```python
    llm_config = {
        "config_list": [
            {"api_type": "mistral", "model": "open-mixtral-8x22b", "api_key": os.environ.get("MISTRAL_API_KEY")}
        ]
    }

    agent = autogen.AssistantAgent("my_agent", llm_config=llm_config)
    ```

Install Mistral.AI python library using: pip install --upgrade mistralai

Resources:
- https://docs.mistral.ai/getting-started/quickstart/

NOTE: Requires mistralai package version >= 1.0.1
"""

import json
import os
import time
import warnings
from typing import Any, Literal

from typing_extensions import Unpack

from ..import_utils import optional_import_block, require_optional_import
from ..llm_config.client import ModelClient
from ..llm_config.entry import LLMConfigEntry, LLMConfigEntryDict
from ..oai.client_utils import should_hide_tools, validate_parameter
from .models import (
    TextContent,
    ToolCallContent,
    UnifiedMessage,
    UnifiedResponse,
    UserRoleEnum,
    normalize_role,
)

with optional_import_block():
    # Mistral libraries
    # pip install mistralai
    from mistralai import (
        AssistantMessage,
        Function,
        FunctionCall,
        Mistral,
        SystemMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )


class MistralEntryDict(LLMConfigEntryDict, total=False):
    api_type: Literal["mistral"]

    safe_prompt: bool
    random_seed: int | None
    stream: bool
    hide_tools: Literal["if_all_run", "if_any_run", "never"]
    tool_choice: Literal["none", "auto", "any"] | None


class MistralLLMConfigEntry(LLMConfigEntry):
    api_type: Literal["mistral"] = "mistral"
    safe_prompt: bool = False
    random_seed: int | None = None
    stream: bool = False
    hide_tools: Literal["if_all_run", "if_any_run", "never"] = "never"
    tool_choice: Literal["none", "auto", "any"] | None = None

    def create_client(self):
        raise NotImplementedError("MistralLLMConfigEntry.create_client is not implemented.")


@require_optional_import("mistralai", "mistral")
class MistralAIClientV2(ModelClient):
    """Client for Mistral.AI's API implementing ModelClientV2 protocol.

    This client returns rich UnifiedResponse objects with typed content blocks
    while maintaining backward compatibility with the legacy ModelClient interface.

    Key Features:
    - Returns UnifiedResponse with TextContent and ToolCallContent blocks
    - Supports tool calls and function execution
    - Provides backward compatibility via create_v1_compatible()
    - Maintains V1 compatibility via message_retrieval()
    """

    RESPONSE_USAGE_KEYS: list[str] = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    def __init__(self, **kwargs: Unpack[MistralEntryDict]):
        """Requires api_key or environment variable to be set

        Args:
            **kwargs: Additional keyword arguments to pass to the Mistral client.
        """
        # Ensure we have the api_key upon instantiation
        self.api_key = kwargs.get("api_key")
        if not self.api_key:
            self.api_key = os.getenv("MISTRAL_API_KEY", None)

        assert self.api_key, (
            "Please specify the 'api_key' in your config list entry for Mistral or set the MISTRAL_API_KEY env variable."
        )

        if "response_format" in kwargs and kwargs["response_format"] is not None:
            warnings.warn("response_format is not supported for Mistral.AI, it will be ignored.", UserWarning)

        self._client = Mistral(api_key=self.api_key)

    def create(self, params: dict[str, Any]) -> UnifiedResponse:  # type: ignore[override]
        """Create a completion and return UnifiedResponse with all features preserved.

        This method implements ModelClient.create() but returns UnifiedResponse instead
        of ModelClientResponseProtocol. The rich UnifiedResponse structure is compatible
        via duck typing - it has .model attribute and works with message_retrieval().

        Args:
            params: Request parameters including:
                - model: Model name (e.g., "open-mixtral-8x22b")
                - messages: List of message dicts
                - temperature: Optional temperature
                - max_tokens: Optional max completion tokens
                - tools: Optional tool definitions
                - **other Mistral parameters

        Returns:
            UnifiedResponse with text content and tool calls preserved
        """
        # 1. Parse parameters to Mistral.AI API's parameters
        mistral_params = self.parse_params(params)

        # 2. Call Mistral.AI API
        mistral_response = self._client.chat.complete(**mistral_params)
        # TODO: Handle streaming

        # 3. Transform Mistral response to UnifiedResponse
        return self._transform_response(mistral_response, mistral_params.get("model", params.get("model", "unknown")))

    def _transform_response(self, mistral_response: Any, model: str) -> UnifiedResponse:
        """Transform Mistral API response to UnifiedResponse.

        This handles the standard Mistral chat completion format including:
        - Text content → TextContent
        - Tool calls → ToolCallContent
        - Usage information and cost calculation

        Args:
            mistral_response: Raw Mistral API response
            model: Model name

        Returns:
            UnifiedResponse with all content blocks properly typed
        """
        messages = []

        # Process each choice
        for choice in mistral_response.choices:
            content_blocks = []
            message_obj = choice.message

            # Extract text content
            if message_obj.content:
                content_blocks.append(TextContent(text=message_obj.content))

            # Extract tool calls
            if getattr(message_obj, "tool_calls", None):
                for tool_call in message_obj.tool_calls:
                    # Mistral tool calls have function.arguments as dict, need to convert to JSON string
                    arguments_str = (
                        json.dumps(tool_call.function.arguments)
                        if isinstance(tool_call.function.arguments, dict)
                        else tool_call.function.arguments
                    )
                    content_blocks.append(
                        ToolCallContent(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=arguments_str,
                        )
                    )

            # Create unified message with normalized role
            messages.append(
                UnifiedMessage(
                    role=normalize_role(getattr(message_obj, "role", "assistant")),
                    content=content_blocks,
                )
            )

        # Extract usage information
        usage = {}
        if getattr(mistral_response, "usage", None):
            usage = {
                "prompt_tokens": mistral_response.usage.prompt_tokens,
                "completion_tokens": mistral_response.usage.completion_tokens,
                "total_tokens": mistral_response.usage.prompt_tokens + mistral_response.usage.completion_tokens,
            }

        # Determine finish reason
        finish_reason = None
        if mistral_response.choices:
            finish_reason = mistral_response.choices[0].finish_reason

        # Build UnifiedResponse
        unified_response = UnifiedResponse(
            id=mistral_response.id,
            model=mistral_response.model if hasattr(mistral_response, "model") else model,
            provider="mistral",
            messages=messages,
            usage=usage,
            finish_reason=finish_reason,
            status="completed",
            provider_metadata={
                "created": getattr(mistral_response, "created", None),
            },
        )

        # Calculate cost
        if usage:
            unified_response.cost = calculate_mistral_cost(
                usage["prompt_tokens"],
                usage["completion_tokens"],
                unified_response.model,
            )

        return unified_response

    @require_optional_import("mistralai", "mistral")
    def parse_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Loads the parameters for Mistral.AI API from the passed in parameters and returns a validated set.

        Checks types, ranges, and sets defaults.

        Args:
            params: Request parameters dict

        Returns:
            Validated Mistral API parameters dict
        """
        mistral_params = {}

        # 1. Validate models
        mistral_params["model"] = params.get("model")
        assert mistral_params["model"], (
            "Please specify the 'model' in your config list entry to nominate the Mistral.ai model to use."
        )

        # 2. Validate allowed Mistral.AI parameters
        mistral_params["temperature"] = validate_parameter(params, "temperature", (int, float), True, 0.7, None, None)
        mistral_params["top_p"] = validate_parameter(params, "top_p", (int, float), True, None, None, None)
        mistral_params["max_tokens"] = validate_parameter(params, "max_tokens", int, True, None, (0, None), None)
        mistral_params["safe_prompt"] = validate_parameter(
            params, "safe_prompt", bool, False, False, None, [True, False]
        )
        mistral_params["random_seed"] = validate_parameter(params, "random_seed", int, True, None, False, None)
        mistral_params["tool_choice"] = validate_parameter(
            params, "tool_choice", str, False, None, None, ["none", "auto", "any"]
        )

        # TODO: Handle streaming
        if params.get("stream", False):
            warnings.warn(
                "Streaming is not currently supported, streaming will be disabled.",
                UserWarning,
            )

        # 3. Convert messages to Mistral format
        mistral_messages = []
        tool_call_ids = {}  # tool call ids to function name mapping
        for message in params["messages"]:
            if message["role"] == "assistant" and "tool_calls" in message and message["tool_calls"] is not None:
                # Convert OAI ToolCall to Mistral ToolCall
                mistral_messages_tools = []
                for toolcall in message["tool_calls"]:
                    mistral_messages_tools.append(
                        ToolCall(
                            id=toolcall["id"],
                            function=FunctionCall(
                                name=toolcall["function"]["name"],
                                arguments=json.loads(toolcall["function"]["arguments"]),
                            ),
                        )
                    )

                mistral_messages.append(AssistantMessage(content="", tool_calls=mistral_messages_tools))

                # Map tool call id to the function name
                for tool_call in message["tool_calls"]:
                    tool_call_ids[tool_call["id"]] = tool_call["function"]["name"]

            elif message["role"] == "system":
                if len(mistral_messages) > 0 and mistral_messages[-1].role == "assistant":
                    # System messages can't appear after an Assistant message, so use a UserMessage
                    mistral_messages.append(UserMessage(content=message["content"]))
                else:
                    mistral_messages.append(SystemMessage(content=message["content"]))
            elif message["role"] == "assistant":
                mistral_messages.append(AssistantMessage(content=message["content"]))
            elif message["role"] == "user":
                mistral_messages.append(UserMessage(content=message["content"]))

            elif message["role"] == "tool":
                # Indicates the result of a tool call, the name is the function name called
                mistral_messages.append(
                    ToolMessage(
                        name=tool_call_ids[message["tool_call_id"]],
                        content=message["content"],
                        tool_call_id=message["tool_call_id"],
                    )
                )
            else:
                warnings.warn(f"Unknown message role {message['role']}", UserWarning)

        # 4. Last message needs to be user or tool, if not, add a "please continue" message
        if not isinstance(mistral_messages[-1], UserMessage) and not isinstance(mistral_messages[-1], ToolMessage):
            mistral_messages.append(UserMessage(content="Please continue."))

        mistral_params["messages"] = mistral_messages

        # 5. Add tools to the call if we have them and aren't hiding them
        if "tools" in params:
            hide_tools = validate_parameter(
                params, "hide_tools", str, False, "never", None, ["if_all_run", "if_any_run", "never"]
            )
            if not should_hide_tools(params["messages"], params["tools"], hide_tools):
                mistral_params["tools"] = tool_def_to_mistral(params["tools"])

        return mistral_params

    def create_v1_compatible(self, params: dict[str, Any]) -> Any:
        """Create completion in backward-compatible ChatCompletion format.

        This method provides compatibility with existing AG2 code that expects
        ChatCompletionExtended format.

        Args:
            params: Same parameters as create()

        Returns:
            ChatCompletion-compatible dict (flattened response)
        """
        # Get rich response
        unified_response = self.create(params)

        # Extract role and convert UserRoleEnum to string
        role = unified_response.messages[0].role if unified_response.messages else UserRoleEnum.ASSISTANT
        role_str = role.value if isinstance(role, UserRoleEnum) else role

        # Extract text content (flatten all content blocks)
        text_content = unified_response.text

        # Extract tool calls if present
        tool_calls = None
        if unified_response.messages:
            message = unified_response.messages[0]
            tool_call_blocks = message.get_tool_calls()
            if tool_call_blocks:
                tool_calls = []
                for tc in tool_call_blocks:
                    # Convert ToolCallContent to OpenAI format
                    # Arguments are already stored as JSON string in ToolCallContent
                    tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,  # Already JSON string from _transform_response
                        },
                    })

        # Determine finish reason
        finish_reason = unified_response.finish_reason or "stop"
        if tool_calls:
            finish_reason = "tool_calls"

        # Build ChatCompletion-like dict
        return {
            "id": unified_response.id,
            "model": unified_response.model,
            "created": unified_response.provider_metadata.get("created", int(time.time())),
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": role_str,
                        "content": text_content,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": unified_response.usage.get("prompt_tokens", 0),
                "completion_tokens": unified_response.usage.get("completion_tokens", 0),
                "total_tokens": unified_response.usage.get("total_tokens", 0),
            },
            "cost": unified_response.cost or 0.0,
        }

    def cost(self, response: UnifiedResponse) -> float:  # type: ignore[override]
        """Calculate cost from response usage.

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
        model = response.model

        return calculate_mistral_cost(prompt_tokens, completion_tokens, model)

    @staticmethod
    def get_usage(response: UnifiedResponse) -> dict[str, Any]:  # type: ignore[override]
        """Extract usage statistics from response.

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
        """Retrieve messages from response in OpenAI-compatible format.

        Returns list of strings for text-only messages, or list of dicts when
        tool calls or complex content is present.

        This matches the behavior of the legacy MistralAIClient which returns:
        - ChatCompletionMessage objects (as dicts) for all messages
        - Tool calls are preserved in the message dict

        Args:
            response: UnifiedResponse from create()

        Returns:
            List of strings (for text-only) OR list of message dicts (for tool calls/complex content)
        """
        result: list[str] | list[dict[str, Any]] = []

        for msg in response.messages:
            # Check for tool calls
            tool_calls = msg.get_tool_calls()

            if tool_calls:
                # Return OpenAI-compatible dict format when tool calls are present
                message_dict: dict[str, Any] = {
                    "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                    "content": msg.get_text() or None,
                }

                # Add optional fields
                if msg.name:
                    message_dict["name"] = msg.name

                # Add tool calls in OpenAI format
                tool_calls_list = []
                for tc in tool_calls:
                    # Convert ToolCallContent to OpenAI format
                    # Arguments are already stored as JSON string in ToolCallContent
                    tool_calls_list.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,  # Already JSON string from _transform_response
                        },
                    })

                message_dict["tool_calls"] = tool_calls_list
                result.append(message_dict)
            else:
                # Simple text content - return string
                # Note: V1 MistralAIClient returns ChatCompletionMessage objects, but for compatibility
                # we return strings for text-only to match OpenAI behavior
                text = msg.get_text()
                result.append(text)

        return result


@require_optional_import("mistralai", "mistral")
def tool_def_to_mistral(tool_definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Converts AG2 tool definition to a mistral tool format"""
    mistral_tools = []

    for autogen_tool in tool_definitions:
        mistral_tool = {
            "type": "function",
            "function": Function(
                name=autogen_tool["function"]["name"],
                description=autogen_tool["function"]["description"],
                parameters=autogen_tool["function"]["parameters"],
            ),
        }

        mistral_tools.append(mistral_tool)

    return mistral_tools


def calculate_mistral_cost(input_tokens: int, output_tokens: int, model_name: str) -> float:
    """Calculate the cost of the mistral response."""
    # Prices per 1 thousand tokens
    # https://mistral.ai/technology/
    model_cost_map = {
        "open-mistral-7b": {"input": 0.00025, "output": 0.00025},
        "open-mixtral-8x7b": {"input": 0.0007, "output": 0.0007},
        "open-mixtral-8x22b": {"input": 0.002, "output": 0.006},
        "mistral-small-latest": {"input": 0.001, "output": 0.003},
        "mistral-medium-latest": {"input": 0.00275, "output": 0.0081},
        "mistral-large-latest": {"input": 0.0003, "output": 0.0003},
        "mistral-large-2407": {"input": 0.0003, "output": 0.0003},
        "open-mistral-nemo-2407": {"input": 0.0003, "output": 0.0003},
        "codestral-2405": {"input": 0.001, "output": 0.003},
    }

    # Ensure we have the model they are using and return the total cost
    if model_name in model_cost_map:
        costs = model_cost_map[model_name]

        return (input_tokens * costs["input"] / 1000) + (output_tokens * costs["output"] / 1000)
    else:
        warnings.warn(f"Cost calculation is not implemented for model {model_name}, will return $0.", UserWarning)
        return 0
