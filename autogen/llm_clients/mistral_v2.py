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
from .client_utils import should_hide_tools, validate_parameter
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
        # TODO: Implement in second commit
        raise NotImplementedError("create() method will be implemented in second commit")

    def _transform_response(self, mistral_response: Any, model: str) -> UnifiedResponse:
        """Transform Mistral API response to UnifiedResponse.

        Args:
            mistral_response: Raw Mistral API response
            model: Model name

        Returns:
            UnifiedResponse with all content blocks properly typed
        """
        # TODO: Implement in second commit
        raise NotImplementedError("_transform_response() method will be implemented in second commit")

    def parse_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Loads the parameters for Mistral.AI API from the passed in parameters and returns a validated set.

        Checks types, ranges, and sets defaults.

        Args:
            params: Request parameters dict

        Returns:
            Validated Mistral API parameters dict
        """
        # TODO: Implement in second commit (can reuse logic from v1)
        raise NotImplementedError("parse_params() method will be implemented in second commit")

    def create_v1_compatible(self, params: dict[str, Any]) -> Any:
        """Create completion in backward-compatible ChatCompletion format.

        This method provides compatibility with existing AG2 code that expects
        ChatCompletionExtended format.

        Args:
            params: Same parameters as create()

        Returns:
            ChatCompletion-compatible dict (flattened response)
        """
        # TODO: Implement in second commit
        raise NotImplementedError("create_v1_compatible() method will be implemented in second commit")

    def cost(self, response: UnifiedResponse) -> float:  # type: ignore[override]
        """Calculate cost from response usage.

        Implements ModelClient.cost() but accepts UnifiedResponse via duck typing.

        Args:
            response: UnifiedResponse with usage information

        Returns:
            Cost in USD for the API call
        """
        # TODO: Implement in second commit
        raise NotImplementedError("cost() method will be implemented in second commit")

    @staticmethod
    def get_usage(response: UnifiedResponse) -> dict[str, Any]:  # type: ignore[override]
        """Extract usage statistics from response.

        Implements ModelClient.get_usage() but accepts UnifiedResponse via duck typing.

        Args:
            response: UnifiedResponse from create()

        Returns:
            Dict with keys from RESPONSE_USAGE_KEYS
        """
        # TODO: Implement in second commit
        raise NotImplementedError("get_usage() method will be implemented in second commit")

    def message_retrieval(self, response: UnifiedResponse) -> list[str] | list[dict[str, Any]]:  # type: ignore[override]
        """Retrieve messages from response in OpenAI-compatible format.

        Returns list of strings for text-only messages, or list of dicts when
        tool calls or complex content is present.

        Args:
            response: UnifiedResponse from create()

        Returns:
            List of strings (for text-only) OR list of message dicts (for tool calls/complex content)
        """
        # TODO: Implement in second commit
        raise NotImplementedError("message_retrieval() method will be implemented in second commit")


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
