# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
OpenAI Responses API V2 Client implementing ModelClientV2 and ModelClient protocols.

This client handles the OpenAI Responses API (client.responses.create) which is a
stateful API that supports:
- Stateful conversations via previous_response_id
- Built-in tools (web_search, image_generation, apply_patch)
- Structured outputs via text_format parameter
- Multimodal content (text, images)
- Reasoning blocks (o3 models)

The client returns rich UnifiedResponse objects with typed content blocks while
maintaining backward compatibility with the ModelClient protocol.

Note: This uses the Responses API (/responses endpoint), NOT Chat Completions API.
"""

import logging
import os
from typing import TYPE_CHECKING, Any

from autogen.code_utils import content_str
from autogen.import_utils import optional_import_block

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.responses.response import Response
else:
    OpenAI = None
    Response = None

with optional_import_block() as openai_result:
    from openai import OpenAI

if openai_result.is_successful:
    openai_import_exception: ImportError | None = None
else:
    OpenAI = None  # type: ignore[assignment]
    openai_import_exception = ImportError(
        "Please install openai to use OpenAIResponsesV2Client. Install with: pip install openai"
    )

from ..llm_config.client import ModelClient
from .models import (
    GenericContent,
    ReasoningContent,
    TextContent,
    ToolCallContent,
    UnifiedMessage,
    UnifiedResponse,
)

logger = logging.getLogger(__name__)


class OpenAIResponsesV2Client(ModelClient):
    """
    OpenAI Responses API V2 client implementing ModelClientV2 protocol.

    This client works with OpenAI's Responses API (client.responses.create) which
    is a stateful API supporting built-in tools, structured outputs, and rich content.

    Key Features:
    - Stateful conversations via previous_response_id tracking
    - Built-in tools (web_search, image_generation, apply_patch)
    - Structured outputs via text_format parameter
    - Multimodal content support
    - Reasoning blocks (o3 models)
    - Returns UnifiedResponse with typed content blocks
    - Backward compatibility via create_v1_compatible()

    Example:
        client = OpenAIResponsesV2Client(api_key="...")

        # Get rich response with stateful conversation
        response = client.create({
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "Hello"}],
            "built_in_tools": ["web_search"]
        })

        # Access text response
        print(f"Answer: {response.text}")

        # Continue conversation (stateful)
        response2 = client.create({
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "Tell me more"}]
        })
    """

    RESPONSE_USAGE_KEYS: list[str] = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        response_format: Any = None,
        workspace_dir: str | None = None,
        allowed_paths: list[str] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize OpenAI Responses API V2 client.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            base_url: Custom base URL for OpenAI API
            timeout: Request timeout in seconds
            response_format: Optional response format (Pydantic model or JSON schema)
            workspace_dir: Workspace directory for apply_patch operations
            allowed_paths: Allowed paths for apply_patch operations
            **kwargs: Additional arguments passed to OpenAI client
        """
        if openai_import_exception is not None:
            raise openai_import_exception

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)  # type: ignore[misc]
        self._default_response_format = response_format

        # Stateful conversation management
        self._previous_response_id: str | None = None

        # Workspace configuration for apply_patch tool
        self._workspace_dir = workspace_dir or os.getcwd()
        self._allowed_paths = allowed_paths or ["**"]

        # Image generation parameters (for future use)
        self.image_output_params = {
            "quality": None,  # "high" or "low"
            "background": None,  # "white" or "black" or "transparent"
            "size": None,  # "1024x1024" or "1024x1792" or "1792x1024"
            "output_format": "png",  # "png", "jpg" or "jpeg" or "webp"
            "output_compression": None,  # 0-100 if output_format is "jpg" or "jpeg" or "webp"
        }


    def _get_previous_response_id(self) -> str | None:
        """Get current conversation state.

        Returns:
            Previous response ID if conversation is in progress, None otherwise
        """
        return self._previous_response_id

    def _set_previous_response_id(self, response_id: str | None) -> None:
        """Update conversation state.

        Args:
            response_id: Response ID to set as previous response, or None to clear state
        """
        self._previous_response_id = response_id

    def reset_conversation(self) -> None:
        """Reset conversation state (start new conversation).

        This clears the previous_response_id, effectively starting a new conversation
        thread. Useful when switching between different conversation contexts.
        """
        self._previous_response_id = None


    def _get_delta_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Get the delta messages from the messages.

        This method extracts only the new messages since the last completed response,
        which is important for stateful conversations in the Responses API.

        Args:
            messages: List of message dicts

        Returns:
            List of delta messages (new messages since last completed response)
        """
        delta_messages = []
        for m in messages[::-1]:
            contents = m.get("content")
            is_last_completed_response = False
            has_apply_patch_call = False

            if isinstance(contents, list):
                for c in contents:
                    # Check if message contains apply_patch_call items
                    if isinstance(c, dict) and c.get("type") == "apply_patch_call":
                        has_apply_patch_call = True
                        continue  # Skip status check for apply_patch_call items
                    if "status" in c and c.get("status") == "completed":
                        is_last_completed_response = True
                        break
            elif isinstance(contents, str):
                is_last_completed_response = "status" in m and m.get("status") == "completed"

            # Don't break if message contains apply_patch_call items - they need to be processed
            if is_last_completed_response and not has_apply_patch_call:
                break
            delta_messages.append(m)
        return delta_messages[::-1]

    def _convert_messages_to_input(
        self,
        messages: list[dict[str, Any]],
        processed_apply_patch_call_ids: set[str],
        image_generation_tool_params: dict[str, Any],
        input_items: list[dict[str, Any]],
    ) -> None:
        """Convert messages to Responses API format and append to input_items.

        This method converts standard message format to the Responses API input format,
        which uses content blocks (input_text, output_text, input_image, etc.).

        Args:
            messages: List of messages to convert
            processed_apply_patch_call_ids: Set of call_ids that have been processed
            image_generation_tool_params: Image generation tool parameters dict (modified in place)
            input_items: List to append converted message items to (modified in place)
        """
        for m in messages[::-1]:  # reverse the list to get the last item first
            role = m.get("role", "user")
            content = m.get("content")
            blocks = []

            if role != "tool":
                content_type = "output_text" if role == "assistant" else "input_text"
                if isinstance(content, list):
                    for c in content:
                        if c.get("type") in ["input_text", "text", "output_text"]:
                            if c.get("type") == "output_text" or role == "assistant":
                                blocks.append({"type": "output_text", "text": c.get("text")})
                            else:
                                blocks.append({"type": "input_text", "text": c.get("text")})
                        elif c.get("type") == "input_image":
                            blocks.append({"type": "input_image", "image_url": c.get("image_url")})
                        elif c.get("type") == "image_params":
                            for k, v in c.get("image_params", {}).items():
                                if k in self.image_output_params:
                                    image_generation_tool_params[k] = v
                        elif c.get("type") == "apply_patch_call":
                            # Skip apply_patch_call items - they've already been processed above
                            # The outputs are already in input_items, recipient doesn't need the raw call
                            continue
                        else:
                            raise ValueError(f"Invalid content type: {c.get('type')}")
                else:
                    blocks.append({"type": content_type, "text": content})

                # Only append if we have valid content blocks
                if blocks:
                    input_items.append({"role": role, "content": blocks})
            else:
                # Tool call response
                tool_call_id = m.get("tool_call_id", None)
                content = content_str(m.get("content"))

                # Skip if this corresponds to an already-processed apply_patch_call
                if tool_call_id and tool_call_id in processed_apply_patch_call_ids:
                    continue
                else:
                    # Regular function call output
                    input_items.append({
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": content,
                    })

    def _parse_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Parse and transform special parameters for Responses API.

        Converts verbosity and reasoning_effort parameters to the format expected
        by the Responses API.

        Args:
            params: Request parameters dict (modified in place)

        Returns:
            Modified params dict
        """
        if "verbosity" in params:
            verbosity = params.pop("verbosity")
            params["text"] = {"verbosity": verbosity}
        if "reasoning_effort" in params:
            reasoning_effort = params.pop("reasoning_effort")
            params["reasoning"] = {"effort": reasoning_effort}
        return params

    def _normalize_messages_for_responses_api(
        self,
        messages: list[dict[str, Any]],
        built_in_tools: list[str],
        workspace_dir: str,
        allowed_paths: list[str],
        previous_apply_patch_calls: dict[str, Any],
        image_generation_tool_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Normalize messages for Responses API format.

        This method processes apply_patch_call items from messages before sending to recipient,
        implementing the diff generation (sender) -> executor (recipient) pattern.

        Args:
            messages: List of messages to normalize
            built_in_tools: List of built-in tools enabled
            workspace_dir: Workspace directory for apply_patch operations
            allowed_paths: Allowed paths for apply_patch operations
            previous_apply_patch_calls: Apply patch calls from previous response
            image_generation_tool_params: Image generation tool parameters dict (modified in place)

        Returns:
            List of input items in Responses API format

        Note:
            This is a simplified version that doesn't execute apply_patch operations yet.
            Full apply_patch support will be added in later commits.
        """
        input_items = []

        # For now, skip apply_patch processing (will be implemented in later commits)
        # This allows us to get the basic structure working first

        # Convert messages to Responses API format
        processed_apply_patch_call_ids: set[str] = set()

        self._convert_messages_to_input(
            messages,
            processed_apply_patch_call_ids,
            image_generation_tool_params,
            input_items,
        )

        # Reverse input_items so that messages come first (in chronological order)
        return input_items[::-1]

    # --------------------------------------------------------------------------
    # Basic create() method
    # --------------------------------------------------------------------------

    def create(self, params: dict[str, Any]) -> UnifiedResponse:  # type: ignore[override]
        """Create a completion with stateful conversation support.

        This method implements ModelClient.create() but returns UnifiedResponse instead
        of ModelClientResponseProtocol. The rich UnifiedResponse structure is compatible
        via duck typing - it has .model attribute and works with message_retrieval().

        Args:
            params: Request parameters including:
                - model: Model name (e.g., "gpt-5", "gpt-4.1")
                - messages: List of message dicts (will be converted to input format)
                - input: List of input items in Responses API format (alternative to messages)
                - previous_response_id: Optional response ID to continue conversation
                - built_in_tools: Optional list of built-in tools to enable
                - workspace_dir: Optional workspace directory for apply_patch
                - allowed_paths: Optional allowed paths for apply_patch
                - **other Responses API parameters

        Returns:
            UnifiedResponse with rich content blocks
        """
        # Make a copy of params to avoid mutating the original
        params = params.copy()

        # Extract workspace and tool configuration
        workspace_dir = params.pop("workspace_dir", self._workspace_dir)
        allowed_paths = params.pop("allowed_paths", self._allowed_paths)
        built_in_tools = params.pop("built_in_tools", [])

        # Handle stateful conversation
        # If previous_response_id is not explicitly provided, use instance state
        if "previous_response_id" not in params and self._previous_response_id is not None:
            params["previous_response_id"] = self._previous_response_id

        # Convert messages to input format if needed
        if "messages" in params and "input" not in params:
            msgs = self._get_delta_messages(params.pop("messages"))
            image_generation_tool_params = {"type": "image_generation"}
            params["input"] = self._normalize_messages_for_responses_api(
                messages=msgs,
                built_in_tools=built_in_tools,
                workspace_dir=workspace_dir,
                allowed_paths=allowed_paths,
                previous_apply_patch_calls={},  # Will be populated in later commits
                image_generation_tool_params=image_generation_tool_params,
            )

        # Initialize tools list
        tools_list = []
        # Add built-in tools if specified
        if built_in_tools:
            if "image_generation" in built_in_tools:
                tools_list.append({"type": "image_generation"})
            if "web_search" in built_in_tools:
                tools_list.append({"type": "web_search_preview"})
            if "apply_patch" in built_in_tools or "apply_patch_async" in built_in_tools:
                tools_list.append({"type": "apply_patch"})

        # Add custom function tools if provided
        if "tools" in params:
            for tool in params["tools"]:
                tool_item = {"type": "function"}
                if "function" in tool:
                    tool_item |= tool["function"]
                tools_list.append(tool_item)

        if tools_list:
            params["tools"] = tools_list
            params["tool_choice"] = params.get("tool_choice", "auto")

        # Validate that we have at least one of the required parameters
        if not any(key in params for key in ["input", "previous_response_id", "prompt"]):
            # If we still don't have any required parameters, create a minimal input
            params["input"] = [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]

        # Parse special parameters (verbosity, reasoning_effort)
        params = self._parse_params(params)

        # Call Responses API
        try:
            response = self.client.responses.create(**params)  # type: ignore[misc]
            # Update state with new response ID
            self._set_previous_response_id(response.id)

            # Transform response to UnifiedResponse
            return self._transform_response(response, params.get("model", "unknown"))
        except Exception as e:
            logger.error(f"Error calling Responses API: {e}")
            raise

    def _transform_response(
        self,
        response: "Response",
        model: str,
    ) -> UnifiedResponse:
        """Transform Responses API Response to UnifiedResponse.

        This method converts the Responses API response format to UnifiedResponse,
        extracting all content blocks from response.output items.

        Args:
            response: Raw Responses API Response object
            model: Model name used for the request

        Returns:
            UnifiedResponse with all content blocks properly typed
        """
        content_blocks = []
        messages = []

        # Get output items from response
        output_items = getattr(response, "output", [])

        # Process each output item
        for item in output_items:
            # Convert Pydantic objects to dicts for uniform handling
            if hasattr(item, "model_dump"):
                item_dict = item.model_dump()
            elif hasattr(item, "dict"):
                item_dict = item.dict()
            elif isinstance(item, dict):
                item_dict = item
            else:
                logger.debug(f"Unknown output item type: {type(item)}")
                continue

            item_type = item_dict.get("type")

            # Handle message type items (output_text blocks)
            if item_type == "message":
                # Extract content blocks from message
                message_content = item_dict.get("content", [])

                # Process each content block in the message
                for block in message_content:
                    block_type = block.get("type")

                    # Handle output_text blocks (main text content)
                    if block_type == "output_text":
                        text = block.get("text", "")
                        if text:
                            content_blocks.append(TextContent(text=text))

            # Handle reasoning type items (o3 models)
            elif item_type == "reasoning":
                # Extract reasoning text from the item
                reasoning_text = item_dict.get("reasoning", "")
                if reasoning_text:
                    # Extract summary if available
                    summary = item_dict.get("summary", None)
                    content_blocks.append(
                        ReasoningContent(
                            reasoning=reasoning_text,
                            summary=summary,
                        )
                    )

            # Handle function_call type items (custom function tools)
            elif item_type == "function_call":
                # Extract function call details
                call_id = item_dict.get("call_id", "")
                function_name = item_dict.get("name", "")
                function_arguments = item_dict.get("arguments", "")

                if call_id and function_name:
                    # Convert arguments to string if it's not already
                    if not isinstance(function_arguments, str):
                        import json

                        function_arguments = json.dumps(function_arguments)

                    content_blocks.append(
                        ToolCallContent(
                            id=call_id,
                            name=function_name,
                            arguments=function_arguments,
                        )
                    )

            # Handle apply_patch_call type items (built-in tool)
            elif item_type == "apply_patch_call":
                # Use GenericContent to preserve all fields (call_id, status, operation, etc.)
                # This preserves the full structure for apply_patch operations
                content_blocks.append(
                    GenericContent(
                        type="apply_patch_call",
                        call_id=item_dict.get("call_id"),
                        status=item_dict.get("status"),
                        operation=item_dict.get("operation", {}),
                        **{
                            k: v
                            for k, v in item_dict.items()
                            if k not in ["type", "call_id", "status", "operation"]
                        },
                    )
                )

        # Create UnifiedMessage with all content blocks
        # Responses API typically returns assistant messages
        if content_blocks:
            messages.append(
                UnifiedMessage(
                    role="assistant",
                    content=content_blocks,
                )
            )
        else:
            # If no content blocks, create empty message (shouldn't happen in practice)
            messages.append(
                UnifiedMessage(
                    role="assistant",
                    content=[TextContent(text="")],
                )
            )

        # Extract usage information
        usage = {}
        usage_obj = getattr(response, "usage", None)

        if usage_obj:
            # Convert Pydantic usage object to dict for uniform access
            if hasattr(usage_obj, "model_dump"):
                usage_dict = usage_obj.model_dump()
            elif hasattr(usage_obj, "dict"):
                usage_dict = usage_obj.dict()
            elif isinstance(usage_obj, dict):
                usage_dict = usage_obj
            else:
                usage_dict = {}

            # Map Responses API usage fields to UnifiedResponse format
            usage = {
                "prompt_tokens": usage_dict.get("input_tokens", 0),
                "completion_tokens": usage_dict.get("output_tokens", 0),
                "total_tokens": usage_dict.get("total_tokens", 0),
            }

            # Extract reasoning tokens if available (for o3 models)
            output_tokens_details = usage_dict.get("output_tokens_details", {})
            if output_tokens_details:
                reasoning_tokens = output_tokens_details.get("reasoning_tokens", 0)
                if reasoning_tokens > 0:
                    usage["reasoning_tokens"] = reasoning_tokens

        # Get model name from response or use provided model
        response_model = getattr(response, "model", None) or model

        # Build UnifiedResponse
        unified_response = UnifiedResponse(
            id=getattr(response, "id", ""),
            model=response_model,
            provider="openai_responses",
            messages=messages,
            usage=usage,
            status="completed",
            provider_metadata={
                "created": getattr(response, "created", None),
            },
        )

        # Calculate cost - pending
        # For now, set to 0
        unified_response.cost = 0.0

        return unified_response
