# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import os
import warnings
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from autogen.code_utils import content_str
from autogen.import_utils import optional_import_block, require_optional_import

if TYPE_CHECKING:
    from autogen.oai.client import ModelClient, OpenAI, OpenAILLMConfigEntry
else:
    # Import at runtime to avoid circular import
    OpenAILLMConfigEntry = None
    ModelClient = None
    OpenAI = None

with optional_import_block() as openai_result:
    from openai.types.responses.response import Response
    from openai.types.responses.response_output_item import ImageGenerationCall

# Image Costs
# Pricing per image (in USD)
PRICING = {
    "gpt-image-1": {
        "low": {"1024x1024": 0.011, "1024x1536": 0.016, "1536x1024": 0.016},
        "medium": {"1024x1024": 0.042, "1024x1536": 0.063, "1536x1024": 0.063},
        "high": {"1024x1024": 0.167, "1024x1536": 0.25, "1536x1024": 0.25},
    },
    "dall-e-3": {
        "standard": {"1024x1024": 0.040, "1024x1792": 0.080, "1792x1024": 0.080},
        "hd": {"1024x1024": 0.080, "1024x1792": 0.120, "1792x1024": 0.120},
    },
    "dall-e-2": {"standard": {"1024x1024": 0.020, "512x512": 0.018, "256x256": 0.016}},
}

# Valid sizes for each model
VALID_SIZES = {
    "gpt-image-1": ["1024x1024", "1024x1536", "1536x1024"],
    "dall-e-3": ["1024x1024", "1024x1792", "1792x1024"],
    "dall-e-2": ["1024x1024", "512x512", "256x256"],
}

logger = logging.getLogger(__name__)


@dataclass
class ApplyPatchCallOutput:
    call_id: str
    status: str
    output: str
    type: str = "apply_patch_call_output"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def calculate_openai_image_cost(
    model: str = "gpt-image-1", size: str = "1024x1024", quality: str = "high"
) -> tuple[float, str]:
    """Calculate the cost for a single image generation.

    Args:
        model: Model name ("gpt-image-1", "dall-e-3" or "dall-e-2")
        size: Image size (e.g., "1024x1024", "1024x1536")
        quality: Quality setting:
                - For gpt-image-1: "low", "medium", or "high"
                - For dall-e-3: "standard" or "hd"
                - For dall-e-2: "standard" only

    Returns:
        Tuple of (cost, error_message)
    """
    # Normalize inputs
    model = model.lower()
    quality = quality.lower()

    # Validate model
    if model not in PRICING:
        return 0.0, f"Invalid model: {model}. Valid models: {list(PRICING.keys())}"

    # Validate size
    if size not in VALID_SIZES[model]:
        return 0.0, f"Invalid size {size} for {model}. Valid sizes: {VALID_SIZES[model]}"

    # Get the cost based on model type
    try:
        if model == "gpt-image-1" or model == "dall-e-3":
            cost = PRICING[model][quality][size]
        elif model == "dall-e-2":
            cost = PRICING[model]["standard"][size]
        else:
            return 0.0, f"Model {model} not properly configured"

        return cost, None

    except KeyError:
        return 0.0, f"Invalid quality '{quality}' for {model}"


def _get_base_class():
    """Lazy import OpenAILLMConfigEntry to avoid circular imports."""
    from autogen.oai.client import OpenAILLMConfigEntry

    return OpenAILLMConfigEntry


# -----------------------------------------------------------------------------
# OpenAI Client that calls the /responses endpoint
# -----------------------------------------------------------------------------
@require_optional_import("openai", "openai")
class OpenAIResponsesClient:
    """Minimal implementation targeting the experimental /responses endpoint.

    We purposefully keep the surface small - *create*, *message_retrieval*,
    *cost* and *get_usage* - enough for ConversableAgent to operate.  Anything
    that the new endpoint does natively (web_search, file_search, image
    generation, function calling, etc.) is transparently passed through by the
    OpenAI SDK so we don't replicate logic here.
    """

    RESPONSE_USAGE_KEYS: list[str] = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    def __init__(
        self,
        client: "OpenAI",
        response_format: BaseModel | dict[str, Any] | None = None,
    ):
        self._oai_client = client  # plain openai.OpenAI instance
        self.response_format = response_format  # kept for parity but unused for now

        # Initialize the image generation parameters
        self.image_output_params = {
            "quality": None,  # "high" or "low"
            "background": None,  # "white" or "black" or "transparent"
            "size": None,  # "1024x1024" or "1024x1792" or "1792x1024"
            "output_format": "png",  # "png", "jpg" or "jpeg" or "webp"
            "output_compression": None,  # 0-100 if output_format is "jpg" or "jpeg" or "webp"
        }
        self.previous_response_id = None

        # Image costs are calculated manually (rather than off returned information)
        self.image_costs = 0

    # ------------------------------------------------------------------ helpers
    # responses objects embed usage similarly to chat completions
    @staticmethod
    def _usage_dict(resp) -> dict:
        usage_obj = getattr(resp, "usage", None) or {}

        # Convert pydantic/BaseModel usage objects to dict for uniform access
        if hasattr(usage_obj, "model_dump"):
            usage = usage_obj.model_dump()
        elif isinstance(usage_obj, dict):
            usage = usage_obj
        else:  # fallback - unknown structure
            usage = {}

        output_tokens_details = usage.get("output_tokens_details", {})

        return {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "cost": getattr(resp, "cost", 0),
            "model": getattr(resp, "model", ""),
            "reasoning_tokens": output_tokens_details.get("reasoning_tokens", 0),
        }

    def _add_image_cost(self, response: "Response") -> None:
        """Add image cost to self._image_costs when an image is generated"""
        for output in response.output:
            if (
                isinstance(output, ImageGenerationCall)
                and hasattr(response.output[0], "model_extra")
                and response.output[0].model_extra
            ):
                extra_fields = output.model_extra

                image_cost, image_error = calculate_openai_image_cost(
                    model="gpt-image-1",
                    size=extra_fields.get("size", "1024x1536"),
                    quality=extra_fields.get("quality", "high"),
                )

                if not image_error and image_cost:
                    self.image_costs += image_cost

    def _get_delta_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Get the delta messages from the messages."""
        delta_messages = []
        for m in messages[::-1]:
            contents = m.get("content")
            is_last_completed_response = False
            if isinstance(contents, list):
                for c in contents:
                    if "status" in c and c.get("status") == "completed":
                        is_last_completed_response = True
                        break
            elif isinstance(contents, str):
                is_last_completed_response = "status" in m and m.get("status") == "completed"

            if is_last_completed_response:
                break
            delta_messages.append(m)
        return delta_messages[::-1]

    def _parse_params(self, params: dict[str, Any]) -> None:
        if "verbosity" in params:
            verbosity = params.pop("verbosity")
            params["text"] = {"verbosity": verbosity}
        return params

    def _apply_patch_operation(
        self,
        operation: dict[str, Any],
        call_id: str,
        workspace_dir: str = os.getcwd(),
        allowed_paths: list[str] = ["**"],
    ) -> ApplyPatchCallOutput:
        """Apply a patch operation and return apply_patch_call_output dict.

        Args:
            operation: Dictionary containing the patch operation with keys:
                - type: "create_file", "update_file", or "delete_file"
                - path: File path
                - diff: Diff string (for create_file and update_file)
            call_id: The call_id for this patch operation
            workspace_dir: a dedicated path for workspace directory
            allowed_paths: list of allowed paths for the workspace directory

        Returns:
            Dict with type, call_id, status, and output keys matching apply_patch_call_output format
        """
        from autogen.tools.apply_patch_tool import WorkspaceEditor

        editor = WorkspaceEditor(workspace_dir=workspace_dir, allowed_paths=allowed_paths)

        op_type = operation.get("type")
        import asyncio

        try:
            # Execute the patch operation
            if op_type == "create_file":
                result = asyncio.run(editor.create_file(operation))
            elif op_type == "update_file":
                result = asyncio.run(editor.update_file(operation))
            elif op_type == "delete_file":
                result = asyncio.run(editor.delete_file(operation))
            else:
                return ApplyPatchCallOutput(
                    call_id=call_id,
                    status="failed",
                    output=f"Unknown operation type: {op_type}",
                )
            # Return in the correct format
            return ApplyPatchCallOutput(
                call_id=call_id,
                status=result.get("status", "failed"),
                output=result.get("output", ""),
            )
        except Exception as e:
            return ApplyPatchCallOutput(
                call_id=call_id,
                status="failed",
                output=f"Error applying patch: {str(e)}",
            )

    def create(self, params: dict[str, Any]) -> "Response":
        """Invoke `client.responses.create() or .parse()`.

        If the caller provided a classic *messages* array we convert it to the
        *input* format expected by the Responses API.
        """
        params = params.copy()

        image_generation_tool_params = {"type": "image_generation"}
        web_search_tool_params = {"type": "web_search_preview"}
        apply_patch_tool_params = {"type": "apply_patch"}
        workspace_dir = params.pop("workspace_dir", None)
        allowed_paths = params.pop("allowed_paths", ["**"])
        built_in_tools = params.pop("built_in_tools", [])

        if self.previous_response_id is not None and "previous_response_id" not in params:
            params["previous_response_id"] = self.previous_response_id

        # Check previous response for apply_patch_call items that need outputs
        # This handles the case where the previous response contained apply_patch_call items
        previous_apply_patch_calls = {}
        if self.previous_response_id is not None:
            try:
                # Retrieve the previous response to check for apply_patch_call items
                previous_response = self._oai_client.responses.retrieve(self.previous_response_id)
                previous_output = getattr(previous_response, "output", [])

                # Extract apply_patch_call items from previous response
                for item in previous_output:
                    if hasattr(item, "model_dump"):
                        item = item.model_dump()
                    if item.get("type") == "apply_patch_call":
                        call_id = item.get("call_id")
                        if call_id:
                            previous_apply_patch_calls[call_id] = item
            except Exception as e:
                logger.debug(f"[apply_patch] Could not retrieve previous response: {e}")

        if "messages" in params and "input" not in params:
            msgs = self._get_delta_messages(params.pop("messages"))
            input_items = []
            apply_patch_call_ids = {}
            # First, add any apply_patch_call items from previous response
            if self.previous_response_id is not None and previous_apply_patch_calls:
                apply_patch_call_ids.update(previous_apply_patch_calls)

            # Then check messages for apply_patch_call items
            for msg_idx, msg in enumerate(msgs):
                role = msg.get("role")
                logger.debug(f"[apply_patch] Message {msg_idx}: role={role}")

                if role == "assistant":
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for item_idx, item in enumerate(content):
                            # Check for apply_patch_call in content items
                            if isinstance(item, dict):
                                item_type = item.get("type")
                                if item_type == "apply_patch_call":
                                    call_id = item.get("call_id")
                                    if call_id:
                                        apply_patch_call_ids[call_id] = item
                    tool_calls = msg.get("tool_calls", [])

                    if isinstance(tool_calls, list):
                        for tool_call_idx, tool_call in enumerate(tool_calls):
                            if isinstance(tool_call, dict):
                                tool_call_type = tool_call.get("type")
                                if tool_call_type == "apply_patch_call":
                                    call_id = tool_call.get("call_id")
                                    if call_id:
                                        apply_patch_call_ids[call_id] = tool_call

            # FIRST: Generate outputs for ALL apply_patch_call items and add them to the beginning
            # This ensures they're available before any user/assistant messages that might reference them
            apply_patch_outputs = []
            if apply_patch_call_ids:
                for call_id, apply_patch_call in apply_patch_call_ids.items():
                    operation = apply_patch_call.get("operation", {})
                    if operation and "apply_patch_async" in built_in_tools:  # Only process if we have an operation
                        # Apply the patch operation and get the full output dict asynchronously
                        output = self._apply_patch_operation(
                            operation,
                            call_id=call_id,
                            workspace_dir=workspace_dir,
                            allowed_paths=allowed_paths,
                            async_patches=True,
                        )
                        apply_patch_outputs.append(output.to_dict())
                    elif operation and "apply_patch" not in built_in_tools:
                        # Apply the patch operation and get the full output dict synchronously
                        apply_patch_outputs.append(
                            self._apply_patch_operation(
                                operation,
                                call_id=call_id,
                                workspace_dir=workspace_dir,
                                allowed_paths=allowed_paths,
                                async_patches=False,
                            ).to_dict()
                        )

            # Add all apply_patch_call_outputs at the beginning of input_items
            input_items.extend(apply_patch_outputs)

            for m in msgs[::-1]:  # reverse the list to get the last item first
                role = m.get("role", "user")
                # First, we need to convert the content to the Responses API format
                content = m.get("content")
                blocks = []
                if role != "tool":
                    # Determine content type based on role: assistant uses output_text, user uses input_text
                    content_type = "output_text" if role == "assistant" else "input_text"
                    if isinstance(content, list):
                        for c in content:
                            if c.get("type") in ["input_text", "text", "output_text"]:
                                # Use the appropriate type based on role, or preserve output_text if already set
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
                                # Skip apply_patch_call items in assistant messages - we've already processed them above
                                # Don't include them in the content blocks
                                continue
                            else:
                                raise ValueError(f"Invalid content type: {c.get('type')}")
                    else:
                        blocks.append({"type": content_type, "text": content})
                    input_items.append({"role": role, "content": blocks})

                else:
                    # tool call response - but apply_patch_call items are NOT tool messages
                    # They're content items, so we skip them here
                    tool_call_id = m.get("tool_call_id", None)
                    content = content_str(m.get("content"))

                    # Check if this tool output corresponds to an apply_patch_call
                    # (This shouldn't happen for apply_patch, but keep for other tool types)
                    if tool_call_id and tool_call_id in apply_patch_call_ids:
                        # This case should already be handled above, but if we get here, skip it
                        continue
                    else:
                        # Regular function call output
                        input_items.append({
                            "type": "function_call_output",
                            "call_id": tool_call_id,
                            "output": content,
                        })

            # Reverse input_items so that apply_patch_call_outputs come first, then messages in chronological order
            # (We added outputs first, then messages in reverse, so reversing gives us: outputs, then messages in order)
            params["input"] = input_items[::-1]

        # Initialize tools list
        tools_list = []
        # Back-compat: add default tools
        if built_in_tools:
            if "image_generation" in built_in_tools:
                tools_list.append(image_generation_tool_params)
            if "web_search" in built_in_tools:
                tools_list.append(web_search_tool_params)
            if "apply_patch" in built_in_tools:
                tools_list.append(apply_patch_tool_params)

        if "tools" in params:
            for tool in params["tools"]:
                tool_item = {"type": "function"}
                if "function" in tool:
                    tool_item |= tool["function"]
                    tools_list.append(tool_item)
        params["tools"] = tools_list
        params["tool_choice"] = "auto"

        # Ensure we don't mix legacy params that Responses doesn't accept
        if params.get("stream") and params.get("background"):
            warnings.warn(
                "Streaming a background response may introduce latency.",
                UserWarning,
            )

        # Validate that we have at least one of the required parameters
        if not any(key in params for key in ["input", "previous_response_id", "prompt"]):
            # If we still don't have any required parameters, create a minimal input
            params["input"] = [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]

        # ------------------------------------------------------------------
        # Structured output handling - mimic OpenAIClient behaviour
        # ------------------------------------------------------------------

        if self.response_format is not None or "response_format" in params:

            def _create_or_parse(**kwargs):
                # For structured output we must convert dict / pydantic model
                # into the JSON-schema body expected by the API.
                if "stream" in kwargs:
                    kwargs.pop("stream")  # Responses API rejects stream with RF for now

                rf = kwargs.get("response_format", self.response_format)

                if isinstance(rf, dict):
                    from autogen.oai.client import _ensure_strict_json_schema

                    kwargs["text_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "schema": _ensure_strict_json_schema(rf, path=(), root=rf),
                            "name": "response_format",
                            "strict": True,
                        },
                    }
                else:
                    # pydantic.BaseModel subclass
                    from autogen.oai.client import type_to_response_format_param

                    kwargs["text_format"] = type_to_response_format_param(rf)
                if "response_format" in kwargs:
                    kwargs["text_format"] = kwargs.pop("response_format")
                try:
                    return self._oai_client.responses.parse(**kwargs)
                except TypeError as e:
                    # Older openai-python versions may not yet expose the
                    # text_format parameter on the Responses endpoint.
                    if "text_format" in str(e) and "unexpected" in str(e):
                        warnings.warn(
                            "Installed openai-python version doesn't support "
                            "`response_format` for the Responses API. "
                            "Falling back to raw text output.",
                            UserWarning,
                        )
                        kwargs.pop("text_format", None)
                        return self._oai_client.responses.create(**kwargs)

            response = _create_or_parse(**params)
            self.previous_response_id = response.id
            return response
        # No structured output
        params = self._parse_params(params)
        response = self._oai_client.responses.create(**params)
        self.previous_response_id = response.id
        # Accumulate image costs
        self._add_image_cost(response)
        return response

    def message_retrieval(self, response) -> list[str] | list["ModelClient.ModelClientResponseProtocol.Choice.Message"]:
        output = getattr(response, "output", [])
        content = []
        tool_calls = []

        for item in output:
            # Convert pydantic objects to plain dicts for uniform handling
            if hasattr(item, "model_dump"):
                item = item.model_dump()

            item_type = item.get("type")

            # Skip reasoning items - they're not messages
            if item_type == "reasoning":
                continue

            if item_type == "message":
                new_item = copy.deepcopy(item)
                new_item["type"] = "text"
                new_item["role"] = "assistant"

                blocks = item.get("content", [])
                if len(blocks) == 1 and blocks[0].get("type") == "output_text":
                    new_item["text"] = blocks[0]["text"]
                elif len(blocks) > 0:
                    # Handle multiple content blocks
                    text_parts = []
                    for block in blocks:
                        if block.get("type") == "output_text":
                            text_parts.append(block.get("text", ""))
                    new_item["text"] = " ".join(text_parts)

                if "content" in new_item:
                    del new_item["content"]
                content.append(new_item)
                continue

            # ------------------------------------------------------------------
            # 2) Custom function calls
            # ------------------------------------------------------------------
            if item_type == "function_call":
                tool_calls.append({
                    "id": item.get("call_id", None),
                    "function": {
                        "name": item.get("name", None),
                        "arguments": item.get("arguments"),
                    },
                    "type": "function_call",
                })
                continue

            if item_type == "apply_patch_call":
                tool_call_args = {
                    "id": item.get("id"),
                    "role": "tool_calls",
                    "type": "apply_patch_call",
                    "call_id": item.get("call_id"),
                    "status": item.get("status", "in_progress"),
                    "operation": item.get("operation", {}),
                }
                content.append(tool_call_args)
                continue

            # ------------------------------------------------------------------
            # 4) Built-in tool calls
            # ------------------------------------------------------------------
            if item_type and item_type.endswith("_call"):
                tool_name = item_type.replace("_call", "")
                tool_call_args = {
                    "id": item.get("id"),
                    "role": "tool_calls",
                    "type": "tool_call",  # Responses API currently routes via function-like tools
                    "name": tool_name,
                }
                if tool_name == "image_generation":
                    for k in self.image_output_params:
                        if k in item:
                            tool_call_args[k] = item[k]
                    encoded_base64_result = item.get("result", "")
                    tool_call_args["content"] = encoded_base64_result
                    # add image_url for image input back to oai response api.
                    output_format = self.image_output_params["output_format"]
                    tool_call_args["image_url"] = f"data:image/{output_format};base64,{encoded_base64_result}"
                elif tool_name == "web_search":
                    pass
                else:
                    raise ValueError(f"Invalid tool name: {tool_name}")
                content.append(tool_call_args)
                continue

            # ------------------------------------------------------------------
            # 5) Fallback - store raw dict so information isn't lost
            # ------------------------------------------------------------------
            content.append(item)

        return [
            {
                "role": "assistant",
                "id": response.id,
                "content": content if content else None,
                "tool_calls": tool_calls,
            }
        ]

    def cost(self, response):
        return self._usage_dict(response).get("cost", 0) + self.image_costs

    @staticmethod
    def get_usage(response):
        return OpenAIResponsesClient._usage_dict(response)
