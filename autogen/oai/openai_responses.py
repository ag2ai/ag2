# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import copy
import warnings
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
        if "reasoning_effort" in params:
            reasoning_effort = params.pop("reasoning_effort")
            params["reasoning"] = {"effort": reasoning_effort}
        return params

    def create(self, params: dict[str, Any]) -> "Response":
        """Invoke `client.responses.create() or .parse()`.

        If the caller provided a classic *messages* array we convert it to the
        *input* format expected by the Responses API.
        """
        params = params.copy()

        image_generation_tool_params = {"type": "image_generation"}
        web_search_tool_params = {"type": "web_search_preview"}
        apply_patch_tool_params = {"type": "apply_patch"}

        if self.previous_response_id is not None and "previous_response_id" not in params:
            params["previous_response_id"] = self.previous_response_id

        # Back-compat: transform messages → input if needed ------------------
        if "messages" in params and "input" not in params:
            msgs = self._get_delta_messages(params.pop("messages"))
            input_items = []
            for m in msgs[::-1]:  # reverse the list to get the last item first
                role = m.get("role", "user")
                # First, we need to convert the content to the Responses API format
                content = m.get("content")
                blocks = []
                if role != "tool":
                    if isinstance(content, list):
                        for c in content:
                            if c.get("type") in ["input_text", "text"]:
                                blocks.append({"type": "input_text", "text": c.get("text")})
                            elif c.get("type") == "input_image":
                                blocks.append({"type": "input_image", "image_url": c.get("image_url")})
                            elif c.get("type") == "image_params":
                                for k, v in c.get("image_params", {}).items():
                                    if k in self.image_output_params:
                                        image_generation_tool_params[k] = v
                            else:
                                raise ValueError(f"Invalid content type: {c.get('type')}")
                    else:
                        blocks.append({"type": "input_text", "text": content})
                    input_items.append({"role": role, "content": blocks})

                else:
                    if input_items:
                        break
                    # tool call response is the last item in the list
                    content = content_str(m.get("content"))
                    input_items.append({
                        "type": "function_call_output",
                        "call_id": m.get("tool_call_id", None),
                        "output": content,
                    })
                    break

            # Ensure we have at least one valid input item
            if input_items:
                params["input"] = input_items[::-1]
            else:
                # If no valid input items were created, create a default one
                # This prevents the API error about missing required parameters
                params["input"] = [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]

        # Initialize tools list
        tools_list = []
        # Back-compat: add default tools
        built_in_tools = params.pop("built_in_tools", [])
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

            # ------------------------------------------------------------------
            # 3) Built-in tool calls
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

            # handling apply_patch_call response from apply_patch tool call
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
            # 4) Fallback - store raw dict so information isn't lost
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
