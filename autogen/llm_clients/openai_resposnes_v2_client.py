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

import asyncio
import json
import logging
import os
import threading
import warnings
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from autogen.code_utils import content_str
from autogen.import_utils import optional_import_block
from autogen.oai.oai_models.chat_completion import ChatCompletionExtended, Choice
from autogen.oai.oai_models.chat_completion_message import ChatCompletionMessage
from autogen.oai.oai_models.chat_completion_message_tool_call import ChatCompletionMessageFunctionToolCall, Function
from autogen.oai.oai_models.completion_usage import CompletionUsage
from autogen.oai.openai_utils import OAI_PRICE1K

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
    CitationContent,
    GenericContent,
    ImageContent,
    ReasoningContent,
    TextContent,
    ToolCallContent,
    UnifiedMessage,
    UnifiedResponse,
)

logger = logging.getLogger(__name__)


@dataclass
class ApplyPatchCallOutput:
    """Output from an apply_patch_call operation.

    This dataclass represents the result of executing an apply_patch operation
    (create_file, update_file, or delete_file) on a file in the workspace.

    Attributes:
        call_id: Unique identifier for the patch call (from the API)
        status: Result status - "success" or "failed"
        output: Output message describing the result
        type: Always "apply_patch_call_output"
    """

    call_id: str
    status: str
    output: str
    type: str = "apply_patch_call_output"

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format for Responses API input."""
        return asdict(self)


# Pricing per image (in USD)
IMAGE_PRICING = {
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

# Valid sizes for each image model
IMAGE_VALID_SIZES = {
    "gpt-image-1": ["1024x1024", "1024x1536", "1536x1024"],
    "dall-e-3": ["1024x1024", "1024x1792", "1792x1024"],
    "dall-e-2": ["1024x1024", "512x512", "256x256"],
}

# Valid quality settings for each image model
IMAGE_VALID_QUALITIES = {
    "gpt-image-1": ["low", "medium", "high"],
    "dall-e-3": ["standard", "hd"],
    "dall-e-2": ["standard"],
}


def calculate_image_cost(
    model: str = "gpt-image-1",
    size: str = "1024x1024",
    quality: str = "high",
) -> tuple[float, str | None]:
    """Calculate the cost for a single image generation.

    Args:
        model: Model name ("gpt-image-1", "dall-e-3" or "dall-e-2")
        size: Image size (e.g., "1024x1024", "1024x1536")
        quality: Quality setting:
                - For gpt-image-1: "low", "medium", or "high"
                - For dall-e-3: "standard" or "hd"
                - For dall-e-2: "standard" only

    Returns:
        Tuple of (cost, error_message). error_message is None if successful.

    Example:
        cost, error = calculate_image_cost("gpt-image-1", "1024x1024", "high")
        if error:
            print(f"Error: {error}")
        else:
            print(f"Cost: ${cost:.3f}")
    """
    # Normalize inputs
    model = model.lower()
    quality = quality.lower()

    # Validate model
    if model not in IMAGE_PRICING:
        return 0.0, f"Invalid model: {model}. Valid models: {list(IMAGE_PRICING.keys())}"

    # Validate size
    if size not in IMAGE_VALID_SIZES[model]:
        return 0.0, f"Invalid size {size} for {model}. Valid sizes: {IMAGE_VALID_SIZES[model]}"

    # Get the cost based on model type
    try:
        if model == "gpt-image-1" or model == "dall-e-3":
            cost = IMAGE_PRICING[model][quality][size]
        elif model == "dall-e-2":
            cost = IMAGE_PRICING[model]["standard"][size]
        else:
            return 0.0, f"Model {model} not properly configured"

        return cost, None

    except KeyError:
        return 0.0, f"Invalid quality '{quality}' for {model}. Valid qualities: {IMAGE_VALID_QUALITIES[model]}"


def calculate_token_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    custom_price: tuple[float, float] | None = None,
) -> float:
    """Calculate the cost for token usage.

    Uses the OAI_PRICE1K pricing table or custom pricing if provided.

    Args:
        model: Model name (e.g., "gpt-5", "gpt-4o")
        prompt_tokens: Number of input/prompt tokens
        completion_tokens: Number of output/completion tokens
        custom_price: Optional tuple of (input_price_per_1k, output_price_per_1k)

    Returns:
        Cost in USD for the token usage

    Example:
        # Using standard pricing
        cost = calculate_token_cost("gpt-5", 1000, 500)

        # Using custom pricing
        cost = calculate_token_cost("custom-model", 1000, 500, custom_price=(0.001, 0.002))
    """
    # Use custom pricing if provided
    if custom_price is not None:
        input_price, output_price = custom_price
        return (prompt_tokens * input_price + completion_tokens * output_price) / 1000

    # Check if model is in the pricing table
    if model not in OAI_PRICE1K:
        # Model not found - return 0 (will be logged as warning)
        return 0.0

    price_1k = OAI_PRICE1K[model]

    # Handle tuple (input, output) or single value pricing
    if isinstance(price_1k, tuple):
        return (price_1k[0] * prompt_tokens + price_1k[1] * completion_tokens) / 1000
    else:
        # Single price for both input and output
        return price_1k * (prompt_tokens + completion_tokens) / 1000



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

    Example - Basic Usage:
        client = OpenAIResponsesV2Client(api_key="...")

        # Get rich response with stateful conversation
        response = client.create({
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "Hello"}],
        })

        # Access text response
        print(f"Answer: {response.text}")

        # Continue conversation (stateful)
        response2 = client.create({
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "Tell me more"}]
        })

    Example - Image Generation:
        # Configure default image output settings
        client.set_image_output_params(quality="high", size="1024x1024")

        # Generate an image
        response = client.create({
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "Generate a sunset image"}],
            "built_in_tools": ["image_generation"]
        })

        # Access generated images
        images = OpenAIResponsesV2Client.get_generated_images(response)
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

        # Image generation parameters
        self.image_output_params = {
            "quality": None,  # "high" or "low"
            "background": None,  # "white" or "black" or "transparent"
            "size": None,  # "1024x1024" or "1024x1792" or "1792x1024"
            "output_format": "png",  # "png", "jpg" or "jpeg" or "webp"
            "output_compression": None,  # 0-100 if output_format is "jpg" or "jpeg" or "webp"
        }

        # Web search configuration parameters
        self.web_search_params = {
            "user_location": None,  # Approximate user location for localized results
            "search_context_size": None,  # "low", "medium", or "high" - amount of context
        }

        # Cost tracking
        # Track cumulative image generation costs (calculated manually since API doesn't return it)
        self._image_costs: float = 0.0
        # Track cumulative token costs
        self._token_costs: float = 0.0
        # Track cumulative total tokens
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        # Custom pricing (input_price_per_1k, output_price_per_1k) - overrides OAI_PRICE1K
        self._custom_price: tuple[float, float] | None = None

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

    @staticmethod
    def get_citations(response: UnifiedResponse) -> list[CitationContent]:
        """Extract all citations from a UnifiedResponse.

        This is a convenience method to extract all CitationContent blocks from
        a response, which is particularly useful when using the web_search tool.

        Args:
            response: UnifiedResponse from create()

        Returns:
            List of CitationContent blocks from the response

        Example:
            response = client.create({
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "What are the latest AI trends?"}],
                "built_in_tools": ["web_search"]
            })

            citations = OpenAIResponsesV2Client.get_citations(response)
            for citation in citations:
                print(f"Source: {citation.title} - {citation.url}")
        """
        citations = []
        for msg in response.messages:
            for block in msg.content:
                if isinstance(block, CitationContent):
                    citations.append(block)
        return citations

    @staticmethod
    def get_web_search_calls(response: UnifiedResponse) -> list[GenericContent]:
        """Extract all web_search_call items from a UnifiedResponse.

        This method extracts the metadata about web searches that were performed,
        which can be useful for tracking and debugging.

        Args:
            response: UnifiedResponse from create()

        Returns:
            List of GenericContent blocks with type="web_search_call"

        Example:
            response = client.create({
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "Search for latest news"}],
                "built_in_tools": ["web_search"]
            })

            search_calls = OpenAIResponsesV2Client.get_web_search_calls(response)
            for call in search_calls:
                print(f"Search ID: {call.get('id')}, Status: {call.get('status')}")
        """
        search_calls = []
        for msg in response.messages:
            for block in msg.content:
                if isinstance(block, GenericContent) and block.type == "web_search_call":
                    search_calls.append(block)
        return search_calls

    @staticmethod
    def get_generated_images(response: UnifiedResponse) -> list[ImageContent]:
        """Extract all generated images from a UnifiedResponse.

        This is a convenience method to extract all ImageContent blocks from
        a response, which is particularly useful when using the image_generation tool.

        Args:
            response: UnifiedResponse from create()

        Returns:
            List of ImageContent blocks from the response

        Example:
            response = client.create({
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "Generate a sunset image"}],
                "built_in_tools": ["image_generation"]
            })

            images = OpenAIResponsesV2Client.get_generated_images(response)
            for image in images:
                if image.data_uri:
                    print(f"Generated image (data URI): {image.data_uri[:50]}...")
        """
        images = []
        for msg in response.messages:
            for block in msg.content:
                if isinstance(block, ImageContent):
                    images.append(block)
        return images

    @staticmethod
    def get_parsed_content(response: UnifiedResponse) -> GenericContent | None:
        """Extract parsed structured output from a UnifiedResponse.

        When using structured outputs (response_format parameter), the parsed
        result is stored as a GenericContent block with type="parsed".

        Args:
            response: UnifiedResponse from create()

        Returns:
            GenericContent block with type="parsed" containing the parsed data,
            or None if no parsed content is found

        Example:
            from pydantic import BaseModel

            class UserInfo(BaseModel):
                name: str
                age: int

            response = client.create({
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "Extract user info"}],
                "response_format": UserInfo
            })

            parsed = OpenAIResponsesV2Client.get_parsed_content(response)
            if parsed:
                # Access the parsed object (Pydantic model instance)
                user = parsed.parsed_object
                print(f"Name: {user.name}, Age: {user.age}")

                # Or access as dict
                data = parsed.parsed_dict
                print(f"Data: {data}")
        """
        for msg in response.messages:
            for block in msg.content:
                if isinstance(block, GenericContent) and block.type == "parsed":
                    return block
        return None

    @staticmethod
    def get_parsed_object(response: UnifiedResponse) -> Any:
        """Extract the parsed Pydantic object from a UnifiedResponse.

        Convenience method to directly get the parsed Pydantic model instance.

        Args:
            response: UnifiedResponse from create()

        Returns:
            The parsed Pydantic model instance, or None if not available

        Example:
            response = client.create({
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "Extract info"}],
                "response_format": MyModel
            })

            obj = OpenAIResponsesV2Client.get_parsed_object(response)
            if obj:
                print(f"Parsed: {obj}")
        """
        parsed = OpenAIResponsesV2Client.get_parsed_content(response)
        if parsed:
            return getattr(parsed, "parsed_object", None)
        return None

    @staticmethod
    def create_image_content(
        image_url: str,
        detail: str = "auto",
    ) -> dict[str, Any]:
        """Create an image content block for multimodal messages.

        Helper method to create properly formatted image content for use in messages.

        Args:
            image_url: URL of the image or base64 data URI
                - HTTP/HTTPS URL: "https://example.com/image.png"
                - Base64 data URI: "data:image/png;base64,..."
            detail: Image detail level - "auto", "low", or "high"

        Returns:
            Dict formatted for use in message content arrays

        Example:
            # Create a multimodal message with text and image
            response = client.create({
                "model": "gpt-5",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        OpenAIResponsesV2Client.create_image_content(
                            "https://example.com/photo.jpg",
                            detail="high"
                        )
                    ]
                }]
            })
        """
        return {
            "type": "input_image",
            "image_url": image_url,
            "detail": detail,
        }

    @staticmethod
    def create_multimodal_message(
        text: str,
        images: list[str] | None = None,
        role: str = "user",
    ) -> dict[str, Any]:
        """Create a multimodal message with text and optional images.

        Convenience method to create messages with mixed content types.

        Args:
            text: The text content of the message
            images: Optional list of image URLs or data URIs
            role: Message role - "user", "assistant", or "system"

        Returns:
            Dict formatted as a message for use in create()

        Example:
            # Message with text and multiple images
            message = OpenAIResponsesV2Client.create_multimodal_message(
                text="Compare these two images",
                images=[
                    "https://example.com/image1.jpg",
                    "https://example.com/image2.jpg"
                ]
            )

            response = client.create({
                "model": "gpt-5",
                "messages": [message]
            })
        """
        content: list[dict[str, Any]] = []

        # Add text content first
        if text:
            content.append({"type": "input_text", "text": text})

        # Add images
        if images:
            for image_url in images:
                content.append(OpenAIResponsesV2Client.create_image_content(image_url))

        return {
            "role": role,
            "content": content,
        }

    @staticmethod
    def get_all_images(response: UnifiedResponse) -> list[ImageContent]:
        """Extract all images from a UnifiedResponse (both generated and input).

        This is an alias for get_generated_images() for semantic clarity when
        working with multimodal responses.

        Args:
            response: UnifiedResponse from create()

        Returns:
            List of all ImageContent blocks from the response
        """
        return OpenAIResponsesV2Client.get_generated_images(response)

    def set_image_output_params(
        self,
        quality: str | None = None,
        size: str | None = None,
        background: str | None = None,
        output_format: str | None = None,
        output_compression: int | None = None,
    ) -> None:
        """Set default image generation parameters for all requests.

        These parameters will be applied to all requests that use the image_generation tool
        unless overridden in the request params.

        Args:
            quality: Image quality - "low", "medium", or "high" (for gpt-image-1)
                    or "standard", "hd" (for dall-e-3)
            size: Image size - "1024x1024", "1024x1536", "1536x1024" (gpt-image-1)
                  or "1024x1024", "1024x1792", "1792x1024" (dall-e-3)
            background: Background type - "white", "black", or "transparent"
            output_format: Output format - "png", "jpg", "jpeg", or "webp"
            output_compression: Compression level 0-100 (only for jpg/jpeg/webp)

        Example:
            client.set_image_output_params(
                quality="high",
                size="1024x1024",
                output_format="png"
            )
        """
        if quality is not None:
            valid_qualities = ["low", "medium", "high", "standard", "hd"]
            if quality not in valid_qualities:
                raise ValueError(f"quality must be one of {valid_qualities}, got: {quality}")
            self.image_output_params["quality"] = quality

        if size is not None:
            # Accept any size format - validation happens at API level
            self.image_output_params["size"] = size

        if background is not None:
            valid_backgrounds = ["white", "black", "transparent"]
            if background not in valid_backgrounds:
                raise ValueError(f"background must be one of {valid_backgrounds}, got: {background}")
            self.image_output_params["background"] = background

        if output_format is not None:
            valid_formats = ["png", "jpg", "jpeg", "webp"]
            if output_format not in valid_formats:
                raise ValueError(f"output_format must be one of {valid_formats}, got: {output_format}")
            self.image_output_params["output_format"] = output_format

        if output_compression is not None:
            if not (0 <= output_compression <= 100):
                raise ValueError(f"output_compression must be between 0 and 100, got: {output_compression}")
            self.image_output_params["output_compression"] = output_compression

    def _build_image_generation_tool_config(
        self,
        image_generation_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build image generation tool configuration.

        Creates the tool configuration dict for the image_generation built-in tool
        with optional parameters for customization.

        Args:
            image_generation_config: Optional configuration parameters:
                - quality: "low", "medium", "high" (gpt-image-1) or "standard", "hd" (dall-e-3)
                - size: Image dimensions (e.g., "1024x1024")
                - background: "white", "black", or "transparent"
                - output_format: "png", "jpg", "jpeg", or "webp"
                - output_compression: 0-100 (for jpg/jpeg/webp)

        Returns:
            Tool configuration dict for the Responses API

        Example:
            config = self._build_image_generation_tool_config({
                "quality": "high",
                "size": "1024x1536",
                "output_format": "png"
            })
        """
        tool_config: dict[str, Any] = {"type": "image_generation"}

        # Merge instance-level image params with request-level config
        effective_config = {**self.image_output_params}
        if image_generation_config:
            effective_config.update(image_generation_config)

        # Add parameters if they have values
        for param in ["quality", "size", "background", "output_format", "output_compression"]:
            value = effective_config.get(param)
            if value is not None:
                tool_config[param] = value

        return tool_config

    def get_image_costs(self) -> float:
        """Get the cumulative image generation costs.

        Returns the total cost of all images generated through this client instance.
        This is tracked separately because the API doesn't include image costs in
        the usage response.

        Returns:
            Total image generation cost in USD

        Example:
            # After generating several images
            total_cost = client.get_image_costs()
            print(f"Total image generation cost: ${total_cost:.2f}")
        """
        return self._image_costs

    def reset_image_costs(self) -> None:
        """Reset the cumulative image generation costs to zero.

        Useful when starting a new session or when you want to track costs
        for a specific set of operations.
        """
        self._image_costs = 0.0

    def get_token_costs(self) -> float:
        """Get the cumulative token costs.

        Returns the total cost of all token usage through this client instance.

        Returns:
            Total token cost in USD

        Example:
            total_token_cost = client.get_token_costs()
            print(f"Total token cost: ${total_token_cost:.4f}")
        """
        return self._token_costs

    def reset_token_costs(self) -> None:
        """Reset the cumulative token costs to zero."""
        self._token_costs = 0.0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    def get_total_costs(self) -> float:
        """Get the total cumulative costs (tokens + images).

        Returns:
            Total cost in USD (token costs + image costs)

        Example:
            total_cost = client.get_total_costs()
            print(f"Total cost: ${total_cost:.4f}")
        """
        return self._token_costs + self._image_costs

    def reset_all_costs(self) -> None:
        """Reset all cumulative cost tracking to zero.

        Resets both token costs and image costs.
        """
        self._token_costs = 0.0
        self._image_costs = 0.0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    def get_cumulative_usage(self) -> dict[str, Any]:
        """Get cumulative usage statistics across all requests.

        Returns:
            Dict with cumulative usage statistics

        Example:
            usage = client.get_cumulative_usage()
            print(f"Total tokens: {usage['total_tokens']}")
            print(f"Total cost: ${usage['total_cost']:.4f}")
        """
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
            "token_cost": self._token_costs,
            "image_cost": self._image_costs,
            "total_cost": self._token_costs + self._image_costs,
        }

    def set_custom_price(
        self,
        input_price_per_1k: float,
        output_price_per_1k: float,
    ) -> None:
        """Set custom pricing for token cost calculation.

        This overrides the default OAI_PRICE1K pricing table.

        Args:
            input_price_per_1k: Price per 1000 input tokens in USD
            output_price_per_1k: Price per 1000 output tokens in USD

        Example:
            # Set custom pricing for a fine-tuned model
            client.set_custom_price(
                input_price_per_1k=0.003,
                output_price_per_1k=0.006
            )
        """
        self._custom_price = (input_price_per_1k, output_price_per_1k)

    def clear_custom_price(self) -> None:
        """Clear custom pricing and use default OAI_PRICE1K table."""
        self._custom_price = None

    @staticmethod
    def get_apply_patch_calls(response: UnifiedResponse) -> list[GenericContent]:
        """Extract all apply_patch_call items from a UnifiedResponse.

        This method extracts the metadata about apply_patch operations that were
        performed, which can be useful for tracking file modifications.

        Args:
            response: UnifiedResponse from create()

        Returns:
            List of GenericContent blocks with type="apply_patch_call"

        Example:
            response = client.create({
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "Update the config file"}],
                "built_in_tools": ["apply_patch"]
            })

            patch_calls = OpenAIResponsesV2Client.get_apply_patch_calls(response)
            for call in patch_calls:
                print(f"Patch ID: {call.call_id}, Status: {call.status}")
        """
        patch_calls = []
        for msg in response.messages:
            for block in msg.content:
                if isinstance(block, GenericContent) and block.type == "apply_patch_call":
                    patch_calls.append(block)
        return patch_calls

    def _apply_patch_operation(
        self,
        operation: dict[str, Any],
        call_id: str,
        workspace_dir: str | None = None,
        allowed_paths: list[str] | None = None,
        async_patches: bool = False,
    ) -> ApplyPatchCallOutput:
        """Apply a patch operation and return apply_patch_call_output.

        This method executes a single file operation (create, update, or delete)
        using the WorkspaceEditor tool.

        Args:
            operation: Dictionary containing the patch operation with keys:
                - type: "create_file", "update_file", or "delete_file"
                - path: File path relative to workspace
                - diff: Diff string (for create_file and update_file)
            call_id: The call_id for this patch operation (from API response)
            workspace_dir: Workspace directory path (defaults to instance setting)
            allowed_paths: List of allowed path patterns (defaults to instance setting)
            async_patches: Whether to apply patches asynchronously

        Returns:
            ApplyPatchCallOutput with call_id, status, and output message

        Example:
            output = self._apply_patch_operation(
                operation={"type": "create_file", "path": "test.py", "diff": "..."},
                call_id="call_123",
                workspace_dir="/workspace"
            )
            print(f"Status: {output.status}, Output: {output.output}")
        """
        from autogen.tools.experimental.apply_patch.apply_patch_tool import WorkspaceEditor

        # Use instance defaults if not provided
        workspace_dir = workspace_dir or self._workspace_dir
        allowed_paths = allowed_paths or self._allowed_paths

        # Valid operation types for apply_patch tool
        valid_operations = {"create_file", "update_file", "delete_file"}

        editor = WorkspaceEditor(workspace_dir=workspace_dir, allowed_paths=allowed_paths)
        op_type = operation.get("type")

        # Validate operation type early
        if op_type not in valid_operations:
            return ApplyPatchCallOutput(
                call_id=call_id,
                status="failed",
                output=f"Invalid operation type: {op_type}. Must be one of {valid_operations}",
            )

        def _run_async_in_thread(coro: Any) -> Any:
            """Run an async coroutine in a separate thread with its own event loop."""
            result: dict[str, Any] = {}

            def runner() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result["value"] = loop.run_until_complete(coro)
                loop.close()

            t = threading.Thread(target=runner)
            t.start()
            t.join()
            return result["value"]

        try:
            # Execute the patch operation
            if async_patches:
                # Check if there's already a running event loop (e.g., in Jupyter)
                try:
                    asyncio.get_running_loop()
                    # If we get here, there's a running loop, so use thread-based approach
                    if op_type == "create_file":
                        result = _run_async_in_thread(editor.a_create_file(operation))
                    elif op_type == "update_file":
                        result = _run_async_in_thread(editor.a_update_file(operation))
                    elif op_type == "delete_file":
                        result = _run_async_in_thread(editor.a_delete_file(operation))
                    else:
                        return ApplyPatchCallOutput(
                            call_id=call_id,
                            status="failed",
                            output=f"Unknown operation type: {op_type}",
                        )
                except RuntimeError:
                    # No running loop, safe to use asyncio.run()
                    if op_type == "create_file":
                        result = asyncio.run(editor.a_create_file(operation))
                    elif op_type == "update_file":
                        result = asyncio.run(editor.a_update_file(operation))
                    elif op_type == "delete_file":
                        result = asyncio.run(editor.a_delete_file(operation))
                    else:
                        return ApplyPatchCallOutput(
                            call_id=call_id,
                            status="failed",
                            output=f"Unknown operation type: {op_type}",
                        )
            else:
                # Synchronous execution
                if op_type == "create_file":
                    result = editor.create_file(operation)
                elif op_type == "update_file":
                    result = editor.update_file(operation)
                elif op_type == "delete_file":
                    result = editor.delete_file(operation)
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
            logger.error(f"Error applying patch: {e}")
            return ApplyPatchCallOutput(
                call_id=call_id,
                status="failed",
                output=f"Error applying patch: {str(e)}",
            )

    def _extract_apply_patch_calls(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Extract apply_patch_call items from messages.

        Scans through messages to find all apply_patch_call items, which contain
        file operations to be executed.

        Args:
            messages: List of messages to scan

        Returns:
            Dictionary mapping call_id to apply_patch_call item dict

        Example:
            calls = self._extract_apply_patch_calls(messages)
            for call_id, call_data in calls.items():
                print(f"Call {call_id}: {call_data.get('operation', {}).get('type')}")
        """
        message_apply_patch_calls: dict[str, Any] = {}
        for msg in messages:
            role = msg.get("role")
            if role == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "apply_patch_call":
                            call_id = item.get("call_id")
                            if call_id:
                                message_apply_patch_calls[call_id] = item
                tool_calls = msg.get("tool_calls", [])
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict) and tool_call.get("type") == "apply_patch_call":
                            call_id = tool_call.get("call_id")
                            if call_id:
                                message_apply_patch_calls[call_id] = tool_call
        return message_apply_patch_calls

    def _execute_apply_patch_calls(
        self,
        calls_dict: dict[str, Any],
        built_in_tools: list[str],
        workspace_dir: str,
        allowed_paths: list[str],
    ) -> list[dict[str, Any]]:
        """Execute apply_patch_call items and return outputs.

        This method processes a collection of apply_patch_call items, executing
        each file operation and returning the results.

        Args:
            calls_dict: Dictionary mapping call_id to apply_patch_call item
            built_in_tools: List of built-in tools enabled (checks for apply_patch/apply_patch_async)
            workspace_dir: Workspace directory for file operations
            allowed_paths: Allowed paths for file operations

        Returns:
            List of apply_patch_call_output dictionaries ready for API input

        Example:
            outputs = self._execute_apply_patch_calls(
                calls_dict={"call_123": {...}},
                built_in_tools=["apply_patch"],
                workspace_dir="/workspace",
                allowed_paths=["**"]
            )
        """
        outputs: list[dict[str, Any]] = []
        if not calls_dict or not ("apply_patch" in built_in_tools or "apply_patch_async" in built_in_tools):
            return outputs

        async_mode = "apply_patch_async" in built_in_tools

        for call_id, apply_patch_call in calls_dict.items():
            operation = apply_patch_call.get("operation", {})
            if operation:
                output = self._apply_patch_operation(
                    operation,
                    call_id=call_id,
                    workspace_dir=workspace_dir,
                    allowed_paths=allowed_paths,
                    async_patches=async_mode,
                )
                outputs.append(output.to_dict())
        return outputs

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

    def _convert_image_to_input_block(self, image_content: dict[str, Any]) -> dict[str, Any] | None:
        """Convert various image content formats to Responses API input_image block.

        Handles multiple image input formats:
        - Responses API format: {"type": "input_image", "image_url": "..."}
        - Chat Completions format: {"type": "image_url", "image_url": {"url": "..."}}
        - Direct URL string in image_url field
        - Base64 data URI format

        Args:
            image_content: Dict containing image content in various formats

        Returns:
            Responses API input_image block dict, or None if conversion fails

        Example:
            # Chat Completions format
            block = self._convert_image_to_input_block({
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png", "detail": "high"}
            })
            # Returns: {"type": "input_image", "image_url": "https://example.com/image.png"}
        """
        content_type = image_content.get("type", "")

        # Already in Responses API format
        if content_type == "input_image":
            image_url = image_content.get("image_url")
            if image_url:
                return {"type": "input_image", "image_url": image_url}

        # Chat Completions API format: {"type": "image_url", "image_url": {"url": "...", "detail": "..."}}
        elif content_type == "image_url":
            image_url_obj = image_content.get("image_url", {})
            if isinstance(image_url_obj, dict):
                url = image_url_obj.get("url")
                if url:
                    return {"type": "input_image", "image_url": url}
            elif isinstance(image_url_obj, str):
                # Direct URL string
                return {"type": "input_image", "image_url": image_url_obj}

        # Handle direct image_url field (legacy format)
        elif "image_url" in image_content and content_type not in ["input_text", "text", "output_text"]:
            image_url = image_content.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                if url:
                    return {"type": "input_image", "image_url": url}
            elif isinstance(image_url, str):
                return {"type": "input_image", "image_url": image_url}

        return None

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

        Supports multimodal content including:
        - Text (input_text, output_text)
        - Images (input_image, image_url from Chat Completions API)
        - Image generation parameters (image_params)
        - Apply patch calls (skipped, processed separately)

        Args:
            messages: List of messages to convert
            processed_apply_patch_call_ids: Set of call_ids that have been processed
            image_generation_tool_params: Image generation tool parameters dict (modified in place)
            input_items: List to append converted message items to (modified in place)
        """
        for m in messages[::-1]:  # reverse the list to get the last item first
            role = m.get("role", "user")
            content = m.get("content")
            blocks: list[dict[str, Any]] = []

            if role != "tool":
                content_type = "output_text" if role == "assistant" else "input_text"
                if isinstance(content, list):
                    for c in content:
                        c_type = c.get("type", "")

                        # Handle text content
                        if c_type in ["input_text", "text", "output_text"]:
                            if c_type == "output_text" or role == "assistant":
                                blocks.append({"type": "output_text", "text": c.get("text", "")})
                            else:
                                blocks.append({"type": "input_text", "text": c.get("text", "")})

                        # Handle image content (multiple formats)
                        elif c_type in ["input_image", "image_url"]:
                            image_block = self._convert_image_to_input_block(c)
                            if image_block:
                                blocks.append(image_block)

                        # Handle image generation parameters
                        elif c_type == "image_params":
                            for k, v in c.get("image_params", {}).items():
                                if k in self.image_output_params:
                                    image_generation_tool_params[k] = v

                        # Skip apply_patch_call items - they've already been processed
                        elif c_type == "apply_patch_call":
                            continue

                        # Handle unknown types with image_url field (legacy/fallback)
                        elif "image_url" in c:
                            image_block = self._convert_image_to_input_block(c)
                            if image_block:
                                blocks.append(image_block)

                        # Skip empty or unrecognized content types with warning
                        else:
                            if c_type:
                                logger.debug(f"Skipping unrecognized content type: {c_type}")

                else:
                    # Simple string content
                    blocks.append({"type": content_type, "text": content or ""})

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

    def _build_web_search_tool_config(
        self,
        web_search_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build web search tool configuration.

        Creates the tool configuration dict for the web_search_preview built-in tool
        with optional parameters for customization.

        Args:
            web_search_config: Optional configuration parameters:
                - user_location: dict with "type": "approximate" and location info
                  for localized search results (e.g., {"type": "approximate", "city": "NYC"})
                - search_context_size: "low", "medium", or "high" - controls how much
                  context from web pages is provided to the model

        Returns:
            Tool configuration dict for the Responses API

        Example:
            config = self._build_web_search_tool_config({
                "user_location": {"type": "approximate", "city": "San Francisco"},
                "search_context_size": "medium"
            })
            # Returns: {"type": "web_search_preview", "user_location": {...}, "search_context_size": "medium"}
        """
        tool_config: dict[str, Any] = {"type": "web_search_preview"}

        # Merge instance-level web search params
        effective_config = {**self.web_search_params}
        if web_search_config:
            effective_config.update(web_search_config)

        # Add user_location if provided
        if effective_config.get("user_location"):
            tool_config["user_location"] = effective_config["user_location"]

        # Add search_context_size if provided (must be "low", "medium", or "high")
        search_context_size = effective_config.get("search_context_size")
        if search_context_size and search_context_size in ["low", "medium", "high"]:
            tool_config["search_context_size"] = search_context_size

        return tool_config

    def set_web_search_params(
        self,
        user_location: dict[str, Any] | None = None,
        search_context_size: str | None = None,
    ) -> None:
        """Set default web search parameters for all requests.

        These parameters will be applied to all requests that use the web_search tool
        unless overridden in the request params.

        Args:
            user_location: Approximate user location for localized search results.
                Should be a dict like {"type": "approximate", "city": "NYC", "country": "US"}
            search_context_size: Amount of context from web pages - "low", "medium", or "high"

        Example:
            client.set_web_search_params(
                user_location={"type": "approximate", "city": "London", "country": "UK"},
                search_context_size="medium"
            )
        """
        if user_location is not None:
            self.web_search_params["user_location"] = user_location
        if search_context_size is not None:
            if search_context_size not in ["low", "medium", "high"]:
                raise ValueError(f"search_context_size must be 'low', 'medium', or 'high', got: {search_context_size}")
            self.web_search_params["search_context_size"] = search_context_size

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

        The pattern is:
        1. Agent A (sender) generates apply_patch_call items in its response
        2. This method (on Agent B/recipient side) extracts and executes those patches
        3. The outputs (apply_patch_call_output) are included in the API input
        4. Agent B then sees the results of the file operations

        Args:
            messages: List of messages to normalize
            built_in_tools: List of built-in tools enabled (apply_patch or apply_patch_async)
            workspace_dir: Workspace directory for apply_patch operations
            allowed_paths: Allowed paths for apply_patch operations
            previous_apply_patch_calls: Apply patch calls from previous response (same agent continuation)
            image_generation_tool_params: Image generation tool parameters dict (modified in place)

        Returns:
            List of input items in Responses API format, with apply_patch outputs first
        """
        input_items: list[dict[str, Any]] = []

        # Extract apply_patch_call items from messages (from sender agent)
        message_apply_patch_calls = self._extract_apply_patch_calls(messages)

        # Execute apply_patch_call items from messages if built_in_tools includes apply_patch
        # This implements the diff generation (sender) -> executor (recipient) pattern
        message_apply_patch_outputs = self._execute_apply_patch_calls(
            message_apply_patch_calls,
            built_in_tools,
            workspace_dir,
            allowed_paths,
        )

        # Execute apply_patch_call items from previous_response_id (same agent continuation)
        previous_apply_patch_outputs = self._execute_apply_patch_calls(
            previous_apply_patch_calls,
            built_in_tools,
            workspace_dir,
            allowed_paths,
        )

        # Combine all outputs: message outputs first (from sender), then previous response outputs
        all_apply_patch_outputs = message_apply_patch_outputs + previous_apply_patch_outputs

        # Add all apply_patch outputs at the beginning
        if all_apply_patch_outputs:
            input_items.extend(all_apply_patch_outputs)

        # Track which apply_patch_call items have been processed
        # (their outputs are already in input_items, so we skip them during message conversion)
        processed_apply_patch_call_ids = set(message_apply_patch_calls.keys()) | set(previous_apply_patch_calls.keys())

        # Convert messages to Responses API format, filtering out processed apply_patch_call items
        self._convert_messages_to_input(
            messages,
            processed_apply_patch_call_ids,
            image_generation_tool_params,
            input_items,
        )

        # Reverse input_items so that messages come first (in chronological order), then outputs
        # We added outputs first, then messages in reverse, so reversing gives us: messages first, then outputs
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
                - response_format: Optional Pydantic model or JSON schema dict for structured output
                - built_in_tools: Optional list of built-in tools to enable:
                    - "web_search": Search the web for current information
                    - "image_generation": Generate images
                    - "apply_patch": File operations (create/update/delete files)
                    - "apply_patch_async": Async file operations
                - workspace_dir: Optional workspace directory for apply_patch
                - allowed_paths: Optional allowed paths for apply_patch
                - **other Responses API parameters

        Returns:
            UnifiedResponse with rich content blocks including:
                - TextContent: Main response text
                - CitationContent: Web search citations (when using web_search)
                - ImageContent: Generated images (when using image_generation)
                - ReasoningContent: Reasoning blocks (o3 models)
                - ToolCallContent: Custom function tool calls
                - GenericContent: Other content types (including type="parsed" for structured output)

        Example:
            # Basic usage
            response = client.create({
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "Hello"}]
            })

            # With structured output (Pydantic model)
            from pydantic import BaseModel

            class UserInfo(BaseModel):
                name: str
                age: int
                city: str

            response = client.create({
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "Extract: John is 30, lives in NYC"}],
                "response_format": UserInfo
            })

            # Access the parsed object
            user = OpenAIResponsesV2Client.get_parsed_object(response)
            print(f"Name: {user.name}, Age: {user.age}")

            # With built-in tools
            response = client.create({
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "Latest news"}],
                "built_in_tools": ["web_search"]
            })
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

        # Extract tool configurations if provided
        web_search_config = params.pop("web_search_config", None)
        image_generation_config = params.pop("image_generation_config", None)

        # Initialize tools list
        tools_list = []
        # Add built-in tools if specified
        if built_in_tools:
            if "image_generation" in built_in_tools:
                # Build image generation tool config with optional parameters
                tools_list.append(self._build_image_generation_tool_config(image_generation_config))
            if "web_search" in built_in_tools:
                # Build web search tool config with optional parameters
                tools_list.append(self._build_web_search_tool_config(web_search_config))
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

        # Check for response_format (Pydantic model or JSON schema dict)
        response_format = params.pop("response_format", None) or self._default_response_format

        if response_format is not None:
            # Convert response_format to text_format for Responses API
            try:
                # text_format = _convert_response_format_to_text_format(response_format)
                params["text_format"] = response_format

                # Remove stream if present (Responses API doesn't support streaming with text_format)
                if "stream" in params:
                    params.pop("stream")

                # Use responses.parse() for structured output
                try:
                    response = self.client.responses.parse(**params)  # type: ignore[misc]
                    self._set_previous_response_id(response.id)
                    return self._transform_response(
                        response,
                        params.get("model", "unknown"),
                        response_format=response_format,
                    )
                except TypeError as e:
                    # Older openai-python versions may not support text_format
                    if "text_format" in str(e) and "unexpected" in str(e):
                        warnings.warn(
                            "Installed openai-python version doesn't support "
                            "`response_format` for the Responses API. "
                            "Falling back to raw text output.",
                            UserWarning,
                        )
                        params.pop("text_format", None)
                        # Fall through to regular create below
                    else:
                        raise
            except ImportError as e:
                logger.warning(f"Could not convert response_format: {e}. Falling back to raw text.")
                # Fall through to regular create below

        # ------------------------------------------------------------------
        # Regular (non-structured or fallback) API call
        # ------------------------------------------------------------------
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
        response_format: type[BaseModel] | dict[str, Any] | None = None,
    ) -> UnifiedResponse:
        """Transform Responses API Response to UnifiedResponse.

        This method converts the Responses API response format to UnifiedResponse,
        extracting all content blocks from response.output items.

        Args:
            response: Raw Responses API Response object
            model: Model name used for the request
            response_format: Optional Pydantic model or JSON schema used for structured output

        Returns:
            UnifiedResponse with all content blocks properly typed
        """
        content_blocks: list[Any] = []
        messages = []

        # When using responses.parse(), the response has an `output_parsed` attribute
        # containing the parsed Pydantic model instance
        output_parsed = getattr(response, "output_parsed", None)
        if output_parsed is not None:
            # Store the parsed object as GenericContent with type="parsed"
            parsed_dict = None
            if hasattr(output_parsed, "model_dump"):
                parsed_dict = output_parsed.model_dump()
            elif hasattr(output_parsed, "dict"):
                parsed_dict = output_parsed.dict()
            elif isinstance(output_parsed, dict):
                parsed_dict = output_parsed

            content_blocks.append(
                GenericContent(
                    type="parsed",
                    parsed_object=output_parsed,
                    parsed_dict=parsed_dict,
                    response_format_type=type(output_parsed).__name__ if output_parsed else None,
                )
            )
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

                        # Extract annotations (citations from web search) if present
                        annotations = block.get("annotations", [])
                        for annotation in annotations:
                            if isinstance(annotation, dict):
                                ann_type = annotation.get("type", "")

                                # Handle URL citations from web search
                                if ann_type == "url_citation":
                                    url = annotation.get("url", "")
                                    title = annotation.get("title", "")
                                    # The text range in the output that this citation refers to
                                    start_index = annotation.get("start_index", 0)
                                    end_index = annotation.get("end_index", 0)

                                    if url:
                                        content_blocks.append(
                                            CitationContent(
                                                url=url,
                                                title=title or url,
                                                snippet=text[start_index:end_index] if start_index < end_index else "",
                                                relevance_score=annotation.get("relevance_score"),
                                            )
                                        )
                                # Handle file citations
                                elif ann_type == "file_citation":
                                    file_id = annotation.get("file_id", "")
                                    content_blocks.append(
                                        GenericContent(
                                            type="file_citation",
                                            file_id=file_id,
                                            **{k: v for k, v in annotation.items() if k not in ["type", "file_id"]},
                                        )
                                    )

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
                        function_arguments = json.dumps(function_arguments)

                    content_blocks.append(
                        ToolCallContent(
                            id=call_id,
                            name=function_name,
                            arguments=function_arguments,
                        )
                    )

            # Handle image_generation_call type items (built-in tool)
            elif item_type == "image_generation_call":
                # Extract the base64 image result and convert to ImageContent
                image_result = item_dict.get("result", "")

                # Extract image metadata from model_extra or item_dict
                extra_fields = item_dict.get("model_extra", {}) or {}
                image_size = extra_fields.get("size") or item_dict.get("size") or "1024x1536"
                image_quality = extra_fields.get("quality") or item_dict.get("quality") or "high"

                if image_result:
                    # Determine output format from image_output_params or default to png
                    output_format = self.image_output_params.get("output_format", "png")

                    # Create data URI for the generated image
                    data_uri = f"data:image/{output_format};base64,{image_result}"

                    # Create ImageContent with metadata stored in extra
                    image_content = ImageContent(
                        image_url=None,  # Not a URL, it's generated
                        data_uri=data_uri,
                        detail="high",  # Generated images are typically high detail
                    )
                    # Store generation metadata in the extra field
                    image_content.extra = {
                        "generated": True,
                        "size": image_size,
                        "quality": image_quality,
                        "output_format": output_format,
                    }
                    content_blocks.append(image_content)

                    # Calculate and accumulate image cost
                    cost, error = calculate_image_cost(
                        model="gpt-image-1",  # Default model for Responses API
                        size=image_size,
                        quality=image_quality,
                    )
                    if not error and cost > 0:
                        self._image_costs += cost
                else:
                    # No image result - store as GenericContent to preserve all fields
                    content_blocks.append(
                        GenericContent(
                            type="image_generation_call",
                            **{k: v for k, v in item_dict.items() if k != "type"},
                        )
                    )

            # Handle web_search_call type items (built-in tool)
            elif item_type == "web_search_call":
                # Web search calls contain the search execution metadata
                # The actual search results and citations come in output_text annotations
                search_id = item_dict.get("id", "")
                search_status = item_dict.get("status", "")

                # Store the web_search_call metadata
                # This allows tracking which searches were performed
                content_blocks.append(
                    GenericContent(
                        type="web_search_call",
                        id=search_id,
                        status=search_status,
                        **{k: v for k, v in item_dict.items() if k not in ["type", "id", "status"]},
                    )
                )

            # Handle web_search type items (search results with annotations/citations)
            elif item_type == "web_search":
                # Web search results may contain annotations with URLs
                # These are references to web pages that were retrieved
                annotations = item_dict.get("annotations", [])

                for annotation in annotations:
                    if isinstance(annotation, dict):
                        url = annotation.get("url", "")
                        title = annotation.get("title", "")
                        # Try multiple fields for the snippet/text content
                        snippet = (
                            annotation.get("text", "")
                            or annotation.get("snippet", "")
                            or annotation.get("description", "")
                        )

                        if url:
                            content_blocks.append(
                                CitationContent(
                                    url=url,
                                    title=title or url,
                                    snippet=snippet,
                                    relevance_score=annotation.get("relevance_score"),
                                )
                            )

                # If no annotations, store the raw web_search data as GenericContent
                if not annotations:
                    content_blocks.append(
                        GenericContent(
                            type="web_search",
                            **{k: v for k, v in item_dict.items() if k != "type"},
                        )
                    )

            # Handle web_search_result type items (individual search results)
            elif item_type == "web_search_result":
                # Individual web search result with URL, title, and content
                url = item_dict.get("url", "")
                title = item_dict.get("title", "")
                snippet = item_dict.get("snippet", "") or item_dict.get("content", "")

                if url:
                    content_blocks.append(
                        CitationContent(
                            url=url,
                            title=title or url,
                            snippet=snippet,
                            relevance_score=item_dict.get("relevance_score"),
                        )
                    )
                else:
                    # Store raw data if no URL
                    content_blocks.append(
                        GenericContent(
                            type="web_search_result",
                            **{k: v for k, v in item_dict.items() if k != "type"},
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
                        **{k: v for k, v in item_dict.items() if k not in ["type", "call_id", "status", "operation"]},
                    )
                )

            # Handle any other built-in tool calls ending in "_call"
            elif item_type and item_type.endswith("_call"):
                # Generic handler for unknown built-in tool calls
                # Preserve all fields using GenericContent
                tool_name = item_type.replace("_call", "")
                content_blocks.append(
                    GenericContent(
                        type=item_type,
                        tool_name=tool_name,
                        **{k: v for k, v in item_dict.items() if k != "type"},
                    )
                )

            # Handle unknown/other output types - store as GenericContent
            else:
                if item_type:
                    content_blocks.append(
                        GenericContent(
                            type=item_type,
                            **{k: v for k, v in item_dict.items() if k != "type"},
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

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Calculate token cost for this request
        token_cost = calculate_token_cost(
            model=response_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            custom_price=self._custom_price,
        )

        # Log warning if model not found in pricing table (and no custom price)
        if token_cost == 0.0 and prompt_tokens > 0 and self._custom_price is None:
            if response_model not in OAI_PRICE1K:
                logger.warning(
                    f"Model {response_model} not found in pricing table. Token cost will be 0. "
                    f"Use set_custom_price() to set custom pricing."
                )

        # Update cumulative token tracking
        self._token_costs += token_cost
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        # Calculate total cost (tokens + images)
        total_cost = token_cost + self._image_costs

        # Add cost breakdown to usage
        usage["token_cost"] = token_cost
        usage["image_cost"] = self._image_costs

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
                "token_cost": token_cost,
                "image_cost": int(self._image_costs),
                "cumulative_token_cost": self._token_costs,
                "cumulative_image_cost": self._image_costs,
            },
        )

        # Set total cost
        unified_response.cost = total_cost

        return unified_response

    def cost(self, response: UnifiedResponse) -> float:  # type: ignore[override]
        """Calculate cost from response.

        Returns the total cost stored in the response, which includes both
        token costs and image generation costs.

        Args:
            response: UnifiedResponse from create()

        Returns:
            Total cost in USD for the API call (token + image costs)

        Example:
            response = client.create({...})
            total_cost = client.cost(response)
            print(f"This request cost: ${total_cost:.4f}")
        """
        return response.cost or 0.0

    @staticmethod
    def get_usage(response: UnifiedResponse) -> dict[str, Any]:  # type: ignore[override]
        """Extract usage statistics from response.

        Args:
            response: UnifiedResponse from create()

        Returns:
            Dict with usage statistics including:
                - prompt_tokens: Number of input tokens
                - completion_tokens: Number of output tokens
                - total_tokens: Total tokens used
                - cost: Total cost in USD (token + image)
                - model: Model name
                - token_cost: Cost for tokens only
                - image_cost: Cost for image generation only
                - reasoning_tokens: (optional) Reasoning tokens for o3 models

        Example:
            response = client.create({...})
            usage = OpenAIResponsesV2Client.get_usage(response)
            print(f"Tokens: {usage['total_tokens']}, Cost: ${usage['cost']:.4f}")
        """
        usage = {
            "prompt_tokens": response.usage.get("prompt_tokens", 0),
            "completion_tokens": response.usage.get("completion_tokens", 0),
            "total_tokens": response.usage.get("total_tokens", 0),
            "cost": response.cost or 0.0,
            "model": response.model,
            "token_cost": response.usage.get("token_cost", 0.0),
            "image_cost": response.usage.get("image_cost", 0.0),
        }

        # Include reasoning tokens if present
        if "reasoning_tokens" in response.usage:
            usage["reasoning_tokens"] = response.usage["reasoning_tokens"]

        return usage

    def message_retrieval(self, response: UnifiedResponse) -> list[str] | list[dict[str, Any]]:  # type: ignore[override]
        """Retrieve messages from response in OpenAI-compatible format.

        Returns list of strings for text-only messages, or list of dicts when
        tool calls, images, or complex content is present.

        Args:
            response: UnifiedResponse from create()

        Returns:
            List of strings (for text-only) OR list of message dicts (for complex content)
        """
        result: list[str] | list[dict[str, Any]] = []

        for msg in response.messages:
            # Check for complex content (tool calls, images, citations)
            has_complex_content = any(
                isinstance(block, (ImageContent, ToolCallContent, CitationContent, GenericContent))
                for block in msg.content
            )

            if has_complex_content:
                # Return dict format for complex content
                content_items = []
                tool_calls = []

                for block in msg.content:
                    if isinstance(block, TextContent):
                        content_items.append({"type": "text", "text": block.text})
                    elif isinstance(block, ImageContent):
                        if block.data_uri:
                            content_items.append({
                                "type": "image",
                                "image_url": block.data_uri,
                                "generated": True,
                            })
                        elif block.image_url:
                            content_items.append({
                                "type": "image",
                                "image_url": block.image_url,
                            })
                    elif isinstance(block, ToolCallContent):
                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": block.arguments,
                            },
                        })
                    elif isinstance(block, CitationContent):
                        content_items.append({
                            "type": "citation",
                            "url": block.url,
                            "title": block.title,
                            "snippet": block.snippet,
                        })
                    elif isinstance(block, GenericContent):
                        # Preserve generic content as-is
                        content_items.append(block.get_all_fields())

                message_dict: dict[str, Any] = {
                    "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                    "content": content_items if content_items else None,
                }

                if tool_calls:
                    message_dict["tool_calls"] = tool_calls

                result.append(message_dict)
            else:
                # Simple text content - return string
                result.append(msg.get_text())

        return result

    def _unified_response_to_chat_completion(
        self,
        response: UnifiedResponse,
    ) -> ChatCompletionExtended:
        """Convert a UnifiedResponse to ChatCompletionExtended for V1 compatibility.

        This method transforms the rich UnifiedResponse format into a ChatCompletion-like
        object that works with OpenAIWrapper and other V1 consumers.

        Args:
            response: UnifiedResponse from create()

        Returns:
            ChatCompletionExtended object compatible with V1 interfaces

        Note:
            Some information may be lost in the conversion (e.g., citations, reasoning
            blocks are flattened to text). For full fidelity, use the V2 interface.
        """
        choices: list[Choice] = []

        for idx, msg in enumerate(response.messages):
            # Build message content
            content_parts: list[str] = []
            tool_calls: list[ChatCompletionMessageFunctionToolCall] = []

            for block in msg.content:
                if isinstance(block, TextContent):
                    content_parts.append(block.text)
                elif isinstance(block, ReasoningContent):
                    # Include reasoning in content for backward compatibility
                    content_parts.append(f"[Reasoning: {block.text}]")
                elif isinstance(block, ToolCallContent):
                    tool_calls.append(
                        ChatCompletionMessageFunctionToolCall(
                            id=block.id,
                            type="function",
                            function=Function(
                                name=block.name,
                                arguments=block.arguments,
                            ),
                        )
                    )
                elif isinstance(block, CitationContent):
                    # Append citation info to content
                    citation_text = f"[{block.title}]({block.url})"
                    if block.snippet:
                        citation_text += f": {block.snippet}"
                    content_parts.append(citation_text)
                elif isinstance(block, ImageContent):
                    # Reference generated images in content
                    if block.data_uri:
                        content_parts.append(f"[Generated image: {block.data_uri[:50]}...]")
                    elif block.image_url:
                        content_parts.append(f"[Image: {block.image_url}]")
                elif isinstance(block, GenericContent):
                    # Include generic content type info
                    content_parts.append(f"[{block.type}: {json.dumps(block.data)}]")

            # Determine finish_reason
            finish_reason: str = "stop"
            if tool_calls:
                finish_reason = "tool_calls"

            # Build ChatCompletionMessage
            message = ChatCompletionMessage(
                role="assistant",
                content="\n".join(content_parts) if content_parts else None,
                tool_calls=tool_calls if tool_calls else None,
            )

            # Build Choice
            choice = Choice(
                index=idx,
                message=message,
                finish_reason=finish_reason,  # type: ignore[arg-type]
                logprobs=None,
            )
            choices.append(choice)

        # Build CompletionUsage
        usage = CompletionUsage(
            prompt_tokens=response.usage.get("prompt_tokens", 0),
            completion_tokens=response.usage.get("completion_tokens", 0),
            total_tokens=response.usage.get("total_tokens", 0),
        )

        # Build ChatCompletionExtended
        chat_completion = ChatCompletionExtended(
            id=response.id,
            model=response.model,
            created=int(response.provider_metadata.get("created", 0) or 0),
            object="chat.completion",
            choices=choices,
            usage=usage,
            cost=response.cost,
        )

        return chat_completion

    def create_v1_compatible(
        self,
        params: dict[str, Any],
    ) -> ChatCompletionExtended:
        """Create a completion and return a ChatCompletion-like response for V1 compatibility.

        This method is designed for backward compatibility with code that expects
        the OpenAI ChatCompletion response format. It internally calls the V2 create()
        method and converts the UnifiedResponse to ChatCompletionExtended.

        Args:
            params: Request parameters (same as create() method)
                - model: Model name (e.g., "gpt-5", "gpt-4.1")
                - messages: List of message dicts
                - built_in_tools: Optional list of built-in tools
                - tools: Optional list of function tools
                - response_format: Optional Pydantic model or JSON schema
                - Other Responses API parameters

        Returns:
            ChatCompletionExtended object compatible with V1 interfaces:
                - .choices[0].message.content: Response text
                - .choices[0].message.tool_calls: Any tool calls
                - .usage.prompt_tokens, .usage.completion_tokens: Token counts
                - .cost: Total cost (if available)

        Example:
            # Use V1-compatible interface
            client = OpenAIResponsesV2Client(api_key="...")

            response = client.create_v1_compatible({
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "Hello"}]
            })

            # Access like a ChatCompletion
            print(response.choices[0].message.content)
            print(f"Tokens: {response.usage.total_tokens}")
            print(f"Cost: ${response.cost}")

        Example - With tools:
            response = client.create_v1_compatible({
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "Search for weather"}],
                "built_in_tools": ["web_search"]
            })

            # Tool calls are preserved
            if response.choices[0].message.tool_calls:
                for tc in response.choices[0].message.tool_calls:
                    print(f"Tool: {tc.function.name}")

        Note:
            For full access to rich content (citations, reasoning blocks, images),
            use the standard create() method which returns UnifiedResponse.
        """
        # Call the V2 create method
        unified_response = self.create(params)

        # Convert to ChatCompletion format
        return self._unified_response_to_chat_completion(unified_response)

    @staticmethod
    def unified_response_to_v1_messages(
        response: UnifiedResponse,
    ) -> list[str] | list[dict[str, Any]]:
        """Convert UnifiedResponse messages to V1-compatible format.

        This is a static utility for converting UnifiedResponse content to the
        simpler V1 message format when needed outside the client context.

        Args:
            response: UnifiedResponse from create()

        Returns:
            List of strings (text-only) or list of message dicts (complex content)

        Example:
            response = client.create({...})
            v1_messages = OpenAIResponsesV2Client.unified_response_to_v1_messages(response)
        """
        result: list[str] | list[dict[str, Any]] = []

        for msg in response.messages:
            # Check for complex content
            has_complex = any(
                isinstance(block, (ImageContent, ToolCallContent, CitationContent, GenericContent))
                for block in msg.content
            )

            if has_complex:
                content_items = []
                tool_calls = []

                for block in msg.content:
                    if isinstance(block, TextContent):
                        content_items.append({"type": "text", "text": block.text})
                    elif isinstance(block, ImageContent):
                        if block.data_uri:
                            content_items.append({
                                "type": "image",
                                "image_url": block.data_uri,
                            })
                        elif block.image_url:
                            content_items.append({
                                "type": "image",
                                "image_url": block.image_url,
                            })
                    elif isinstance(block, ToolCallContent):
                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": block.arguments,
                            },
                        })
                    elif isinstance(block, CitationContent):
                        content_items.append({
                            "type": "citation",
                            "url": block.url,
                            "title": block.title,
                            "snippet": block.snippet,
                        })
                    elif isinstance(block, GenericContent):
                        content_items.append(block.get_all_fields())

                msg_dict: dict[str, Any] = {
                    "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                    "content": content_items if content_items else None,
                }

                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls

                result.append(msg_dict)
            else:
                # Simple text
                result.append(msg.get_text())

        return result

    def is_v1_compatible(self) -> bool:
        """Check if this client supports V1 compatibility mode.

        Returns:
            True - this client always supports V1 compatibility via create_v1_compatible()
        """
        return True
