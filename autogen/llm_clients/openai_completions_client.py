# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
OpenAI Chat Completions API Client implementing ModelClientV2 and ModelClient protocols.

This client handles the OpenAI Chat Completions API (client.chat.completions.create)
which returns rich responses with:
- Reasoning blocks (o1, o3 models with 'reasoning' field)
- Tool calls and function execution
- Multimodal content (text, images)
- Standard chat messages

The client preserves all provider-specific features in UnifiedResponse format
and is compatible with AG2's agent system through ModelClient protocol.

Note: This uses the Chat Completions API, NOT the newer Responses API (client.responses.create).
"""

from typing import Any

from autogen.import_utils import optional_import_block

with optional_import_block() as openai_result:
    from openai import OpenAI

if openai_result.is_successful:
    openai_import_exception: ImportError | None = None
else:
    OpenAI = None  # type: ignore[assignment]
    openai_import_exception = ImportError(
        "Please install openai to use OpenAICompletionsClient. Install with: pip install openai"
    )

from ..llm_config.client import ModelClient
from .models import (
    AudioContent,
    CitationContent,
    GenericContent,
    ImageContent,
    ReasoningContent,
    TextContent,
    ToolCallContent,
    UnifiedMessage,
    UnifiedResponse,
    UserRoleEnum,
    VideoContent,
    normalize_role,
)


class OpenAICompletionsClient(ModelClient):
    """
    OpenAI Chat Completions API client implementing ModelClientV2 protocol.

    This client works with OpenAI's Chat Completions API (client.chat.completions.create)
    which returns structured output with reasoning blocks (o1/o3 models), tool calls, and more.

    Key Features:
    - Preserves reasoning blocks as ReasoningContent (o1/o3 models)
    - Handles tool calls and results
    - Supports multimodal content
    - Provides backward compatibility via create_v1_compatible()

    Example:
        client = OpenAICompletionsClient(api_key="...")

        # Get rich response with reasoning
        response = client.create({
            "model": "o1-preview",
            "messages": [{"role": "user", "content": "Explain quantum computing"}]
        })

        # Access reasoning blocks
        for reasoning in response.reasoning:
            print(f"Reasoning: {reasoning.reasoning}")

        # Get text response
        print(f"Answer: {response.text}")
    """

    RESPONSE_USAGE_KEYS: list[str] = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        response_format: Any = None,
        **kwargs: Any,
    ):
        """
        Initialize OpenAI Chat Completions API client.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            base_url: Custom base URL for OpenAI API
            timeout: Request timeout in seconds
            response_format: Optional response format (Pydantic model or JSON schema)
            **kwargs: Additional arguments passed to OpenAI client
        """
        if openai_import_exception is not None:
            raise openai_import_exception

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)  # type: ignore[misc]
        self._default_response_format = response_format
        self._cost_per_token = {
            # GPT-5 series - Latest flagship models (per million tokens)
            "gpt-5": {"prompt": 1.25 / 1_000_000, "completion": 10.00 / 1_000_000},
            "gpt-5-mini": {"prompt": 0.25 / 1_000_000, "completion": 2.00 / 1_000_000},
            "gpt-5-nano": {"prompt": 0.05 / 1_000_000, "completion": 0.40 / 1_000_000},
            # GPT-4o series - Multimodal flagship (per million tokens)
            "gpt-4o": {"prompt": 2.50 / 1_000_000, "completion": 10.00 / 1_000_000},
            "gpt-4o-mini": {"prompt": 0.15 / 1_000_000, "completion": 0.60 / 1_000_000},
            # GPT-4 Turbo (per million tokens)
            "gpt-4-turbo": {"prompt": 10.00 / 1_000_000, "completion": 30.00 / 1_000_000},
            # GPT-4 legacy (per million tokens)
            "gpt-4": {"prompt": 10.00 / 1_000_000, "completion": 30.00 / 1_000_000},
            # GPT-3.5 Turbo (per million tokens)
            "gpt-3.5-turbo": {"prompt": 0.50 / 1_000_000, "completion": 1.50 / 1_000_000},
            # o1 series - Reasoning models (keep existing if still valid)
            "o1-preview": {"prompt": 0.015 / 1000, "completion": 0.060 / 1000},
            "o1-mini": {"prompt": 0.003 / 1000, "completion": 0.012 / 1000},
            "o3-mini": {"prompt": 0.003 / 1000, "completion": 0.012 / 1000},
        }

    def create(self, params: dict[str, Any]) -> UnifiedResponse:  # type: ignore[override]
        """
        Create a completion and return UnifiedResponse with all features preserved.

        This method implements ModelClient.create() but returns UnifiedResponse instead
        of ModelClientResponseProtocol. The rich UnifiedResponse structure is compatible
        via duck typing - it has .model attribute and works with message_retrieval().

        Args:
            params: Request parameters including:
                - model: Model name (e.g., "o1-preview")
                - messages: List of message dicts
                - temperature: Optional temperature (not supported by o1 models)
                - max_tokens: Optional max completion tokens
                - tools: Optional tool definitions
                - response_format: Optional Pydantic BaseModel or JSON schema dict
                - **other OpenAI parameters

        Returns:
            UnifiedResponse with reasoning blocks, citations, and all content preserved
        """
        # Make a copy of params to avoid mutating the original
        params = params.copy()

        # Merge default response_format if not already in params
        if self._default_response_format is not None and "response_format" not in params:
            params["response_format"] = self._default_response_format

        # Process reasoning model parameters (o1/o3 models)
        if self._is_reasoning_model(params.get("model")):
            self._process_reasoning_model_params(params)

        # Check if response_format is a Pydantic BaseModel
        response_format = params.get("response_format")
        use_parse = self._is_pydantic_model(response_format)

        # Call OpenAI API - use parse() for Pydantic models, create() otherwise
        if use_parse:
            # parse() doesn't support stream parameter - remove it if present
            parse_params = params.copy()
            parse_params.pop("stream", None)
            response = self.client.chat.completions.parse(**parse_params)
        else:
            response = self.client.chat.completions.create(**params)

        # Transform to UnifiedResponse
        return self._transform_response(response, params.get("model", "unknown"), use_parse=use_parse)

    def _is_pydantic_model(self, obj: Any) -> bool:
        """
        Check if object is a Pydantic BaseModel class.

        Args:
            obj: Object to check

        Returns:
            True if obj is a Pydantic BaseModel class (not instance)
        """
        try:
            import inspect

            from pydantic import BaseModel

            return inspect.isclass(obj) and issubclass(obj, BaseModel)
        except (ImportError, TypeError):
            return False

    def _is_reasoning_model(self, model: str | None) -> bool:
        """
        Check if model is an o1/o3 reasoning model.

        Args:
            model: Model name to check

        Returns:
            True if model is an o1 or o3 reasoning model
        """
        if not model:
            return False
        return model.startswith(("o1", "o3"))

    def _process_reasoning_model_params(self, params: dict[str, Any]) -> None:
        """
        Process parameters for o1/o3 reasoning models.

        Reasoning models have special requirements and limitations:
        https://platform.openai.com/docs/guides/reasoning#limitations

        This method:
        1. Removes unsupported parameters (temperature, top_p, etc.)
        2. Converts max_tokens to max_completion_tokens
        3. Converts system messages to user messages for older o1 models
        4. Blocks tools for o1 models (they don't support function calling)
        5. Blocks streaming (not supported by o1 models)

        Args:
            params: Request parameters dict (modified in place)
        """
        import warnings

        model_name = params.get("model", "unknown")

        # 1. Remove unsupported parameters
        unsupported_params = [
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "logprobs",
            "top_logprobs",
            "logit_bias",
        ]
        for param in unsupported_params:
            if param in params:
                warnings.warn(
                    f"`{param}` is not supported with {model_name} model and will be ignored.",
                    UserWarning,
                    stacklevel=3,
                )
                params.pop(param)

        # 2. Replace max_tokens with max_completion_tokens
        # Reasoning tokens are now factored in, max_tokens isn't valid
        if "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")

        # 3. Convert system messages to user messages (for older o1 models)
        # Newer o1 models (like o1-2024-12-17) support system messages
        system_not_allowed = model_name in ("o1-mini", "o1-preview", "o1-mini-2024-09-12", "o1-preview-2024-09-12")

        if "messages" in params and system_not_allowed:
            # o1-mini/o1-preview don't support role='system' messages
            # Replace with user messages prepended with "System message: "
            for msg in params["messages"]:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    msg["role"] = "user"
                    msg["content"] = f"System message: {msg['content']}"

        # 4. Block tools for o1 models (they don't support function calling)
        if "tools" in params:
            if params["tools"]:
                warnings.warn(
                    f"Tools/function calling is not supported with {model_name} model and will be removed.",
                    UserWarning,
                    stacklevel=3,
                )
            params.pop("tools")

        # 5. Block streaming for o1 models
        if params.get("stream", False):
            warnings.warn(
                f"The {model_name} model does not support streaming. Setting stream=False.",
                UserWarning,
                stacklevel=3,
            )
            params["stream"] = False

    def _transform_response(self, openai_response: Any, model: str, use_parse: bool = False) -> UnifiedResponse:
        """
        Transform OpenAI ChatCompletion response to UnifiedResponse.

        This handles the standard ChatCompletion format including o1/o3 models
        which include a 'reasoning' field in the message object.

        Content handling:
        - Text content → TextContent
        - Reasoning blocks (o1/o3 models) → ReasoningContent
        - Tool calls → ToolCallContent
        - Parsed Pydantic objects (when use_parse=True) → GenericContent with 'parsed' type
        - Refusals (when use_parse=True) → GenericContent with 'refusal' type
        - Unknown message fields → GenericContent (forward compatibility)

        This ensures that new OpenAI features are preserved even if we don't have
        specific content types defined yet.

        Args:
            openai_response: Raw OpenAI API response (from create() or parse())
            model: Model name
            use_parse: Whether response came from parse() method (has .parsed field)

        Returns:
            UnifiedResponse with all content blocks properly typed
        """
        messages = []

        # Process each choice
        for choice in openai_response.choices:
            content_blocks = []
            message_obj = choice.message

            # Extract reasoning if present (o1/o3 models)
            if getattr(message_obj, "reasoning", None):
                content_blocks.append(
                    ReasoningContent(
                        reasoning=message_obj.reasoning,
                        summary=None,
                    )
                )

            # Extract parsed Pydantic object if present (from parse() method)
            if use_parse and getattr(message_obj, "parsed", None):
                # Store parsed object as GenericContent to preserve it
                parsed_obj = message_obj.parsed
                # Convert to dict for storage
                if hasattr(parsed_obj, "model_dump"):
                    parsed_dict = parsed_obj.model_dump()
                elif hasattr(parsed_obj, "dict"):
                    parsed_dict = parsed_obj.dict()
                else:
                    parsed_dict = {"value": str(parsed_obj)}

                content_blocks.append(GenericContent(type="parsed", parsed=parsed_dict))

            # Extract refusal if present (from parse() method)
            if use_parse and getattr(message_obj, "refusal", None):
                content_blocks.append(GenericContent(type="refusal", refusal=message_obj.refusal))

            # Extract text content
            # Note: OpenAI Chat Completions API always returns content as str, never list
            # (List content is only used in REQUEST messages for multimodal inputs)
            if message_obj.content:
                content_blocks.append(TextContent(text=message_obj.content))

            # Extract tool calls
            if getattr(message_obj, "tool_calls", None):
                for tool_call in message_obj.tool_calls:
                    content_blocks.append(
                        ToolCallContent(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        )
                    )

            # Extract citations if present (future-proofing)
            # Note: Not currently available in Chat Completions API
            if getattr(message_obj, "citations", None):
                for citation in message_obj.citations:
                    content_blocks.append(
                        CitationContent(
                            url=citation.get("url", ""),
                            title=citation.get("title", ""),
                            snippet=citation.get("snippet", ""),
                            relevance_score=citation.get("relevance_score"),
                        )
                    )

            # Handle any other unknown fields from OpenAI response as GenericContent
            # This ensures forward compatibility with new OpenAI features
            known_fields = {
                "role",
                "content",
                "reasoning",
                "tool_calls",
                "citations",
                "name",
                "function_call",
                "parsed",
                "refusal",
            }
            message_dict = message_obj.model_dump() if hasattr(message_obj, "model_dump") else {}
            for field_name, field_value in message_dict.items():
                if field_name not in known_fields and field_value is not None:
                    # Create GenericContent for unknown field
                    content_blocks.append(GenericContent(type=field_name, **{field_name: field_value}))

            # Create unified message with normalized role (convert to UserRoleEnum for known roles)
            messages.append(
                UnifiedMessage(
                    role=normalize_role(message_obj.role),
                    content=content_blocks,
                    name=getattr(message_obj, "name", None),
                )
            )

        # Extract usage information
        usage = {}
        if getattr(openai_response, "usage", None):
            usage = {
                "prompt_tokens": openai_response.usage.prompt_tokens,
                "completion_tokens": openai_response.usage.completion_tokens,
                "total_tokens": openai_response.usage.total_tokens,
            }

        # Build UnifiedResponse
        unified_response = UnifiedResponse(
            id=openai_response.id,
            model=openai_response.model,
            provider="openai",
            messages=messages,
            usage=usage,
            finish_reason=openai_response.choices[0].finish_reason if openai_response.choices else None,
            status="completed",
            provider_metadata={
                "created": getattr(openai_response, "created", None),
                "system_fingerprint": getattr(openai_response, "system_fingerprint", None),
                "service_tier": getattr(openai_response, "service_tier", None),
            },
        )

        # Calculate cost
        unified_response.cost = self.cost(unified_response)

        return unified_response

    def create_v1_compatible(self, params: dict[str, Any]) -> Any:
        """
        Create completion in backward-compatible ChatCompletionExtended format.

        This method provides compatibility with existing AG2 code that expects
        ChatCompletionExtended format. Note that reasoning blocks and citations
        will be lost in this format.

        Args:
            params: Same parameters as create()

        Returns:
            ChatCompletionExtended-compatible dict (flattened response)

        Warning:
            This method loses information (reasoning blocks, citations) when
            converting to the legacy format. Prefer create() for new code.
        """
        # Get rich response
        unified_response = self.create(params)

        # Convert to legacy format (simplified - would need full ChatCompletionExtended in practice)
        # Extract role and convert UserRoleEnum to string
        role = unified_response.messages[0].role if unified_response.messages else UserRoleEnum.ASSISTANT
        role_str = role.value if isinstance(role, UserRoleEnum) else role

        return {
            "id": unified_response.id,
            "model": unified_response.model,
            "created": unified_response.provider_metadata.get("created"),
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": role_str,
                        "content": unified_response.text,
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

        model = response.model
        prompt_tokens = response.usage.get("prompt_tokens", 0)
        completion_tokens = response.usage.get("completion_tokens", 0)

        # Find pricing for model (exact match or prefix)
        pricing = None
        for model_key in self._cost_per_token:
            if model.startswith(model_key):
                pricing = self._cost_per_token[model_key]
                break

        if not pricing:
            # Unknown model - use default pricing (GPT-4 Turbo level, per million tokens)
            pricing = {"prompt": 10.00 / 1_000_000, "completion": 30.00 / 1_000_000}

        return (prompt_tokens * pricing["prompt"]) + (completion_tokens * pricing["completion"])

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
        tool calls, function calls, or complex content is present.

        This matches the behavior of the legacy OpenAIClient which returns:
        - Strings for simple text responses
        - ChatCompletionMessage objects (as dicts) when tool_calls/function_call present

        The returned dicts follow OpenAI's ChatCompletion message format:
        {
            "role": "assistant",
            "content": "text content or None",
            "tool_calls": [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}],
            "name": "agent_name" (optional)
        }

        Args:
            response: UnifiedResponse from create()

        Returns:
            List of strings (for text-only) OR list of message dicts (for tool calls/complex content)
        """
        result: list[str] | list[dict[str, Any]] = []

        for msg in response.messages:
            # Check for tool calls
            tool_calls = msg.get_tool_calls()

            # Check for complex/multimodal content that needs dict format
            has_complex_content = any(
                isinstance(block, (ImageContent, AudioContent, VideoContent)) for block in msg.content
            )

            if tool_calls or has_complex_content:
                # Return OpenAI-compatible dict format
                message_dict: dict[str, Any] = {
                    "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                    "content": msg.get_text() or None,
                }

                # Add optional fields
                if msg.name:
                    message_dict["name"] = msg.name

                # Add tool calls in OpenAI format
                if tool_calls:
                    message_dict["tool_calls"] = [
                        {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                        for tc in tool_calls
                    ]

                # Handle multimodal content - convert to OpenAI content array format
                if has_complex_content:
                    message_dict["content"] = self._convert_to_openai_content_array(msg)

                result.append(message_dict)
            else:
                # Simple text content - return string
                result.append(msg.get_text())

        return result

    def _convert_to_openai_content_array(self, msg: UnifiedMessage) -> list[dict[str, Any]]:
        """
        Convert UnifiedMessage content blocks to OpenAI content array format.

        This handles multimodal content (text, images, audio, video) and converts
        it to the format expected by OpenAI's API for input messages.

        Args:
            msg: UnifiedMessage with content blocks

        Returns:
            List of content dicts in OpenAI format
        """
        content_array = []

        for block in msg.content:
            if isinstance(block, TextContent):
                content_array.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageContent):
                # OpenAI image format
                image_url = block.url or f"data:{block.mime_type or 'image/jpeg'};base64,{block.data}"
                content_array.append({
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": block.detail or "auto"},
                })
            elif isinstance(block, AudioContent):
                # OpenAI doesn't have standard audio input format yet
                # Fall back to text representation
                content_array.append({"type": "text", "text": block.get_text()})
            elif isinstance(block, VideoContent):
                # OpenAI doesn't have standard video input format yet
                # Fall back to text representation
                content_array.append({"type": "text", "text": block.get_text()})
            # Skip ToolCallContent, ReasoningContent - handled separately in message_dict

        # If no content blocks were converted, return text fallback
        return content_array if content_array else [{"type": "text", "text": msg.get_text()}]
