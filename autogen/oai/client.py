# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import inspect
import json
import logging
import re
import sys
import uuid
import warnings
from collections import deque
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl
from pydantic.type_adapter import TypeAdapter

from autogen.oai.oai_models.chat_completion import ChatCompletionExtended

from ..cache import Cache
from ..code_utils import content_str
from ..doc_utils import export_module
from ..events.client_events import StreamEvent, UsageSummaryEvent
from ..exception_utils import ModelToolNotSupportedError
from ..import_utils import optional_import_block, require_optional_import
from ..io.base import IOStream
from ..llm_config import ModelClient
from ..llm_config.entry import LLMConfigEntry, LLMConfigEntryDict
from ..logger.logger_utils import get_current_ts
from ..runtime_logging import log_chat_completion, log_new_client, log_new_wrapper, logging_enabled
from .client_utils import FormatterProtocol, logging_formatter, merge_config_with_tools
from .openai_utils import OAI_PRICE1K, get_key, is_valid_api_key

TOOL_ENABLED = False
with optional_import_block() as openai_result:
    import openai

if openai_result.is_successful:
    # raises exception if openai>=1 is installed and something is wrong with imports
    from openai import APIError, APITimeoutError, AzureOpenAI, OpenAI
    from openai import __version__ as openai_version
    from openai.lib._parsing._completions import type_to_response_format_param
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    from openai.types.chat.chat_completion import ChatCompletionMessage, Choice  # type: ignore [attr-defined]
    from openai.types.chat.chat_completion_chunk import (
        ChoiceDeltaFunctionCall,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )
    from openai.types.completion import Completion
    from openai.types.completion_usage import CompletionUsage

    from autogen.oai.openai_responses import OpenAIResponsesClient

    if openai.__version__ >= "1.1.0":
        TOOL_ENABLED = True
    ERROR: ImportError | None = None
    from openai.lib._pydantic import _ensure_strict_json_schema
else:
    ERROR = ImportError("Please install openai>=1 and diskcache to use autogen.OpenAIWrapper.")  # type: ignore[assignment]

    # OpenAI = object
    # AzureOpenAI = object

with optional_import_block() as cerebras_result:
    from cerebras.cloud.sdk import (  # noqa
        AuthenticationError as cerebras_AuthenticationError,
        InternalServerError as cerebras_InternalServerError,
        RateLimitError as cerebras_RateLimitError,
    )

    from .cerebras import CerebrasClient

if cerebras_result.is_successful:
    cerebras_import_exception: ImportError | None = None
else:
    cerebras_AuthenticationError = cerebras_InternalServerError = cerebras_RateLimitError = Exception  # type: ignore[assignment,misc]  # noqa: N816
    cerebras_import_exception = ImportError("cerebras_cloud_sdk not found")

with optional_import_block() as gemini_result:
    from google.api_core.exceptions import (  # noqa
        InternalServerError as gemini_InternalServerError,
        ResourceExhausted as gemini_ResourceExhausted,
    )

    from .gemini import GeminiClient

if gemini_result.is_successful:
    gemini_import_exception: ImportError | None = None
else:
    gemini_InternalServerError = gemini_ResourceExhausted = Exception  # type: ignore[assignment,misc]  # noqa: N816
    gemini_import_exception = ImportError("google-genai not found")

with optional_import_block() as anthropic_result:
    from anthropic import (  # noqa
        InternalServerError as anthorpic_InternalServerError,
        RateLimitError as anthorpic_RateLimitError,
    )

    from .anthropic import AnthropicClient

if anthropic_result.is_successful:
    anthropic_import_exception: ImportError | None = None
else:
    anthorpic_InternalServerError = anthorpic_RateLimitError = Exception  # type: ignore[assignment,misc]  # noqa: N816
    anthropic_import_exception = ImportError("anthropic not found")

with optional_import_block() as mistral_result:
    from mistralai.models import (  # noqa
        HTTPValidationError as mistral_HTTPValidationError,
        SDKError as mistral_SDKError,
    )

    from .mistral import MistralAIClient

if mistral_result.is_successful:
    mistral_import_exception: ImportError | None = None
else:
    mistral_SDKError = mistral_HTTPValidationError = Exception  # noqa: N816
    mistral_import_exception = ImportError("mistralai not found")

with optional_import_block() as together_result:
    from together.error import TogetherException as together_TogetherException

    from .together import TogetherClient

if together_result.is_successful:
    together_import_exception: ImportError | None = None
else:
    together_TogetherException = Exception  # noqa: N816
    together_import_exception = ImportError("together not found")

with optional_import_block() as groq_result:
    from groq import (  # noqa
        APIConnectionError as groq_APIConnectionError,
        InternalServerError as groq_InternalServerError,
        RateLimitError as groq_RateLimitError,
    )

    from .groq import GroqClient

if groq_result.is_successful:
    groq_import_exception: ImportError | None = None
else:
    groq_InternalServerError = groq_RateLimitError = groq_APIConnectionError = Exception  # noqa: N816
    groq_import_exception = ImportError("groq not found")

with optional_import_block() as cohere_result:
    from cohere.errors import (  # noqa
        InternalServerError as cohere_InternalServerError,
        ServiceUnavailableError as cohere_ServiceUnavailableError,
        TooManyRequestsError as cohere_TooManyRequestsError,
    )

    from .cohere import CohereClient

if cohere_result.is_successful:
    cohere_import_exception: ImportError | None = None
else:
    cohere_InternalServerError = cohere_TooManyRequestsError = cohere_ServiceUnavailableError = Exception  # noqa: N816
    cohere_import_exception = ImportError("cohere not found")

with optional_import_block() as ollama_result:
    from ollama import (  # noqa
        RequestError as ollama_RequestError,
        ResponseError as ollama_ResponseError,
    )

    from .ollama import OllamaClient

if ollama_result.is_successful:
    ollama_import_exception: ImportError | None = None
else:
    ollama_RequestError = ollama_ResponseError = Exception  # type: ignore[assignment,misc]  # noqa: N816
    ollama_import_exception = ImportError("ollama not found")

with optional_import_block() as bedrock_result:
    from botocore.exceptions import (  # noqa
        BotoCoreError as bedrock_BotoCoreError,
        ClientError as bedrock_ClientError,
    )

    from .bedrock import BedrockClient

if bedrock_result.is_successful:
    bedrock_import_exception: ImportError | None = None
else:
    bedrock_BotoCoreError = bedrock_ClientError = Exception  # noqa: N816
    bedrock_import_exception = ImportError("botocore not found")

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Add the console handler.
    _ch = logging.StreamHandler(stream=sys.stdout)
    _ch.setFormatter(logging_formatter)
    logger.addHandler(_ch)

LEGACY_DEFAULT_CACHE_SEED = 41
LEGACY_CACHE_DIR = ".cache"
OPEN_API_BASE_URL_PREFIX = "https://api.openai.com"

OPENAI_FALLBACK_KWARGS = {
    "api_key",
    "organization",
    "project",
    "base_url",
    "websocket_base_url",
    "timeout",
    "max_retries",
    "default_headers",
    "default_query",
    "http_client",
    "_strict_response_validation",
    "webhook_secret",
}

AOPENAI_FALLBACK_KWARGS = {
    "azure_endpoint",
    "azure_deployment",
    "api_version",
    "api_key",
    "azure_ad_token",
    "azure_ad_token_provider",
    "organization",
    "websocket_base_url",
    "timeout",
    "max_retries",
    "default_headers",
    "default_query",
    "http_client",
    "_strict_response_validation",
    "base_url",
    "project",
    "webhook_secret",
}


@lru_cache(maxsize=128)
def log_cache_seed_value(cache_seed_value: str | int, client: ModelClient) -> None:
    logger.debug(f"Using cache with seed value {cache_seed_value} for client {client.__class__.__name__}")


class OpenAIEntryDict(LLMConfigEntryDict, total=False):
    api_type: Literal["openai"]

    price: list[float] | None
    tool_choice: Literal["none", "auto", "required"] | None
    user: str | None
    stream: bool
    verbosity: Literal["low", "medium", "high"] | None
    extra_body: dict[str, Any] | None
    reasoning_effort: Literal["none", "low", "minimal", "medium", "high", "xhigh"] | None
    max_completion_tokens: int | None
    workspace_dir: str | None
    allowed_paths: list[str] | None


class OpenAILLMConfigEntry(LLMConfigEntry):
    api_type: Literal["openai"] = "openai"

    price: list[float] | None = Field(default=None, min_length=2, max_length=2)
    tool_choice: Literal["none", "auto", "required"] | None = None
    user: str | None = None
    stream: bool = False
    verbosity: Literal["low", "medium", "high"] | None = None
    #   The extra_body parameter flows from OpenAILLMConfigEntry to the LLM request through this path:
    #   1. Config Definition: extra_body is defined in OpenAILLMConfigEntry (autogen/oai/client.py:248)
    #   2. Parameter Classification: It's classified as an OpenAI client parameter (not AG2-specific) via the openai_kwargs property (autogen/oai/client.py:752-758)
    #   3. Request Separation: In _separate_create_config() (autogen/oai/client.py:842), extra_body goes into create_config since it's not in the extra_kwargs set.
    #   4. API Call: The create_config becomes params and gets passed directly to OpenAI's create() method via **params (autogen/oai/client.py:551,658)
    extra_body: dict[str, Any] | None = (
        None  # For VLLM - See here: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters
    )
    # reasoning models - see: https://platform.openai.com/docs/api-reference/chat/create#chat-create-reasoning_effort
    reasoning_effort: Literal["none", "low", "minimal", "medium", "high", "xhigh"] | None = None
    max_completion_tokens: int | None = None

    def create_client(self) -> ModelClient:
        raise NotImplementedError("create_client method must be implemented in the derived class.")


class AzureOpenAIEntryDict(LLMConfigEntryDict, total=False):
    api_type: Literal["azure"]

    azure_ad_token_provider: str | Callable[[], str] | None
    stream: bool
    tool_choice: Literal["none", "auto", "required"] | None
    user: str | None
    reasoning_effort: Literal["low", "minimal", "medium", "high"] | None
    max_completion_tokens: int | None


class AzureOpenAILLMConfigEntry(LLMConfigEntry):
    api_type: Literal["azure"] = "azure"

    azure_ad_token_provider: str | Callable[[], str] | None = None
    stream: bool = False
    tool_choice: Literal["none", "auto", "required"] | None = None
    user: str | None = None
    # reasoning models - see:
    # - https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/reasoning
    # - https://learn.microsoft.com/en-us/azure/ai-services/openai/reference-preview
    reasoning_effort: Literal["low", "minimal", "medium", "high"] | None = None
    max_completion_tokens: int | None = None

    def create_client(self) -> ModelClient:
        raise NotImplementedError


class DeepSeekEntyDict(LLMConfigEntryDict, total=False):
    api_type: Literal["deepseek"]

    base_url: HttpUrl
    stream: bool
    tool_choice: Literal["none", "auto", "required"] | None


class DeepSeekLLMConfigEntry(LLMConfigEntry):
    api_type: Literal["deepseek"] = "deepseek"

    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    max_tokens: int = Field(8192, ge=1, le=8192)

    base_url: HttpUrl = HttpUrl("https://api.deepseek.com/v1")
    stream: bool = False
    tool_choice: Literal["none", "auto", "required"] | None = None

    def create_client(self) -> None:  # type: ignore [override]
        raise NotImplementedError("DeepSeekLLMConfigEntry.create_client is not implemented.")


class PlaceHolderClient:
    def __init__(self, config):
        self.config = config


@require_optional_import("openai>=1.66.2", "openai")
class OpenAIClient:
    """Follows the Client protocol and wraps the OpenAI client."""

    RESPONSE_USAGE_KEYS: list[str] = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    def __init__(self, client: OpenAI | AzureOpenAI, response_format: BaseModel | dict[str, Any] | None = None):
        self._oai_client = client
        self.response_format = response_format
        if (
            not isinstance(client, openai.AzureOpenAI)
            and str(client.base_url).startswith(OPEN_API_BASE_URL_PREFIX)
            and not is_valid_api_key(self._oai_client.api_key)
        ):
            logger.warning(
                "The API key specified is not a valid OpenAI format; it won't work with the OpenAI-hosted model."
            )

    def message_retrieval(self, response: ChatCompletion | Completion) -> list[str] | list[ChatCompletionMessage]:
        """Retrieve the messages from the response.

        Args:
            response (ChatCompletion | Completion): The response from openai.


        Returns:
            The message from the response.
        """
        choices = response.choices
        if isinstance(response, Completion):
            return [choice.text for choice in choices]  # type: ignore [union-attr]

        def _format_content(content: str | list[dict[str, Any]] | None) -> str:
            normalized_content = content_str(content)
            return (
                self.response_format.model_validate_json(normalized_content).format()
                if isinstance(self.response_format, FormatterProtocol)
                else normalized_content
            )

        if TOOL_ENABLED:
            return [  # type: ignore [return-value]
                (
                    choice.message  # type: ignore [union-attr]
                    if choice.message.function_call is not None or choice.message.tool_calls is not None  # type: ignore [union-attr]
                    else _format_content(choice.message.content)
                )  # type: ignore [union-attr]
                for choice in choices
            ]
        else:
            return [  # type: ignore [return-value]
                choice.message if choice.message.function_call is not None else _format_content(choice.message.content)  # type: ignore [union-attr]
                for choice in choices
            ]

    @staticmethod
    def _is_agent_name_error_message(message: str) -> bool:
        pattern = re.compile(r"Invalid 'messages\[\d+\]\.name': string does not match pattern.")
        return bool(pattern.match(message))

    @staticmethod
    def _move_system_message_to_beginning(messages: list[dict[str, Any]]) -> None:
        for msg in messages:
            if msg["role"] == "system":
                messages.insert(0, messages.pop(messages.index(msg)))
                break

    @staticmethod
    def _patch_messages_for_deepseek_reasoner(**kwargs: Any) -> Any:
        if (
            "model" not in kwargs
            or kwargs["model"] != "deepseek-reasoner"
            or "messages" not in kwargs
            or len(kwargs["messages"]) == 0
        ):
            return kwargs

        # The system message of deepseek-reasoner must be put on the beginning of the message sequence.
        OpenAIClient._move_system_message_to_beginning(kwargs["messages"])

        new_messages = []
        previous_role = None
        for message in kwargs["messages"]:
            if "role" in message:
                current_role = message["role"]

                # This model requires alternating roles
                if current_role == previous_role:
                    # Swap the role
                    if current_role == "user":
                        message["role"] = "assistant"
                    elif current_role == "assistant":
                        message["role"] = "user"

                previous_role = message["role"]

            new_messages.append(message)

        # The last message of deepseek-reasoner must be a user message
        # , or an assistant message with prefix mode on (but this is supported only for beta api)
        if new_messages[-1]["role"] != "user":
            new_messages.append({"role": "user", "content": "continue"})

        kwargs["messages"] = new_messages

        return kwargs

    @staticmethod
    def _handle_openai_bad_request_error(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any):
            try:
                kwargs = OpenAIClient._patch_messages_for_deepseek_reasoner(**kwargs)
                return func(*args, **kwargs)
            except openai.BadRequestError as e:
                response_json = e.response.json()
                # Check if the error message is related to the agent name. If so, raise a ValueError with a more informative message.
                if (
                    "error" in response_json
                    and "message" in response_json["error"]
                    and OpenAIClient._is_agent_name_error_message(response_json["error"]["message"])
                ):
                    error_message = (
                        f"This error typically occurs when the agent name contains invalid characters, such as spaces or special symbols.\n"
                        "Please ensure that your agent name follows the correct format and doesn't include any unsupported characters.\n"
                        "Check the agent name and try again.\n"
                        f"Here is the full BadRequestError from openai:\n{e.message}."
                    )
                    raise ValueError(error_message)

                raise e

        return wrapper

    @staticmethod
    def _convert_system_role_to_user(messages: list[dict[str, Any]]) -> None:
        for msg in messages:
            if msg.get("role", "") == "system":
                msg["role"] = "user"

    @staticmethod
    def _add_streaming_usage_to_params(params: dict[str, Any]) -> None:
        if params.get("stream", False):
            params.setdefault("stream_options", {}).setdefault("include_usage", True)

    def create(self, params: dict[str, Any]) -> ChatCompletion:
        """Create a completion for a given config using openai's client.

        Args:
            params: The params for the completion.

        Returns:
            The completion.
        """
        iostream = IOStream.get_default()

        is_structured_output = self.response_format is not None or "response_format" in params

        if is_structured_output:

            def _create_or_parse(*args, **kwargs):
                if "stream" in kwargs:
                    kwargs.pop("stream")
                    kwargs.pop("stream_options", None)

                if (
                    isinstance(kwargs["response_format"], dict)
                    and kwargs["response_format"].get("type") != "json_object"
                ):
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "schema": _ensure_strict_json_schema(
                                kwargs["response_format"], path=(), root=kwargs["response_format"]
                            ),
                            "name": "response_format",
                            "strict": True,
                        },
                    }
                else:
                    kwargs["response_format"] = type_to_response_format_param(
                        self.response_format or params["response_format"]
                    )

                return self._oai_client.chat.completions.create(*args, **kwargs)

            create_or_parse = _create_or_parse
        else:
            completions = self._oai_client.chat.completions if "messages" in params else self._oai_client.completions  # type: ignore [attr-defined]
            create_or_parse = completions.create
        # Wrap _create_or_parse with exception handling
        create_or_parse = OpenAIClient._handle_openai_bad_request_error(create_or_parse)

        # needs to be updated when the o3 is released to generalize
        is_o1 = "model" in params and params["model"].startswith("o1")

        is_mistral = "model" in params and "mistral" in params["model"]
        if is_mistral:
            OpenAIClient._convert_system_role_to_user(params["messages"])

        # If streaming is enabled and has messages, then iterate over the chunks of the response and is not using structured outputs.
        if params.get("stream", False) and "messages" in params and not is_o1 and not is_structured_output:
            # Usage will be returned as the last chunk
            OpenAIClient._add_streaming_usage_to_params(params)

            response_contents = [""] * params.get("n", 1)
            finish_reasons = [""] * params.get("n", 1)
            completion_tokens = 0

            # Prepare for potential function call
            full_function_call: dict[str, Any] | None = None
            full_tool_calls: list[dict[str, Any] | None] | None = None

            # Send the chat completion request to OpenAI's API and process the response in chunks
            chunks_id: str = ""
            chunks_model: str = ""
            chunks_created: int = 0
            chunks_usage_prompt_tokens: int = 0
            chunks_usage_completion_tokens: int = 0
            for chunk in create_or_parse(**params):
                if not isinstance(chunk, ChatCompletionChunk):
                    logger.debug(f"Skipping unexpected chunk type: {type(chunk)}")
                    continue

                chunk_cc: ChatCompletionChunk = chunk
                if chunk_cc.choices:
                    for choice in chunk_cc.choices:
                        content = choice.delta.content
                        tool_calls_chunks = choice.delta.tool_calls
                        finish_reasons[choice.index] = choice.finish_reason

                        # todo: remove this after function calls are removed from the API
                        # the code should work regardless of whether function calls are removed or not, but test_chat_functions_stream should fail
                        # begin block
                        function_call_chunk = (
                            choice.delta.function_call if hasattr(choice.delta, "function_call") else None
                        )
                        # Handle function call
                        if function_call_chunk:
                            # Handle function call
                            if function_call_chunk:
                                full_function_call, completion_tokens = OpenAIWrapper._update_function_call_from_chunk(
                                    function_call_chunk, full_function_call, completion_tokens
                                )
                            if not content:
                                continue
                        # end block

                        # Handle tool calls
                        if tool_calls_chunks:
                            for tool_calls_chunk in tool_calls_chunks:
                                # the current tool call to be reconstructed
                                ix = tool_calls_chunk.index
                                if full_tool_calls is None:
                                    full_tool_calls = []
                                if ix >= len(full_tool_calls):
                                    # in case ix is not sequential
                                    full_tool_calls = full_tool_calls + [None] * (ix - len(full_tool_calls) + 1)

                                full_tool_calls[ix], completion_tokens = OpenAIWrapper._update_tool_calls_from_chunk(
                                    tool_calls_chunk, full_tool_calls[ix], completion_tokens
                                )
                                if not content:
                                    continue

                        # End handle tool calls

                        # If content is present, print it to the terminal and update response variables
                        if content is not None:
                            iostream.send(StreamEvent(content=content))
                            response_contents[choice.index] += content
                            completion_tokens += 1
                        else:
                            pass
                else:
                    if chunk_cc.usage:
                        # Usage will be in the last chunk as we have set include_usage=True on stream_options
                        chunks_usage_prompt_tokens = getattr(chunk_cc.usage, "prompt_tokens", 0)
                        chunks_usage_completion_tokens = getattr(chunk_cc.usage, "completion_tokens", 0)

                if not chunks_id:
                    chunks_id = chunk_cc.id
                    chunks_model = chunk_cc.model
                    chunks_created = chunk_cc.created

            # Prepare the final ChatCompletion object based on the accumulated data
            response = ChatCompletion(
                id=chunks_id,
                model=chunks_model,
                created=chunks_created,
                object="chat.completion",
                choices=[],
                usage=CompletionUsage(
                    prompt_tokens=chunks_usage_prompt_tokens,
                    completion_tokens=chunks_usage_completion_tokens,
                    total_tokens=chunks_usage_prompt_tokens + chunks_usage_completion_tokens,
                ),
            )
            for i in range(len(response_contents)):
                if openai_version >= "1.5":  # pragma: no cover
                    # OpenAI versions 1.5.0 and above
                    choice = Choice(
                        index=i,
                        finish_reason=finish_reasons[i],
                        message=ChatCompletionMessage(
                            role="assistant",
                            content=response_contents[i],
                            function_call=full_function_call,
                            tool_calls=full_tool_calls,
                        ),
                        logprobs=None,
                    )
                else:
                    # OpenAI versions below 1.5.0
                    choice = Choice(  # type: ignore [call-arg]
                        index=i,
                        finish_reason=finish_reasons[i],
                        message=ChatCompletionMessage(
                            role="assistant",
                            content=response_contents[i],
                            function_call=full_function_call,
                            tool_calls=full_tool_calls,
                        ),
                    )

                response.choices.append(choice)
        else:
            # If streaming is not enabled, send a regular chat completion request
            params = params.copy()
            if is_o1:
                # add a warning that model does not support stream
                if params.get("stream", False):
                    warnings.warn(
                        f"The {params.get('model')} model does not support streaming. The stream will be set to False."
                    )
                if "tools" in params:
                    if params["tools"]:  # If tools exist, raise as unsupported
                        raise ModelToolNotSupportedError(params.get("model"))
                    else:
                        params.pop("tools")  # Remove empty tools list
                self._process_reasoning_model_params(params)
            params["stream"] = False
            response = create_or_parse(**params)
            # remove the system_message from the response and add it in the prompt at the start.
            if is_o1:
                for msg in params["messages"]:
                    if msg["role"] == "user" and msg["content"].startswith("System message: "):
                        msg["role"] = "system"
                        msg["content"] = msg["content"][len("System message: ") :]

        return response

    def _process_reasoning_model_params(self, params: dict[str, Any]) -> None:
        """Cater for the reasoning model (o1, o3..) parameters
        please refer: https://platform.openai.com/docs/guides/reasoning#limitations
        """
        # Unsupported parameters
        unsupported_params = [
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "logprobs",
            "top_logprobs",
            "logit_bias",
        ]
        model_name = params.get("model")
        for param in unsupported_params:
            if param in params:
                warnings.warn(f"`{param}` is not supported with {model_name} model and will be ignored.")
                params.pop(param)
        # Replace max_tokens with max_completion_tokens as reasoning tokens are now factored in
        # and max_tokens isn't valid
        if "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")

        # TODO - When o1-mini and o1-preview point to newer models (e.g. 2024-12-...), remove them from this list but leave the 2024-09-12 dated versions
        system_not_allowed = model_name in ("o1-mini", "o1-preview", "o1-mini-2024-09-12", "o1-preview-2024-09-12")

        if "messages" in params and system_not_allowed:
            # o1-mini (2024-09-12) and o1-preview (2024-09-12) don't support role='system' messages, only 'user' and 'assistant'
            # replace the system messages with user messages preappended with "System message: "
            for msg in params["messages"]:
                if msg["role"] == "system":
                    msg["role"] = "user"
                    msg["content"] = f"System message: {msg['content']}"

    def cost(self, response: ChatCompletion | Completion) -> float:
        """Calculate the cost of the response."""
        model = response.model
        if model not in OAI_PRICE1K:
            # log warning that the model is not found
            logger.warning(
                f'Model {model} is not found. The cost will be 0. In your config_list, add field {{"price" : [prompt_price_per_1k, completion_token_price_per_1k]}} for customized pricing.'
            )
            return 0

        n_input_tokens = response.usage.prompt_tokens if response.usage is not None else 0  # type: ignore [union-attr]
        n_output_tokens = response.usage.completion_tokens if response.usage is not None else 0  # type: ignore [union-attr]
        if n_output_tokens is None:
            n_output_tokens = 0
        tmp_price1K = OAI_PRICE1K[model]  # noqa: N806
        # First value is input token rate, second value is output token rate
        if isinstance(tmp_price1K, tuple):
            return (tmp_price1K[0] * n_input_tokens + tmp_price1K[1] * n_output_tokens) / 1000  # type: ignore [no-any-return]
        return tmp_price1K * (n_input_tokens + n_output_tokens) / 1000  # type: ignore [operator]

    @staticmethod
    def get_usage(response: ChatCompletion | Completion) -> dict[str, Any]:
        return {
            "prompt_tokens": response.usage.prompt_tokens if response.usage is not None else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage is not None else 0,
            "total_tokens": response.usage.total_tokens if response.usage is not None else 0,
            "cost": response.cost if hasattr(response, "cost") else 0,
            "model": response.model,
        }


@export_module("autogen")
class OpenAIWrapper:
    """A wrapper class for openai client."""

    extra_kwargs = {
        "agent",
        "cache",
        "cache_seed",
        "filter_func",
        "allow_format_str_template",
        "context",
        "api_version",
        "api_type",
        "tags",
        "price",
    }

    @property
    def openai_kwargs(self) -> set[str]:
        if openai_result.is_successful:
            return set(inspect.getfullargspec(OpenAI.__init__).kwonlyargs) | set(
                inspect.getfullargspec(AzureOpenAI.__init__).kwonlyargs
            )
        else:
            return OPENAI_FALLBACK_KWARGS | AOPENAI_FALLBACK_KWARGS

    total_usage_summary: dict[str, Any] | None = None
    actual_usage_summary: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        config_list: list[dict[str, Any]] | None = None,
        **base_config: Any,
    ):
        """Initialize the OpenAIWrapper.

        Args:
            config_list: a list of config dicts to override the base_config.
                They can contain additional kwargs as allowed in the [create](https://docs.ag2.ai/latest/docs/api-reference/autogen/OpenAIWrapper/#autogen.OpenAIWrapper.create) method. E.g.,

                ```python
                    config_list = [
                        {
                            "model": "gpt-4",
                            "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
                            "api_type": "azure",
                            "base_url": os.environ.get("AZURE_OPENAI_API_BASE"),
                            "api_version": "2024-02-01",
                        },
                        {
                            "model": "gpt-3.5-turbo",
                            "api_key": os.environ.get("OPENAI_API_KEY"),
                            "base_url": "https://api.openai.com/v1",
                        },
                        {
                            "model": "llama-7B",
                            "base_url": "http://127.0.0.1:8080",
                        },
                    ]
                ```

            base_config: base config. It can contain both keyword arguments for openai client
                and additional kwargs.
                When using OpenAI or Azure OpenAI endpoints, please specify a non-empty 'model' either in `base_config` or in each config of `config_list`.
        """
        if logging_enabled():
            log_new_wrapper(self, locals())
        openai_config, extra_kwargs = self._separate_openai_config(base_config)
        # It's OK if "model" is not provided in base_config or config_list
        # Because one can provide "model" at `create` time.

        self._clients: list[ModelClient] = []
        self._config_list: list[dict[str, Any]] = []

        # Determine routing_method from base_config only.
        self.routing_method = base_config.get("routing_method") or "fixed_order"
        self._round_robin_index = 0

        # Response metadata storage (for serializable responses)
        # Store metadata separately instead of mutating response objects
        self._response_metadata: dict[str, dict[str, Any]] = {}  # response_id → metadata
        self._response_buffer: deque[str] = deque(maxlen=100)  # Circular buffer of response IDs
        self._response_buffer_size = base_config.get("response_buffer_size", 100)
        if self._response_buffer_size != 100:
            self._response_buffer = deque(maxlen=self._response_buffer_size)

        # Remove routing_method from extra_kwargs after it has been used to set self.routing_method
        # This ensures it's not part of the individual client configurations that are based on extra_kwargs.
        extra_kwargs.pop("routing_method", None)

        if config_list:
            config_list = [config.copy() for config in config_list]  # make a copy before modifying
            for config_item in config_list:
                self._register_default_client(config_item, openai_config)
                # Construct current_config_extra_kwargs using the cleaned extra_kwargs
                # (which doesn't have routing_method from base_config)
                # and specific non-openai kwargs from config_item.
                config_item_specific_extras = {k: v for k, v in config_item.items() if k not in self.openai_kwargs}
                self._config_list.append({**extra_kwargs, **config_item_specific_extras})
        else:
            # For a single config passed via base_config (already in extra_kwargs)
            self._register_default_client(extra_kwargs, openai_config)
            # extra_kwargs has already had routing_method popped.
            self._config_list = [extra_kwargs]

        self.wrapper_id = id(self)

    def _separate_openai_config(self, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Separate the config into openai_config and extra_kwargs."""
        openai_config = {k: v for k, v in config.items() if k in self.openai_kwargs}
        extra_kwargs = {k: v for k, v in config.items() if k not in self.openai_kwargs}
        return openai_config, extra_kwargs

    def _separate_create_config(self, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Separate the config into create_config and extra_kwargs."""
        create_config = {k: v for k, v in config.items() if k not in self.extra_kwargs}
        extra_kwargs = {k: v for k, v in config.items() if k in self.extra_kwargs}
        return create_config, extra_kwargs

    def _store_response_metadata(
        self, response_id: str, client: ModelClient, config_id: int, pass_filter: bool
    ) -> None:
        """Store response metadata with circular buffer to prevent memory overflow.

        Args:
            response_id: Unique ID of the response (response.id)
            client: ModelClient that generated the response
            config_id: Index of the client in config_list
            pass_filter: Whether the response passed the filter function
        """
        # If buffer is full, remove oldest entry
        if len(self._response_buffer) >= self._response_buffer_size:
            oldest_id = self._response_buffer[0]  # Will be auto-removed by deque
            self._response_metadata.pop(oldest_id, None)

        # Add new metadata
        self._response_metadata[response_id] = {
            "client": client,
            "config_id": config_id,
            "pass_filter": pass_filter,
        }
        self._response_buffer.append(response_id)

    def _configure_azure_openai(self, config: dict[str, Any], openai_config: dict[str, Any]) -> None:
        openai_config["azure_deployment"] = openai_config.get("azure_deployment", config.get("model"))
        openai_config["azure_endpoint"] = openai_config.get("azure_endpoint", openai_config.pop("base_url", None))

        # Create a default Azure token provider if requested
        if openai_config.get("azure_ad_token_provider") == "DEFAULT":
            import azure.identity

            openai_config["azure_ad_token_provider"] = azure.identity.get_bearer_token_provider(
                azure.identity.DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
            )

    def _configure_openai_config_for_bedrock(self, config: dict[str, Any], openai_config: dict[str, Any]) -> None:
        """Update openai_config with AWS credentials from config."""
        required_keys = ["aws_access_key", "aws_secret_key", "aws_region"]
        optional_keys = ["aws_session_token", "aws_profile_name"]
        for key in required_keys:
            if key in config:
                openai_config[key] = config[key]
        for key in optional_keys:
            if key in config:
                openai_config[key] = config[key]

    def _configure_openai_config_for_vertextai(self, config: dict[str, Any], openai_config: dict[str, Any]) -> None:
        """Update openai_config with Google credentials from config."""
        required_keys = ["gcp_project_id", "gcp_region", "gcp_auth_token"]
        for key in required_keys:
            if key in config:
                openai_config[key] = config[key]

    def _configure_openai_config_for_gemini(self, config: dict[str, Any], openai_config: dict[str, Any]) -> None:
        """Update openai_config with additional gemini genai configs."""
        optional_keys = ["proxy"]
        for key in optional_keys:
            if key in config:
                openai_config[key] = config[key]

    def _register_default_client(self, config: dict[str, Any], openai_config: dict[str, Any]) -> None:
        """Create a client with the given config to override openai_config,
        after removing extra kwargs.

        For Azure models/deployment names there's a convenience modification of model removing dots in
        the it's value (Azure deployment names can't have dots). I.e. if you have Azure deployment name
        "gpt-35-turbo" and define model "gpt-3.5-turbo" in the config the function will remove the dot
        from the name and create a client that connects to "gpt-35-turbo" Azure deployment.
        """
        openai_config = {**openai_config, **{k: v for k, v in config.items() if k in self.openai_kwargs}}
        api_type = config.get("api_type")
        model_client_cls_name = config.get("model_client_cls")
        response_format = config.get("response_format")
        if model_client_cls_name is not None:
            # a config for a custom client is set
            # adding placeholder until the register_model_client is called with the appropriate class
            self._clients.append(PlaceHolderClient(config))
            # codeql[py/clear-text-logging-sensitive-data]
            logger.info(
                f"Detected custom model client in config: {model_client_cls_name}, model client can not be used until register_model_client is called."
            )
            # TODO: logging for custom client
        else:
            if api_type is not None and api_type.startswith("azure"):

                @require_optional_import("openai>=1.66.2", "openai")
                def create_azure_openai_client() -> AzureOpenAI:
                    self._configure_azure_openai(config, openai_config)
                    client = AzureOpenAI(**openai_config)
                    self._clients.append(OpenAIClient(client, response_format=response_format))  # type: ignore[arg-type]
                    return client

                client = create_azure_openai_client()
            elif api_type is not None and api_type.startswith("cerebras"):
                if cerebras_import_exception:
                    raise ImportError("Please install `cerebras_cloud_sdk` to use Cerebras OpenAI API.")
                client = CerebrasClient(response_format=response_format, **openai_config)
                self._clients.append(client)  # type: ignore[arg-type]
            elif api_type is not None and api_type.startswith("google"):
                if gemini_import_exception:
                    raise ImportError("Please install `google-genai` and 'vertexai' to use Google's API.")
                self._configure_openai_config_for_gemini(config, openai_config)
                client = GeminiClient(response_format=response_format, **openai_config)
                self._clients.append(client)  # type: ignore[arg-type]
            elif api_type is not None and api_type.startswith("anthropic"):
                if "api_key" not in config and "aws_region" in config:
                    self._configure_openai_config_for_bedrock(config, openai_config)
                elif "api_key" not in config and "gcp_region" in config:
                    self._configure_openai_config_for_vertextai(config, openai_config)
                if anthropic_import_exception:
                    raise ImportError("Please install `anthropic` to use Anthropic API.")
                client = AnthropicClient(response_format=response_format, **openai_config)
                self._clients.append(client)  # type: ignore[arg-type]
            elif api_type is not None and api_type.startswith("mistral"):
                if mistral_import_exception:
                    raise ImportError("Please install `mistralai` to use the Mistral.AI API.")
                client = MistralAIClient(response_format=response_format, **openai_config)
                self._clients.append(client)  # type: ignore[arg-type]
            elif api_type is not None and api_type.startswith("together"):
                if together_import_exception:
                    raise ImportError("Please install `together` to use the Together.AI API.")
                client = TogetherClient(response_format=response_format, **openai_config)
                self._clients.append(client)  # type: ignore[arg-type]
            elif api_type is not None and api_type.startswith("groq"):
                if groq_import_exception:
                    raise ImportError("Please install `groq` to use the Groq API.")
                client = GroqClient(response_format=response_format, **openai_config)
                self._clients.append(client)  # type: ignore[arg-type]
            elif api_type is not None and api_type.startswith("cohere"):
                if cohere_import_exception:
                    raise ImportError("Please install `cohere` to use the Cohere API.")
                client = CohereClient(response_format=response_format, **openai_config)
                self._clients.append(client)  # type: ignore[arg-type]
            elif api_type is not None and api_type.startswith("ollama"):
                if ollama_import_exception:
                    raise ImportError("Please install `ollama` and `fix-busted-json` to use the Ollama API.")
                client = OllamaClient(response_format=response_format, **openai_config)
                self._clients.append(client)  # type: ignore[arg-type]
            elif api_type is not None and api_type.startswith("bedrock"):
                self._configure_openai_config_for_bedrock(config, openai_config)
                if bedrock_import_exception:
                    raise ImportError("Please install `boto3` to use the Amazon Bedrock API.")
                client = BedrockClient(response_format=response_format, **openai_config)
                self._clients.append(client)  # type: ignore[arg-type]
            elif api_type is not None and api_type.startswith("openai_v2"):
                # OpenAI V2 Client with ModelClientV2 architecture (rich UnifiedResponse)
                from autogen.llm_clients import OpenAICompletionsClient as V2Client

                v2_client = V2Client(
                    api_key=openai_config.get("api_key"),
                    base_url=openai_config.get("base_url"),
                    timeout=openai_config.get("timeout", 60.0),
                    response_format=response_format,
                )
                self._clients.append(v2_client)  # type: ignore[arg-type]
                client = v2_client
            elif api_type is not None and api_type.startswith("responses"):
                # OpenAI Responses API (stateful). Reuse the same OpenAI SDK but call the `/responses` endpoint via the new client.
                @require_optional_import("openai>=1.66.2", "openai")
                def create_responses_client() -> OpenAI:
                    client = OpenAI(**openai_config)
                    self._clients.append(OpenAIResponsesClient(client, response_format=response_format))  # type: ignore[arg-type]
                    return client

                client = create_responses_client()
            else:

                @require_optional_import("openai>=1.66.2", "openai")
                def create_openai_client() -> OpenAI:
                    client = OpenAI(**openai_config)
                    self._clients.append(OpenAIClient(client, response_format))  # type: ignore[arg-type]
                    return client

                client = create_openai_client()

            if logging_enabled():
                log_new_client(client, self, openai_config)

    def register_model_client(self, model_client_cls: ModelClient, **kwargs: Any):
        """Register a model client.

        Args:
            model_client_cls: A custom client class that follows the ModelClient interface
            kwargs: The kwargs for the custom client class to be initialized with
        """
        existing_client_class = False
        for i, client in enumerate(self._clients):
            if isinstance(client, PlaceHolderClient):
                placeholder_config = client.config

                if placeholder_config.get("model_client_cls") == model_client_cls.__name__:
                    self._clients[i] = model_client_cls(placeholder_config, **kwargs)
                    return
            elif isinstance(client, model_client_cls):
                existing_client_class = True

        if existing_client_class:
            logger.warning(
                f"Model client {model_client_cls.__name__} is already registered. Add more entries in the config_list to use multiple model clients."
            )
        else:
            raise ValueError(
                f'Model client "{model_client_cls.__name__}" is being registered but was not found in the config_list. '
                f'Please make sure to include an entry in the config_list with "model_client_cls": "{model_client_cls.__name__}"'
            )

    @classmethod
    def instantiate(
        cls,
        template: str | Callable[[dict[str, Any]], str] | None,
        context: dict[str, Any] | None = None,
        allow_format_str_template: bool | None = False,
    ) -> str | None:
        if not context or template is None:
            return template  # type: ignore [return-value]
        if isinstance(template, str):
            return template.format(**context) if allow_format_str_template else template
        return template(context)

    def _construct_create_params(self, create_config: dict[str, Any], extra_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Prime the create_config with additional_kwargs."""
        # Validate the config
        prompt: str | None = create_config.get("prompt")
        messages: list[dict[str, Any]] | None = create_config.get("messages")
        if (prompt is None) == (messages is None):
            raise ValueError("Either prompt or messages should be in create config but not both.")
        context = extra_kwargs.get("context")
        if context is None:
            # No need to instantiate if no context is provided.
            return create_config
        # Instantiate the prompt or messages
        allow_format_str_template = extra_kwargs.get("allow_format_str_template", False)
        # Make a copy of the config
        params = create_config.copy()
        if prompt is not None:
            # Instantiate the prompt
            params["prompt"] = self.instantiate(prompt, context, allow_format_str_template)
        elif context:
            # Instantiate the messages
            params["messages"] = [
                (
                    {
                        **m,
                        "content": self.instantiate(m["content"], context, allow_format_str_template),
                    }
                    if m.get("content")
                    else m
                )
                for m in messages  # type: ignore [union-attr]
            ]
        return params

    def create(self, **config: Any) -> ModelClient.ModelClientResponseProtocol:
        """Make a completion for a given config using available clients.
        Besides the kwargs allowed in openai's [or other] client, we allow the following additional kwargs.
        The config in each client will be overridden by the config.

        Args:
            **config: The config for the completion.

        Raises:
            RuntimeError: If all declared custom model clients are not registered
            APIError: If any model client create call raises an APIError
        """
        # if ERROR:
        #     raise ERROR
        invocation_id = str(uuid.uuid4())
        last = len(self._clients) - 1
        # Check if all configs in config list are activated
        non_activated = [
            client.config["model_client_cls"] for client in self._clients if isinstance(client, PlaceHolderClient)
        ]
        if non_activated:
            raise RuntimeError(
                f"Model client(s) {non_activated} are not activated. Please register the custom model clients using `register_model_client` or filter them out form the config list."
            )

        ordered_clients_indices = list(range(len(self._clients)))
        if self.routing_method == "round_robin" and len(self._clients) > 0:
            ordered_clients_indices = (
                ordered_clients_indices[self._round_robin_index :] + ordered_clients_indices[: self._round_robin_index]
            )
            self._round_robin_index = (self._round_robin_index + 1) % len(self._clients)

        for i in ordered_clients_indices:
            # merge the input config with the i-th config in the config list
            client_config = self._config_list[i]
            full_config = merge_config_with_tools(config, client_config)

            # separate the config into create_config and extra_kwargs
            create_config, extra_kwargs = self._separate_create_config(full_config)
            # construct the create params
            params = self._construct_create_params(create_config, extra_kwargs)
            # get the cache_seed, filter_func and context
            cache_seed = extra_kwargs.get("cache_seed")
            cache = extra_kwargs.get("cache")
            filter_func = extra_kwargs.get("filter_func")
            context = extra_kwargs.get("context")
            agent = extra_kwargs.get("agent")
            price = extra_kwargs.get("price", None)
            if isinstance(price, list):
                price = tuple(price)
            elif isinstance(price, (float, int)):
                logger.warning(
                    "Input price is a float/int. Using the same price for prompt and completion tokens. Use a list/tuple if prompt and completion token prices are different."
                )
                price = (price, price)

            total_usage = None
            actual_usage = None

            cache_client = None
            if cache is not None:
                # Use the cache object if provided.
                cache_client = cache
            elif cache_seed is not None:
                # Legacy cache behavior, if cache_seed is given, use DiskCache.
                cache_client = Cache.disk(cache_seed, LEGACY_CACHE_DIR)

            client = self._clients[i]
            log_cache_seed_value(cache if cache is not None else cache_seed, client=client)

            if cache_client is not None:
                with cache_client as cache:
                    # Try to get the response from cache
                    key = get_key(
                        {
                            **params,
                            **{"response_format": json.dumps(TypeAdapter(params["response_format"]).json_schema())},
                        }
                        if "response_format" in params and not isinstance(params["response_format"], dict)
                        else params
                    )
                    request_ts = get_current_ts()

                    response: ChatCompletionExtended | None = cache.get(key, None)

                    if response is not None:
                        # Backward compatibility: set message_retrieval_function for ChatCompletionExtended
                        if hasattr(response, "message_retrieval_function"):
                            response.message_retrieval_function = client.message_retrieval

                        try:
                            response.cost
                        except AttributeError:
                            # update attribute if cost is not calculated
                            response.cost = client.cost(response)
                            cache.set(key, response)
                        total_usage = client.get_usage(response)

                        if logging_enabled():
                            # Log the cache hit
                            # TODO: log the config_id and pass_filter etc.
                            log_chat_completion(
                                invocation_id=invocation_id,
                                client_id=id(client),
                                wrapper_id=id(self),
                                agent=agent,
                                request=params,
                                response=response,
                                is_cached=1,
                                cost=response.cost if response.cost is not None else 0.0,
                                start_time=request_ts,
                            )

                        # check the filter
                        pass_filter = filter_func is None or filter_func(context=context, response=response)
                        if pass_filter or i == last:
                            # Store metadata for serializable responses
                            if hasattr(response, "id"):
                                self._store_response_metadata(response.id, client, i, pass_filter)

                            # Backward compatibility: set attributes on ChatCompletionExtended
                            if hasattr(response, "config_id"):
                                response.config_id = i
                            if hasattr(response, "pass_filter"):
                                response.pass_filter = pass_filter
                            self._update_usage(actual_usage=actual_usage, total_usage=total_usage)
                            return response
                        continue  # filter is not passed; try the next config
            try:
                request_ts = get_current_ts()
                response = client.create(params)
            except Exception as e:
                if openai_result.is_successful:
                    if APITimeoutError is not None and isinstance(e, APITimeoutError):
                        # logger.debug(f"config {i} timed out", exc_info=True)
                        if i == last:
                            raise TimeoutError(
                                "OpenAI API call timed out. This could be due to congestion or too small a timeout value. The timeout can be specified by setting the 'timeout' value (in seconds) in the llm_config (if you are using agents) or the OpenAIWrapper constructor (if you are using the OpenAIWrapper directly)."
                            ) from e
                    elif APIError is not None and isinstance(e, APIError):
                        error_code = getattr(e, "code", None)
                        if logging_enabled():
                            log_chat_completion(
                                invocation_id=invocation_id,
                                client_id=id(client),
                                wrapper_id=id(self),
                                agent=agent,
                                request=params,
                                response=f"error_code:{error_code}, config {i} failed",
                                is_cached=0,
                                cost=0,
                                start_time=request_ts,
                            )

                        if error_code == "content_filter":
                            # raise the error for content_filter
                            raise
                        # logger.debug(f"config {i} failed", exc_info=True)
                        if i == last:
                            raise
                    else:
                        raise
                else:
                    raise
            except (
                gemini_InternalServerError,
                gemini_ResourceExhausted,
                anthorpic_InternalServerError,
                anthorpic_RateLimitError,
                mistral_SDKError,
                mistral_HTTPValidationError,
                together_TogetherException,
                groq_InternalServerError,
                groq_RateLimitError,
                groq_APIConnectionError,
                cohere_InternalServerError,
                cohere_TooManyRequestsError,
                cohere_ServiceUnavailableError,
                ollama_RequestError,
                ollama_ResponseError,
                bedrock_BotoCoreError,
                bedrock_ClientError,
                cerebras_AuthenticationError,
                cerebras_InternalServerError,
                cerebras_RateLimitError,
            ):
                # logger.debug(f"config {i} failed", exc_info=True)
                if i == last:
                    raise
            else:
                # add cost calculation before caching no matter filter is passed or not
                if price is not None:
                    response.cost = self._cost_with_customized_price(response, price)
                else:
                    response.cost = client.cost(response)
                actual_usage = client.get_usage(response)
                total_usage = actual_usage.copy() if actual_usage is not None else total_usage
                self._update_usage(actual_usage=actual_usage, total_usage=total_usage)

                if cache_client is not None:
                    # Cache the response
                    with cache_client as cache:
                        cache.set(key, response)

                if logging_enabled():
                    # TODO: log the config_id and pass_filter etc.
                    log_chat_completion(
                        invocation_id=invocation_id,
                        client_id=id(client),
                        wrapper_id=id(self),
                        agent=agent,
                        request=params,
                        response=response,
                        is_cached=0,
                        cost=response.cost,
                        start_time=request_ts,
                    )

                # Store metadata instead of mutating response
                # Keep backward compatibility by setting message_retrieval_function for now
                if hasattr(response, "message_retrieval_function"):
                    response.message_retrieval_function = client.message_retrieval

                # check the filter
                pass_filter = filter_func is None or filter_func(context=context, response=response)
                if pass_filter or i == last:
                    # Store metadata for serializable responses
                    if hasattr(response, "id"):
                        self._store_response_metadata(response.id, client, i, pass_filter)

                    # Backward compatibility: set attributes on ChatCompletionExtended
                    if hasattr(response, "config_id"):
                        response.config_id = i
                    if hasattr(response, "pass_filter"):
                        response.pass_filter = pass_filter

                    # Return the response if it passes the filter or it is the last client
                    return response
                continue  # filter is not passed; try the next config
        raise RuntimeError("Should not reach here.")

    @staticmethod
    def _cost_with_customized_price(response: ChatCompletion | Completion, price_1k: tuple[float, float]) -> float:
        """If a customized cost is passed, overwrite the cost in the response."""
        n_input_tokens = response.usage.prompt_tokens if response.usage is not None else 0
        n_output_tokens = response.usage.completion_tokens if response.usage is not None else 0
        if n_output_tokens is None:
            n_output_tokens = 0
        return (n_input_tokens * price_1k[0] + n_output_tokens * price_1k[1]) / 1000

    @staticmethod
    def _update_dict_from_chunk(chunk: BaseModel, d: dict[str, Any], field: str) -> int:
        """Update the dict from the chunk.

        Reads `chunk.field` and if present updates `d[field]` accordingly.

        Args:
            chunk: The chunk.
            d: The dict to be updated in place.
            field: The field.

        Returns:
            The updated dict.

        """
        completion_tokens = 0
        assert isinstance(d, dict), d
        if hasattr(chunk, field) and getattr(chunk, field) is not None:
            new_value = getattr(chunk, field)
            if isinstance(new_value, (list, dict)):
                raise NotImplementedError(
                    f"Field {field} is a list or dict, which is currently not supported. "
                    "Only string and numbers are supported."
                )
            if field not in d:
                d[field] = ""
            if isinstance(new_value, str):
                d[field] += getattr(chunk, field)
            else:
                d[field] = new_value
            completion_tokens = 1

        return completion_tokens

    @staticmethod
    def _update_function_call_from_chunk(
        function_call_chunk: ChoiceDeltaToolCallFunction | ChoiceDeltaFunctionCall,
        full_function_call: dict[str, Any] | None,
        completion_tokens: int,
    ) -> tuple[dict[str, Any], int]:
        """Update the function call from the chunk.

        Args:
            function_call_chunk: The function call chunk.
            full_function_call: The full function call.
            completion_tokens: The number of completion tokens.

        Returns:
            The updated full function call and the updated number of completion tokens.

        """
        # Handle function call
        if function_call_chunk:
            if full_function_call is None:
                full_function_call = {}
            for field in ["name", "arguments"]:
                completion_tokens += OpenAIWrapper._update_dict_from_chunk(
                    function_call_chunk, full_function_call, field
                )

        if full_function_call:
            return full_function_call, completion_tokens
        else:
            raise RuntimeError("Function call is not found, this should not happen.")

    @staticmethod
    def _update_tool_calls_from_chunk(
        tool_calls_chunk: ChoiceDeltaToolCall,
        full_tool_call: dict[str, Any] | None,
        completion_tokens: int,
    ) -> tuple[dict[str, Any], int]:
        """Update the tool call from the chunk.

        Args:
            tool_calls_chunk: The tool call chunk.
            full_tool_call: The full tool call.
            completion_tokens: The number of completion tokens.

        Returns:
            The updated full tool call and the updated number of completion tokens.

        """
        # future proofing for when tool calls other than function calls are supported
        if tool_calls_chunk.type and tool_calls_chunk.type != "function":
            raise NotImplementedError(
                f"Tool call type {tool_calls_chunk.type} is currently not supported. Only function calls are supported."
            )

        # Handle tool call
        assert full_tool_call is None or isinstance(full_tool_call, dict), full_tool_call
        if tool_calls_chunk:
            if full_tool_call is None:
                full_tool_call = {}
            for field in ["index", "id", "type"]:
                completion_tokens += OpenAIWrapper._update_dict_from_chunk(tool_calls_chunk, full_tool_call, field)

            if hasattr(tool_calls_chunk, "function") and tool_calls_chunk.function:
                if "function" not in full_tool_call:
                    full_tool_call["function"] = None

                full_tool_call["function"], completion_tokens = OpenAIWrapper._update_function_call_from_chunk(
                    tool_calls_chunk.function, full_tool_call["function"], completion_tokens
                )

        if full_tool_call:
            return full_tool_call, completion_tokens
        else:
            raise RuntimeError("Tool call is not found, this should not happen.")

    def _update_usage(self, actual_usage, total_usage):
        def update_usage(usage_summary, response_usage):
            # go through RESPONSE_USAGE_KEYS and check that they are in response_usage and if not just return usage_summary
            for key in ModelClient.RESPONSE_USAGE_KEYS:
                if key not in response_usage:
                    return usage_summary

            model = response_usage["model"]
            cost = response_usage["cost"]
            prompt_tokens = response_usage["prompt_tokens"]
            completion_tokens = response_usage["completion_tokens"]
            if completion_tokens is None:
                completion_tokens = 0
            total_tokens = response_usage["total_tokens"]

            if usage_summary is None:
                usage_summary = {"total_cost": cost}
            else:
                usage_summary["total_cost"] += cost

            usage_summary[model] = {
                "cost": usage_summary.get(model, {}).get("cost", 0) + cost,
                "prompt_tokens": usage_summary.get(model, {}).get("prompt_tokens", 0) + prompt_tokens,
                "completion_tokens": usage_summary.get(model, {}).get("completion_tokens", 0) + completion_tokens,
                "total_tokens": usage_summary.get(model, {}).get("total_tokens", 0) + total_tokens,
            }
            return usage_summary

        if total_usage is not None:
            self.total_usage_summary = update_usage(self.total_usage_summary, total_usage)
        if actual_usage is not None:
            self.actual_usage_summary = update_usage(self.actual_usage_summary, actual_usage)

    def print_usage_summary(self, mode: str | list[str] = ["actual", "total"]) -> None:
        """Print the usage summary."""
        iostream = IOStream.get_default()

        if isinstance(mode, list):
            if len(mode) == 0 or len(mode) > 2:
                raise ValueError(f'Invalid mode: {mode}, choose from "actual", "total", ["actual", "total"]')
            if "actual" in mode and "total" in mode:
                mode = "both"
            elif "actual" in mode:
                mode = "actual"
            elif "total" in mode:
                mode = "total"

        iostream.send(
            UsageSummaryEvent(
                actual_usage_summary=self.actual_usage_summary, total_usage_summary=self.total_usage_summary, mode=mode
            )
        )

    def clear_usage_summary(self) -> None:
        """Clear the usage summary."""
        self.total_usage_summary = None
        self.actual_usage_summary = None

    def extract_text_or_completion_object(self, response: Any) -> list[str] | list[dict[str, Any]]:
        """Extract the text or ChatCompletion objects from a completion or chat response.

        Supports both legacy responses (with message_retrieval_function) and new serializable responses.

        Args:
            response: The response from any client (ChatCompletion, UnifiedResponse, etc.)

        Returns:
            A list of text, or a list of message dicts if function_call/tool_calls are present.
        """
        # Option 1: Legacy path - response has message_retrieval_function attached
        if hasattr(response, "message_retrieval_function") and callable(response.message_retrieval_function):
            return response.message_retrieval_function(response)  # type: ignore [misc]

        # Option 2: Use stored metadata to find client
        if hasattr(response, "id") and response.id in self._response_metadata:
            metadata = self._response_metadata[response.id]
            client = metadata["client"]
            return client.message_retrieval(response)

        # Option 3: Fallback - try to extract from response structure directly
        # This handles cases where response is not in buffer
        if hasattr(response, "choices"):
            # OpenAI-style response
            return [
                choice.message
                if hasattr(choice.message, "tool_calls") and choice.message.tool_calls
                else getattr(choice.message, "content", "")
                for choice in response.choices
            ]

        # Last resort: return empty list
        warnings.warn(
            f"Could not extract messages from response type {type(response).__name__}. "
            "Response may not be in metadata buffer or may not support extraction.",
            UserWarning,
        )
        return []


# -----------------------------------------------------------------------------
# New: Responses API config entry (OpenAI-hosted preview endpoint)
# -----------------------------------------------------------------------------


class OpenAIResponsesEntryDict(LLMConfigEntryDict, total=False):
    api_type: Literal["responses"]

    tool_choice: Literal["none", "auto", "required"] | None
    built_in_tools: list[str] | None


class OpenAIResponsesLLMConfigEntry(OpenAILLMConfigEntry):
    """LLMConfig entry for the OpenAI Responses API (stateful, tool-enabled).

    This reuses all the OpenAI fields but changes *api_type* so the wrapper can
    route traffic to the `client.responses` endpoint instead of
    `chat.completions`.  It inherits everything else – including reasoning
    fields – from *OpenAILLMConfigEntry* so users can simply set

    ```python
    {
        "api_type": "responses",  # <-- key differentiator
        "model": "o3",  # reasoning model
        "reasoning_effort": "medium",  # low / medium / high
        "stream": True,
    }
    ```
    """

    api_type: Literal["responses"] = "responses"
    tool_choice: Literal["none", "auto", "required"] | None = "auto"
    built_in_tools: list[Literal["web_search", "image_generation", "apply_patch", "apply_patch_async"]] | None = (
        None  # added type safety for built-in tools and IDE autocomplete
    )
    workspace_dir: str | None = None
    allowed_paths: list[str] | None = None

    def create_client(self) -> ModelClient:  # pragma: no cover
        raise NotImplementedError("Handled via OpenAIWrapper._register_default_client")


class OpenAIV2EntryDict(LLMConfigEntryDict, total=False):
    api_type: Literal["openai_v2"]


class OpenAIV2LLMConfigEntry(OpenAILLMConfigEntry):
    """LLMConfig entry for OpenAI V2 Client with ModelClientV2 architecture.

    This uses the new OpenAIResponsesClient from autogen.llm_clients which returns
    rich UnifiedResponse objects with typed content blocks (ReasoningContent,
    CitationContent, ToolCallContent, etc.).

    Example:
    ```python
    {
        "api_type": "openai_v2",  # <-- uses ModelClientV2 architecture
        "model": "gpt-4o-mini",  # vision-capable model
        "api_key": "...",
    }
    ```

    Benefits over standard OpenAI client:
    - Returns UnifiedResponse with typed content blocks
    - Access to reasoning blocks from o1/o3 models via response.reasoning
    - Forward-compatible with unknown content types via GenericContent
    - Rich metadata and citations support
    - Type-safe with Pydantic validation
    """

    api_type: Literal["openai_v2"] = "openai_v2"

    def create_client(self) -> ModelClient:  # pragma: no cover
        raise NotImplementedError("Handled via OpenAIWrapper._register_default_client")
