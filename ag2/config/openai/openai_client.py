# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any, Literal, TypedDict

import httpx2
from fast_depends.library.serializer import SerializerProto
from openai import DEFAULT_MAX_RETRIES, AsyncOpenAI, AsyncStream, Omit, not_given, omit
from openai.types import ChatModel
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionPredictionContentParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolUnionParam,
)
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
from openai.types.chat.completion_create_params import PromptCacheOptions, WebSearchOptions
from typing_extensions import Required

from ag2.config.client import LLMClient
from ag2.context import ConversationContext
from ag2.events import (
    BaseEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    Usage,
)
from ag2.exceptions import UnsupportedToolError
from ag2.response import ResponseProto
from ag2.tools.schemas import ToolSchema

from .mappers import convert_messages, normalize_usage, response_proto_to_schema, tool_to_api

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
# The closed sets the Chat Completions API takes for these options, restated as literals
# so a value it does not accept is rejected here rather than by the API.
Modality = Literal["text", "audio"]
ServiceTier = Literal["auto", "default", "flex", "scale", "priority", "fast"]
Verbosity = Literal["low", "medium", "high"]


class CreateOptions(TypedDict, total=False):
    model: Required[ChatModel | str]

    temperature: float | None | Omit
    top_p: float | None | Omit
    max_tokens: int | None | Omit
    max_completion_tokens: int | None | Omit
    frequency_penalty: float | None | Omit
    presence_penalty: float | None | Omit
    seed: int | None | Omit
    stop: str | list[str] | None | Omit
    n: int | None | Omit
    user: str | Omit
    logprobs: bool | None | Omit
    top_logprobs: int | None | Omit
    tool_choice: ChatCompletionToolChoiceOptionParam | Omit
    parallel_tool_calls: bool | Omit
    logit_bias: dict[str, int] | None | Omit
    metadata: dict[str, str] | None | Omit
    modalities: list[Modality] | None | Omit
    prediction: ChatCompletionPredictionContentParam | None | Omit
    prompt_cache_key: str | Omit
    prompt_cache_options: PromptCacheOptions | Omit
    safety_identifier: str | Omit
    service_tier: ServiceTier | None | Omit
    store: bool | None | Omit
    verbosity: Verbosity | None | Omit
    web_search_options: WebSearchOptions | Omit
    stream: bool
    stream_options: ChatCompletionStreamOptionsParam | None | Omit
    reasoning_effort: ReasoningEffort | None | Omit
    extra_body: dict[str, Any] | None


class OpenAIClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        websocket_base_url: str | None = None,
        timeout: Any = not_given,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: dict[str, str] | None = None,
        default_query: dict[str, object] | None = None,
        http_client: httpx2.AsyncClient | None = None,
        create_options: CreateOptions | None = None,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            project=project,
            base_url=base_url,
            websocket_base_url=websocket_base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
        )

        # Left as ``None`` rather than widened to an empty mapping: ``model`` is
        # required, so there is no such thing as an empty set of create options.
        self._create_options = create_options

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: "ConversationContext",
        *,
        tools: Iterable[ToolSchema],
        response_schema: ResponseProto[Any] | None,
        serializer: SerializerProto,
    ) -> ModelResponse:
        if response_schema and response_schema.system_prompt:
            prompt: Iterable[str] = chain(context.prompt, (response_schema.system_prompt,))
        else:
            prompt = context.prompt

        if self._create_options is None:
            raise ValueError("OpenAIClient was built without create options, so it has no model to call.")

        openai_messages = convert_messages(prompt, messages, serializer)

        openai_tools: list[ChatCompletionToolUnionParam] = [tool_to_api(t) for t in tools]

        response = await self._client.chat.completions.create(
            **self._create_options,
            response_format=response_proto_to_schema(response_schema) or omit,
            messages=openai_messages,
            tools=openai_tools or omit,
        )

        if isinstance(response, AsyncStream):
            return await self._process_stream(response, context)
        return await self._process_completion(response, context)

    async def _process_completion(
        self,
        completion: ChatCompletion,
        context: "ConversationContext",
    ) -> ModelResponse:
        model_msg: ModelMessage | None = None
        calls: list[ToolCallEvent] = []
        finish_reason: str | None = None

        # A completion can arrive with no choices — a content filter answers that way.
        # The turn still spent tokens and still has to come back as a response.
        if completion.choices:
            choice = completion.choices[0]
            msg = choice.message
            finish_reason = choice.finish_reason

            if r := getattr(msg, "reasoning", None):
                await context.send(ModelReasoning(r))

            if c := msg.content:
                model_msg = ModelMessage(c)
                await context.send(model_msg)

            for call in msg.tool_calls or ():
                if not isinstance(call, ChatCompletionMessageFunctionToolCall):
                    # ag2 sends function tools only, so there is nothing else to call back.
                    raise UnsupportedToolError(call.type, "openai-completions")
                calls.append(
                    ToolCallEvent(
                        id=call.id,
                        name=call.function.name,
                        arguments=call.function.arguments,
                    )
                )

        return ModelResponse(
            message=model_msg,
            tool_calls=ToolCallsEvent(calls),
            usage=normalize_usage(completion.usage) if completion.usage else Usage(),
            model=completion.model,
            provider="openai",
            finish_reason=finish_reason,
            response_id=completion.id,
        )

    async def _process_stream(
        self,
        response_stream: AsyncStream[ChatCompletionChunk],
        context: "ConversationContext",
    ) -> ModelResponse:
        full_content: str = ""
        usage = Usage()
        finish_reason: str | None = None
        resolved_model: str | None = None
        response_id: str | None = None

        # Accumulate tool calls by index (streaming sends partial updates per index)
        full_tool_calls: list[dict[str, str]] = []

        async for chunk in response_stream:
            # Usage is available only in the last chunk
            if chunk.usage:
                usage = normalize_usage(chunk.usage)

            if chunk.model:
                resolved_model = chunk.model

            response_id = chunk.id

            for choice in chunk.choices:
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta

                if r := getattr(delta, "reasoning_content", None):
                    await context.send(ModelReasoning(r))

                if c := delta.content:
                    full_content += c
                    await context.send(ModelMessageChunk(c))

                for tc in delta.tool_calls or []:
                    ix = tc.index
                    if ix >= len(full_tool_calls):
                        full_tool_calls.extend(
                            {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                            for _ in range(ix - len(full_tool_calls) + 1)
                        )
                    acc = full_tool_calls[ix]
                    if tc.id is not None:
                        acc["id"] = tc.id
                    if tc.function is not None:
                        if tc.function.name:
                            acc["name"] = tc.function.name
                        acc["arguments"] += tc.function.arguments or ""

        message: ModelMessage | None = None
        if full_content:
            message = ModelMessage(full_content)
            await context.send(message)

        calls = [
            ToolCallEvent(
                id=acc["id"],
                name=acc["name"],
                arguments=acc["arguments"],
            )
            for acc in full_tool_calls
        ]

        return ModelResponse(
            message=message,
            tool_calls=ToolCallsEvent(calls),
            usage=usage,
            model=resolved_model,
            provider="openai",
            finish_reason=finish_reason,
            response_id=response_id,
        )
