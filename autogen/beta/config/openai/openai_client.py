# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any, Literal, TypedDict

import httpx
from openai import DEFAULT_MAX_RETRIES, AsyncOpenAI, AsyncStream, not_given
from openai._types import Omit
from openai.types import ChatModel
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from typing_extensions import Required

from autogen.beta.config.client import LLMClient
from autogen.beta.context import ConversationContext
from autogen.beta.events import (
    BaseEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    Usage,
)
from autogen.beta.response import ResponseProto
from autogen.beta.tools.schemas import ToolSchema

from .mappers import convert_messages, normalize_usage, response_proto_to_schema, tool_to_api

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


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
    tool_choice: str | dict[str, Any] | Omit
    parallel_tool_calls: bool | Omit
    logit_bias: dict[str, int] | None | Omit
    metadata: dict[str, str] | None | Omit
    modalities: list[str] | None | Omit
    prediction: dict[str, Any] | None | Omit
    prompt_cache_key: str | Omit
    safety_identifier: str | Omit
    service_tier: str | None | Omit
    store: bool | None | Omit
    verbosity: str | None | Omit
    web_search_options: dict[str, Any] | Omit
    stream: bool
    stream_options: dict[str, Any] | Omit
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
        http_client: httpx.AsyncClient | None = None,
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

        self._create_options = create_options or {}
        self._streaming = self._create_options.get("stream", False)

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: "ConversationContext",
        *,
        tools: Iterable[ToolSchema],
        response_schema: ResponseProto | None,
    ) -> ModelResponse:
        if response_schema and response_schema.system_prompt:
            prompt: Iterable[str] = chain(context.prompt, (response_schema.system_prompt,))
        else:
            prompt = context.prompt

        openai_messages = convert_messages(prompt, messages)

        openai_tools = [tool_to_api(t) for t in tools]

        kwargs = {}
        if r := response_proto_to_schema(response_schema):
            kwargs["response_format"] = r

        response = await self._client.chat.completions.create(
            **self._create_options,
            **kwargs,
            messages=openai_messages,
            tools=openai_tools,
        )

        if self._streaming:
            result = await self._process_stream(response, context)
        else:
            result = await self._process_completion(response, context)

        return result

    async def _process_completion(
        self,
        completion: ChatCompletion,
        context: "ConversationContext",
    ) -> ModelResponse:
        for choice in completion.choices or ():
            msg = choice.message

            if r := getattr(msg, "reasoning", None):
                await context.send(ModelReasoning(r))

            model_msg: ModelMessage | None = None
            if c := msg.content:
                model_msg = ModelMessage(c)
                await context.send(model_msg)

            calls = [
                ToolCallEvent(
                    id=c.id,
                    name=c.function.name,
                    arguments=c.function.arguments,
                )
                for c in (msg.tool_calls or ())
            ]

            return ModelResponse(
                message=model_msg,
                tool_calls=ToolCallsEvent(calls),
                usage=normalize_usage(completion.usage) if completion.usage else Usage(),
                model=completion.model,
                provider="openai",
                finish_reason=choice.finish_reason,
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

        # Accumulate tool calls by index (streaming sends partial updates per index)
        full_tool_calls: list[dict[str, str]] = []

        async for chunk in response_stream:
            # Usage is available only in the last chunk
            if chunk.usage:
                usage = normalize_usage(chunk.usage)

            if chunk.model:
                resolved_model = chunk.model

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
                    if getattr(tc.function, "name", None):
                        acc["name"] = tc.function.name
                    args_chunk = getattr(tc.function, "arguments", None) or ""
                    acc["arguments"] += args_chunk

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
        )
