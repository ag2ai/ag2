# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any, Literal, Required, TypedDict

import httpx
from openai import DEFAULT_MAX_RETRIES, AsyncOpenAI, AsyncStream, not_given
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from autogen.beta.config.client import LLMClient
from autogen.beta.context import Context
from autogen.beta.events import (
    BaseEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelResponse,
    ToolCall,
    ToolCalls,
)
from autogen.beta.tools import Tool

from .mappers import convert_messages, tool_to_api

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class CreateOptions(TypedDict, total=False):
    model: Required[str]

    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    max_completion_tokens: int | None
    response_format: dict[str, Any] | None
    frequency_penalty: float | None
    presence_penalty: float | None
    seed: int | None
    stop: str | list[str] | None
    n: int | None
    user: str
    logprobs: bool | None
    top_logprobs: int | None
    tool_choice: str | dict[str, Any]
    parallel_tool_calls: bool
    logit_bias: dict[str, int] | None
    metadata: dict[str, str] | None
    modalities: list[str] | None
    prediction: dict[str, Any] | None
    prompt_cache_key: str
    prompt_cache_retention: str | None
    safety_identifier: str
    service_tier: str | None
    store: bool | None
    verbosity: str | None
    web_search_options: dict[str, Any]
    stream: bool
    stream_options: dict[str, Any]
    reasoning_effort: ReasoningEffort


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
        *messages: BaseEvent,
        ctx: Context,
        tools: Iterable[Tool],
    ) -> None:
        openai_messages = convert_messages(ctx.prompt, messages)

        response = await self._client.chat.completions.create(
            **self._create_options,
            messages=openai_messages,
            tools=[tool_to_api(t) for t in tools],
        )

        if self._streaming:
            await self._process_stream(response, ctx)
        else:
            await self._process_completion(response, ctx)

    async def _process_completion(
        self,
        completion: ChatCompletion,
        ctx: Context,
    ) -> ToolCalls | ModelMessage:
        for choice in completion.choices or ():
            msg = choice.message

            if r := getattr(msg, "reasoning", None):
                await ctx.send(ModelReasoning(content=r))

            model_msg: ModelMessage | None = None
            if c := msg.content:
                model_msg = ModelMessage(content=c)
                await ctx.send(model_msg)

            calls = [
                ToolCall(
                    id=c.id,
                    name=c.function.name,
                    arguments=c.function.arguments,
                )
                for c in (msg.tool_calls or ())
            ]

            await ctx.send(
                ModelResponse(
                    message=model_msg,
                    tool_calls=ToolCalls(calls=calls),
                    usage=completion.usage.model_dump() if completion.usage else {},
                )
            )

    async def _process_stream(
        self,
        response_stream: AsyncStream[ChatCompletionChunk],
        ctx: Context,
    ) -> ToolCalls | ModelMessage:
        full_content: str = ""
        usage: dict[str, Any] = {}

        # Accumulate tool calls by index (streaming sends partial updates per index)
        full_tool_calls: list[dict[str, str]] = []

        async for chunk in response_stream:
            # Usage is available only in the last chunk
            if chunk.usage:
                usage = chunk.usage.model_dump()

            for choice in chunk.choices:
                delta = choice.delta

                if r := getattr(delta, "reasoning_content", None):
                    await ctx.send(ModelReasoning(content=r))

                if c := delta.content:
                    full_content += c
                    await ctx.send(ModelMessageChunk(content=c))

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
            message = ModelMessage(content=full_content)
            await ctx.send(message)

        calls = [
            ToolCall(
                id=acc["id"],
                name=acc["name"],
                arguments=acc["arguments"],
            )
            for acc in full_tool_calls
        ]

        await ctx.send(
            ModelResponse(
                message=message,
                tool_calls=ToolCalls(calls=calls),
                usage=usage,
            )
        )
