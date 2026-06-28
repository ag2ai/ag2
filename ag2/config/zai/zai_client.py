# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import Iterable, Iterator, Sequence
from itertools import chain
from typing import Any, TypedDict

import httpx
from fast_depends.library.serializer import SerializerProto
from zai import ZaiClient
from zai.types.chat.chat_completion import Completion
from zai.types.chat.chat_completion_chunk import ChatCompletionChunk

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
from ag2.response import ResponseProto
from ag2.tools.schemas import ToolSchema

from .mappers import (
    PROVIDER,
    convert_messages,
    normalize_usage,
    response_proto_to_format,
    schema_instruction,
    tool_call_event,
    tool_call_index,
    tool_to_api,
)

_STREAM_DONE = object()


class CreateOptions(TypedDict, total=False):
    model: str
    stream: bool
    max_tokens: int
    temperature: float
    top_p: float
    stop: str | list[str]
    seed: int
    tool_choice: str | dict[str, Any]
    request_id: str
    user_id: str
    do_sample: bool
    meta: dict[str, str]
    sensitive_word_check: Any
    extra: Any
    timeout: float | httpx.Timeout | None
    watermark_enabled: bool
    tool_stream: bool
    reasoning_effort: str
    thinking: dict[str, Any]
    extra_body: dict[str, Any]
    extra_headers: dict[str, str]


def _merge_extra_body(options: CreateOptions) -> dict[str, Any]:
    kwargs = dict(options.get("extra_body") or {})
    for key, value in options.items():
        if key == "extra_body" or value is None:
            continue
        kwargs[key] = value
    return kwargs


class ZAIClient(LLMClient):
    """Z.AI client adapter for the synchronous zai-sdk chat completions API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        max_retries: int = 3,
        http_client: httpx.Client | None = None,
        custom_headers: dict[str, str] | None = None,
        disable_token_cache: bool = True,
        source_channel: str | None = None,
        create_options: CreateOptions | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._http_client = http_client
        self._custom_headers = custom_headers
        self._disable_token_cache = disable_token_cache
        self._source_channel = source_channel
        self._create_options = create_options or {}
        self._streaming = self._create_options.get("stream", False)
        self._client: ZaiClient | None = None

    def _get_client(self) -> ZaiClient:
        if self._client is None:
            kwargs: dict[str, Any] = {}
            if self._api_key is not None:
                kwargs["api_key"] = self._api_key
            if self._base_url is not None:
                kwargs["base_url"] = self._base_url
            if self._timeout is not None:
                kwargs["timeout"] = self._timeout
            kwargs["max_retries"] = self._max_retries
            if self._http_client is not None:
                kwargs["http_client"] = self._http_client
            if self._custom_headers is not None:
                kwargs["custom_headers"] = self._custom_headers
            kwargs["disable_token_cache"] = self._disable_token_cache
            if self._source_channel is not None:
                kwargs["source_channel"] = self._source_channel
            self._client = ZaiClient(**kwargs)
        return self._client

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: "ConversationContext",
        *,
        tools: Iterable[ToolSchema],
        response_schema: ResponseProto | None,
        serializer: SerializerProto,
    ) -> ModelResponse:
        schema_prompt = (
            (response_schema.system_prompt or schema_instruction(response_schema)) if response_schema else None
        )
        if schema_prompt:
            prompt: Iterable[str] = chain(context.prompt, (schema_prompt,))
        else:
            prompt = context.prompt

        zai_messages = convert_messages(prompt, messages, serializer)
        tools_list = [tool_to_api(t) for t in tools]
        kwargs: dict[str, Any] = {
            "messages": zai_messages,
            **_merge_extra_body(self._create_options),
        }

        if tools_list:
            kwargs["tools"] = tools_list
        if response_format := response_proto_to_format(response_schema):
            kwargs["response_format"] = response_format

        client = await asyncio.to_thread(self._get_client)

        if self._streaming:
            response = await asyncio.to_thread(client.chat.completions.create, **kwargs)
            return await self._process_stream(iter(response), context)

        response = await asyncio.to_thread(client.chat.completions.create, **kwargs)
        return await self._process_completion(response, context)

    async def _process_completion(self, response: Completion, context: "ConversationContext") -> ModelResponse:
        choices = response.choices or []
        choice = choices[0] if choices else None
        message = choice.message if choice else None

        if message and message.reasoning_content:
            await context.send(ModelReasoning(message.reasoning_content))

        model_msg: ModelMessage | None = None
        content = message.content if message else None
        if isinstance(content, str) and content:
            model_msg = ModelMessage(content)
            await context.send(model_msg)

        calls: list[ToolCallEvent] = []
        for tc in (message.tool_calls if message else None) or []:
            function = tc.function
            if function is None:
                continue
            call = tool_call_event(tc.id, function.name, function.arguments)
            if call is not None:
                calls.append(call)

        return ModelResponse(
            message=model_msg,
            tool_calls=ToolCallsEvent(calls),
            usage=normalize_usage(response.usage),
            model=response.model or self._create_options.get("model"),
            provider=PROVIDER,
            finish_reason=choice.finish_reason if choice else None,
        )

    async def _process_stream(
        self, stream: Iterator[ChatCompletionChunk], context: "ConversationContext"
    ) -> ModelResponse:
        full_content = ""
        usage = Usage()
        finish_reason: str | None = None
        model = self._create_options.get("model")
        tool_accs: dict[int, dict[str, str]] = {}

        while (chunk := await asyncio.to_thread(next, stream, _STREAM_DONE)) is not _STREAM_DONE:
            if chunk.model:
                model = chunk.model
            if chunk.usage:
                usage = normalize_usage(chunk.usage)

            for choice in chunk.choices or []:
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                delta = choice.delta
                if delta is None:
                    continue
                if delta.reasoning_content:
                    await context.send(ModelReasoning(delta.reasoning_content))
                if delta.content:
                    full_content += delta.content
                    await context.send(ModelMessageChunk(delta.content))
                for tc in delta.tool_calls or []:
                    index = tool_call_index(tc.index, len(tool_accs))
                    acc = tool_accs.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        acc["id"] = tc.id
                    function = tc.function
                    if function is None:
                        continue
                    if function.name:
                        acc["name"] = function.name
                    arguments = function.arguments
                    if arguments:
                        acc["arguments"] += arguments if isinstance(arguments, str) else json.dumps(arguments)

        calls: list[ToolCallEvent] = []
        for _, acc in sorted(tool_accs.items()):
            call = tool_call_event(acc["id"], acc["name"], acc["arguments"] or "{}")
            if call is not None:
                calls.append(call)

        message: ModelMessage | None = None
        if full_content:
            message = ModelMessage(full_content)
            await context.send(message)

        return ModelResponse(
            message=message,
            tool_calls=ToolCallsEvent(calls),
            usage=usage,
            model=model,
            provider=PROVIDER,
            finish_reason=finish_reason,
        )
