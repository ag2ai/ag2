# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any, TypedDict

import httpx
from openai import DEFAULT_MAX_RETRIES, AsyncOpenAI, not_given, omit
from openai.types import ChatModel
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_output_item import ImageGenerationCall
from typing_extensions import Required

from autogen.beta.config.client import LLMClient
from autogen.beta.context import Context
from autogen.beta.events import (
    BaseEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
)
from autogen.beta.response import ResponseProto
from autogen.beta.tools.schemas import ToolSchema

from .mappers import (
    events_to_responses_input,
    normalize_responses_usage,
    response_proto_to_text_config,
    tool_to_responses_api,
)


class CreateOptions(TypedDict, total=False):
    model: Required[ChatModel | str]

    temperature: float | None
    top_p: float | None
    max_output_tokens: int | None
    max_tool_calls: int | None
    parallel_tool_calls: bool
    top_logprobs: int | None
    store: bool | None
    metadata: dict[str, str] | None
    service_tier: str | None
    user: str
    stream: bool
    truncation: str | None


class OpenAIResponsesClient(LLMClient):
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
        context: Context,
        *,
        tools: Iterable[ToolSchema],
        response_schema: ResponseProto | None,
    ) -> ModelResponse:
        input_items = events_to_responses_input(messages)

        if response_schema and response_schema.system_prompt:
            prompt: Iterable[str] = chain(context.prompt, (response_schema.system_prompt,))
        else:
            prompt = context.prompt

        instructions = "\n".join(prompt) or None

        openai_tools = [tool_to_responses_api(t) for t in tools]

        kwargs: dict[str, Any] = {}
        if r := response_proto_to_text_config(response_schema):
            kwargs["text"] = r

        response = await self._client.responses.create(
            **self._create_options,
            **kwargs,
            input=input_items,
            instructions=instructions,
            tools=openai_tools or omit,
        )

        if self._streaming:
            return await self._process_stream(response, context)
        return await self._process_response(response, context)

    async def _process_response(
        self,
        response: Response,
        context: Context,
    ) -> ModelResponse:
        model_msg: ModelMessage | None = None
        calls: list[ToolCallEvent] = []
        images: list[bytes] = []

        for item in response.output:
            if isinstance(item, ResponseReasoningItem):
                for summary in item.summary or []:
                    if hasattr(summary, "text") and summary.text:
                        await context.send(ModelReasoning(content=summary.text))

            elif isinstance(item, ResponseOutputMessage):
                for part in item.content:
                    if hasattr(part, "text") and part.text:
                        model_msg = ModelMessage(content=part.text)
                        await context.send(model_msg)

            elif isinstance(item, ResponseFunctionToolCall):
                calls.append(
                    ToolCallEvent(
                        id=item.call_id,
                        name=item.name,
                        arguments=item.arguments,
                    )
                )

            elif isinstance(item, ImageGenerationCall) and item.result:
                images.append(base64.b64decode(item.result))

        usage = normalize_responses_usage(response.usage.model_dump() if response.usage else {})

        return ModelResponse(
            message=model_msg,
            tool_calls=ToolCallsEvent(calls=calls),
            usage=usage,
            model=response.model,
            provider="openai",
            finish_reason=response.status,
            images=images,
        )

    async def _process_stream(
        self,
        response_stream: Any,
        context: Context,
    ) -> ModelResponse:
        full_content: str = ""
        usage: dict[str, Any] = {}
        calls: list[ToolCallEvent] = []
        images: list[bytes] = []
        finish_reason: str | None = None
        resolved_model: str | None = None

        async for event in response_stream:
            event: ResponseStreamEvent

            if isinstance(event, ResponseTextDeltaEvent):
                full_content += event.delta
                await context.send(ModelMessageChunk(content=event.delta))

            elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
                calls.append(
                    ToolCallEvent(
                        id=event.item_id,
                        name=event.name,
                        arguments=event.arguments,
                    )
                )

            elif isinstance(event, ResponseCompletedEvent):
                if event.response.usage:
                    usage = event.response.usage.model_dump()
                finish_reason = event.response.status
                resolved_model = event.response.model
                for item in event.response.output:
                    if isinstance(item, ImageGenerationCall) and item.result:
                        images.append(base64.b64decode(item.result))

        message: ModelMessage | None = None
        if full_content:
            message = ModelMessage(content=full_content)
            await context.send(message)

        return ModelResponse(
            message=message,
            tool_calls=ToolCallsEvent(calls=calls),
            usage=normalize_responses_usage(usage),
            model=resolved_model,
            provider="openai",
            finish_reason=finish_reason,
            images=images,
        )
