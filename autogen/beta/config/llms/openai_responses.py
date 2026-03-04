# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any, Literal, Required, TypedDict

import httpx
from openai import DEFAULT_MAX_RETRIES, AsyncOpenAI, not_given
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

from autogen.beta.context import Context
from autogen.beta.events import (
    BaseEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolCalls,
    ToolResults,
)
from autogen.beta.tools import Tool

from .client import LLMClient

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class CreateOptions(TypedDict, total=False):
    model: Required[str]

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
        *messages: BaseEvent,
        ctx: Context,
        tools: Iterable[Tool],
    ) -> None:
        input_items = self._convert_input(messages)
        instructions = "\n\n".join(ctx.prompt) if ctx.prompt else None

        tools_list = [self._tool_to_api(t) for t in tools]

        response = await self._client.responses.create(
            **self._create_options,
            input=input_items,
            instructions=instructions,
            tools=tools_list if tools_list else not_given,
        )

        if self._streaming:
            await self._process_stream(response, ctx)
        else:
            await self._process_response(response, ctx)

    @staticmethod
    def _tool_to_api(t: Tool) -> dict[str, Any]:
        d = t.schema.function.model_dump(exclude_none=True)
        d["type"] = "function"
        d["parameters"] = {"additionalProperties": False} | d["parameters"]
        return d

    def _convert_input(
        self,
        messages: tuple[BaseEvent, ...],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message, ModelRequest):
                result.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": message.content}],
                })
            elif isinstance(message, ModelResponse):
                # Reconstruct assistant message
                content: list[dict[str, Any]] = []
                if message.message:
                    content.append({"type": "output_text", "text": message.message.content})
                if content:
                    result.append({
                        "role": "assistant",
                        "content": content,
                    })
                # Add function call items from the response
                for call in message.tool_calls.calls:
                    result.append({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                    })
            elif isinstance(message, ToolResults):
                for r in message.results:
                    result.append({
                        "type": "function_call_output",
                        "call_id": r.parent_id,
                        "output": r.content,
                    })

        return result

    async def _process_response(
        self,
        response: Response,
        ctx: Context,
    ) -> None:
        model_msg: ModelMessage | None = None
        calls: list[ToolCall] = []

        for item in response.output:
            if isinstance(item, ResponseReasoningItem):
                for summary in item.summary or []:
                    if hasattr(summary, "text") and summary.text:
                        await ctx.send(ModelReasoning(content=summary.text))

            elif isinstance(item, ResponseOutputMessage):
                for part in item.content:
                    if hasattr(part, "text") and part.text:
                        model_msg = ModelMessage(content=part.text)
                        await ctx.send(model_msg)

            elif isinstance(item, ResponseFunctionToolCall):
                calls.append(
                    ToolCall(
                        id=item.call_id,
                        name=item.name,
                        arguments=item.arguments,
                    )
                )

        usage = response.usage.model_dump() if response.usage else {}

        await ctx.send(
            ModelResponse(
                message=model_msg,
                tool_calls=ToolCalls(calls=calls),
                usage=usage,
            )
        )

    async def _process_stream(
        self,
        response_stream: Any,
        ctx: Context,
    ) -> None:
        full_content: str = ""
        usage: dict[str, Any] = {}
        calls: list[ToolCall] = []

        async for event in response_stream:
            event: ResponseStreamEvent

            if isinstance(event, ResponseTextDeltaEvent):
                full_content += event.delta
                await ctx.send(ModelMessageChunk(content=event.delta))

            elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
                calls.append(
                    ToolCall(
                        id=event.item_id,
                        name=event.name,
                        arguments=event.arguments,
                    )
                )

            elif isinstance(event, ResponseCompletedEvent):
                if event.response.usage:
                    usage = event.response.usage.model_dump()

        message: ModelMessage | None = None
        if full_content:
            message = ModelMessage(content=full_content)
            await ctx.send(message)

        await ctx.send(
            ModelResponse(
                message=message,
                tool_calls=ToolCalls(calls=calls),
                usage=usage,
            )
        )
