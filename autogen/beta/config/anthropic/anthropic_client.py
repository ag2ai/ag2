# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable
from typing import Any, TypedDict

import httpx
from anthropic import NOT_GIVEN, AsyncAnthropic
from anthropic.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)

from autogen.beta.config.client import LLMClient
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


class CreateOptions(TypedDict, total=False):
    model: str
    max_tokens: int
    temperature: float | None
    top_p: float | None
    top_k: int | None
    stop_sequences: list[str] | None
    stream: bool
    metadata: dict[str, str] | None
    service_tier: str | None


class AnthropicClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        create_options: CreateOptions | None = None,
    ) -> None:
        self._client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout if timeout is not None else NOT_GIVEN,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
        )

        self._create_options = {k: v for k, v in (create_options or {}).items() if k != "stream"}
        self._streaming = (create_options or {}).get("stream", False)

    async def __call__(
        self,
        *messages: BaseEvent,
        ctx: Context,
        tools: Iterable[Tool],
    ) -> None:
        anthropic_messages = self._convert_messages(messages)
        system_prompt = "\n\n".join(ctx.prompt) if ctx.prompt else NOT_GIVEN

        tools_list = [self._tool_to_api(t) for t in tools]

        if self._streaming:
            async with self._client.messages.stream(
                **self._create_options,
                system=system_prompt,
                messages=anthropic_messages,
                tools=tools_list if tools_list else NOT_GIVEN,
            ) as stream:
                await self._process_stream(stream, ctx)
        else:
            response = await self._client.messages.create(
                **self._create_options,
                system=system_prompt,
                messages=anthropic_messages,
                tools=tools_list if tools_list else NOT_GIVEN,
            )
            await self._process_response(response, ctx)

    @staticmethod
    def _tool_to_api(t: Tool) -> dict[str, Any]:
        return {
            "name": t.schema.function.name,
            "description": t.schema.function.description,
            "input_schema": t.schema.function.parameters,
        }

    def _convert_messages(
        self,
        messages: tuple[BaseEvent, ...],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message, ModelRequest):
                result.append({
                    "role": "user",
                    "content": message.content,
                })
            elif isinstance(message, ModelResponse):
                content: list[dict[str, Any]] = []
                if message.message:
                    content.append({"type": "text", "text": message.message.content})
                for call in message.tool_calls.calls:
                    content.append({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": json.loads(call.arguments),
                    })
                if content:
                    result.append({"role": "assistant", "content": content})
            elif isinstance(message, ToolResults):
                tool_results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": r.parent_id,
                        "content": r.content,
                    }
                    for r in message.results
                ]
                result.append({"role": "user", "content": tool_results})

        return result

    async def _process_response(
        self,
        response: Message,
        ctx: Context,
    ) -> None:
        model_msg: ModelMessage | None = None
        calls: list[ToolCall] = []

        for block in response.content:
            if isinstance(block, ThinkingBlock):
                if block.thinking:
                    await ctx.send(ModelReasoning(content=block.thinking))

            elif isinstance(block, TextBlock):
                model_msg = ModelMessage(content=block.text)
                await ctx.send(model_msg)

            elif isinstance(block, ToolUseBlock):
                calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.input),
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
        stream: Any,
        ctx: Context,
    ) -> None:
        full_content: str = ""
        calls: list[ToolCall] = []

        current_tool: dict[str, Any] | None = None

        async for event in stream:
            event_type = getattr(event, "type", None)

            if event_type == "content_block_start":
                block = event.content_block
                if getattr(block, "type", None) == "tool_use":
                    current_tool = {
                        "id": block.id,
                        "name": block.name,
                        "arguments": "",
                    }

            elif event_type == "content_block_delta":
                delta = event.delta
                delta_type = getattr(delta, "type", None)

                if delta_type == "text_delta":
                    full_content += delta.text
                    await ctx.send(ModelMessageChunk(content=delta.text))

                elif delta_type == "thinking_delta":
                    await ctx.send(ModelReasoning(content=delta.thinking))

                elif delta_type == "input_json_delta" and current_tool is not None:
                    current_tool["arguments"] += delta.partial_json

            elif event_type == "content_block_stop":
                if current_tool is not None:
                    calls.append(
                        ToolCall(
                            id=current_tool["id"],
                            name=current_tool["name"],
                            arguments=current_tool["arguments"],
                        )
                    )
                    current_tool = None

        message: ModelMessage | None = None
        if full_content:
            message = ModelMessage(content=full_content)
            await ctx.send(message)

        final_message = await stream.get_final_message()
        usage = final_message.usage.model_dump() if final_message.usage else {}

        await ctx.send(
            ModelResponse(
                message=message,
                tool_calls=ToolCalls(calls=calls),
                usage=usage,
            )
        )
