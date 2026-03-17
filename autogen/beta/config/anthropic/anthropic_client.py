# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
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
    ModelResponse,
    ToolCall,
    ToolCalls,
)
from autogen.beta.tools.schemas import ToolSchema

from .mappers import convert_messages, tool_to_api


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
        prompt_caching: bool = True,
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
        self._prompt_caching = prompt_caching

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        *,
        tools: Iterable[ToolSchema],
    ) -> ModelResponse:
        anthropic_messages = convert_messages(messages)

        if context.prompt:
            system: Any = self._build_system(context.prompt)
        else:
            system = NOT_GIVEN

        if self._prompt_caching and anthropic_messages:
            self._inject_cache_control(anthropic_messages)

        tools_list = [tool_to_api(t) for t in tools]

        if self._streaming:
            async with self._client.messages.stream(
                **self._create_options,
                system=system,
                messages=anthropic_messages,
                tools=tools_list if tools_list else NOT_GIVEN,
            ) as stream:
                return await self._process_stream(stream, context)
        else:
            response = await self._client.messages.create(
                **self._create_options,
                system=system,
                messages=anthropic_messages,
                tools=tools_list if tools_list else NOT_GIVEN,
            )
            return await self._process_response(response, context)

    def _build_system(self, prompt: list[str]) -> Any:
        text = "\n\n".join(prompt)
        if self._prompt_caching:
            return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
        return text

    @staticmethod
    def _inject_cache_control(messages: list[dict[str, Any]]) -> None:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
                elif isinstance(content, list) and content:
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

    async def _process_response(
        self,
        response: Message,
        context: Context,
    ) -> ModelResponse:
        model_msg: ModelMessage | None = None
        calls: list[ToolCall] = []

        for block in response.content:
            if isinstance(block, ThinkingBlock):
                if block.thinking:
                    await context.send(ModelReasoning(content=block.thinking))

            elif isinstance(block, TextBlock):
                model_msg = ModelMessage(content=block.text)
                await context.send(model_msg)

            elif isinstance(block, ToolUseBlock):
                calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.input),
                    )
                )

        usage = response.usage.model_dump() if response.usage else {}

        return ModelResponse(
            message=model_msg,
            tool_calls=ToolCalls(calls=calls),
            usage=usage,
        )

    async def _process_stream(
        self,
        stream: Any,
        context: Context,
    ) -> ModelResponse:
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
                    await context.send(ModelMessageChunk(content=delta.text))

                elif delta_type == "thinking_delta":
                    await context.send(ModelReasoning(content=delta.thinking))

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
            await context.send(message)

        final_message = await stream.get_final_message()
        usage = final_message.usage.model_dump() if final_message.usage else {}

        return ModelResponse(
            message=message,
            tool_calls=ToolCalls(calls=calls),
            usage=usage,
        )
