# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable
from typing import Any, TypedDict

from ollama import AsyncClient

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

OLLAMA_DEFAULT_HOST = "http://localhost:11434"


class CreateOptions(TypedDict, total=False):
    temperature: float | None
    top_p: float | None
    num_predict: int | None
    stop: str | list[str] | None
    seed: int | None
    frequency_penalty: float | None
    presence_penalty: float | None


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str,
        host: str = OLLAMA_DEFAULT_HOST,
        streaming: bool = False,
        create_options: CreateOptions | None = None,
    ) -> None:
        self._model = model
        self._host = host
        self._streaming = streaming
        self._create_options = {k: v for k, v in (create_options or {}).items() if v is not None}
        self._client = AsyncClient(host=host)

    async def __call__(
        self,
        *messages: BaseEvent,
        ctx: Context,
        tools: Iterable[Tool],
    ) -> None:
        ollama_messages = self._convert_messages(ctx.prompt, messages)
        tools_list = [self._tool_to_api(t) for t in tools]

        kwargs: dict[str, Any] = {}
        if self._create_options:
            kwargs["options"] = self._create_options

        if tools_list:
            kwargs["tools"] = tools_list

        if self._streaming:
            await self._call_streaming(ollama_messages, kwargs, ctx)
        else:
            await self._call_non_streaming(ollama_messages, kwargs, ctx)

    def _convert_messages(
        self,
        system_prompt: Iterable[str],
        messages: tuple[BaseEvent, ...],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = [{"content": p, "role": "system"} for p in system_prompt]

        for message in messages:
            if isinstance(message, ModelRequest):
                result.append({"role": "user", "content": message.content})
            elif isinstance(message, ModelResponse):
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.message.content if message.message else "",
                }
                tool_calls = [
                    {
                        "function": {
                            "name": c.name,
                            "arguments": json.loads(c.arguments) if c.arguments else {},
                        },
                    }
                    for c in message.tool_calls.calls
                ]
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                result.append(msg)
            elif isinstance(message, ToolResults):
                for r in message.results:
                    result.append({
                        "role": "tool",
                        "content": r.content,
                    })

        return result

    @staticmethod
    def _tool_to_api(t: Tool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": t.schema.function.name,
                "description": t.schema.function.description,
                "parameters": t.schema.function.parameters,
            },
        }

    async def _call_non_streaming(
        self,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
        ctx: Context,
    ) -> None:
        response = await self._client.chat(
            model=self._model,
            messages=messages,
            **kwargs,
        )

        msg = response.message

        if msg.thinking:
            await ctx.send(ModelReasoning(content=msg.thinking))

        model_msg: ModelMessage | None = None
        if msg.content:
            model_msg = ModelMessage(content=msg.content)
            await ctx.send(model_msg)

        calls = [
            ToolCall(
                id=f"call_{i}",
                name=tc.function.name,
                arguments=json.dumps(tc.function.arguments),
            )
            for i, tc in enumerate(msg.tool_calls or [])
        ]

        usage_dict = {
            "prompt_tokens": response.prompt_eval_count or 0,
            "completion_tokens": response.eval_count or 0,
            "total_tokens": (response.prompt_eval_count or 0) + (response.eval_count or 0),
        }

        await ctx.send(
            ModelResponse(
                message=model_msg,
                tool_calls=ToolCalls(calls=calls),
                usage=usage_dict,
            )
        )

    async def _call_streaming(
        self,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
        ctx: Context,
    ) -> None:
        response_stream = await self._client.chat(
            model=self._model,
            messages=messages,
            stream=True,
            **kwargs,
        )

        full_content: str = ""
        usage_dict: dict[str, Any] = {}
        calls: list[ToolCall] = []

        async for chunk in response_stream:
            msg = chunk.message

            if msg.thinking:
                await ctx.send(ModelReasoning(content=msg.thinking))

            if msg.content:
                full_content += msg.content
                await ctx.send(ModelMessageChunk(content=msg.content))

            for i, tc in enumerate(msg.tool_calls or []):
                calls.append(
                    ToolCall(
                        id=f"call_{len(calls) + i}",
                        name=tc.function.name,
                        arguments=json.dumps(tc.function.arguments),
                    )
                )

            if chunk.done:
                usage_dict = {
                    "prompt_tokens": chunk.prompt_eval_count or 0,
                    "completion_tokens": chunk.eval_count or 0,
                    "total_tokens": (chunk.prompt_eval_count or 0) + (chunk.eval_count or 0),
                }

        message: ModelMessage | None = None
        if full_content:
            message = ModelMessage(content=full_content)
            await ctx.send(message)

        await ctx.send(
            ModelResponse(
                message=message,
                tool_calls=ToolCalls(calls=calls),
                usage=usage_dict,
            )
        )
