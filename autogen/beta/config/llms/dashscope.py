# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable
from typing import Any, TypedDict

import dashscope
from dashscope.aigc.generation import AioGeneration

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

DASHSCOPE_INTL_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"


class CreateOptions(TypedDict, total=False):
    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    stop: str | list[str] | None
    seed: int | None
    frequency_penalty: float | None
    presence_penalty: float | None


class DashScopeClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = DASHSCOPE_INTL_BASE_URL,
        streaming: bool = False,
        create_options: CreateOptions | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._streaming = streaming
        self._create_options = {k: v for k, v in (create_options or {}).items() if v is not None}

    async def __call__(
        self,
        *messages: BaseEvent,
        ctx: Context,
        tools: Iterable[Tool],
    ) -> None:
        ds_messages = self._convert_messages(ctx.prompt, messages)
        tools_list = [self._tool_to_api(t) for t in tools]

        kwargs: dict[str, Any] = {
            **self._create_options,
            "result_format": "message",
        }

        if tools_list:
            kwargs["tools"] = tools_list

        # Set the base URL for this call (SDK uses a global)
        dashscope.base_http_api_url = self._base_url

        if self._streaming:
            await self._call_streaming(ds_messages, kwargs, ctx)
        else:
            await self._call_non_streaming(ds_messages, kwargs, ctx)

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
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
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
                        "tool_call_id": r.parent_id,
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
        response = await AioGeneration.call(
            model=self._model,
            messages=messages,
            api_key=self._api_key,
            **kwargs,
        )

        if response.status_code != 200:
            raise RuntimeError(f"DashScope error: {response.code} - {response.message}")

        choice = response.output.choices[0]
        msg = choice.message

        # Handle reasoning content (thinking models)
        # Use .get() because SDK's DictMixin.__getattr__ raises KeyError, not AttributeError
        # (Mark Sze) Have raised a PR to fix: https://github.com/dashscope/dashscope-sdk-python/pull/115
        if reasoning := msg.get("reasoning_content"):
            await ctx.send(ModelReasoning(content=reasoning))

        model_msg: ModelMessage | None = None
        if content := msg.get("content"):
            model_msg = ModelMessage(content=content)
            await ctx.send(model_msg)

        calls = []
        for tc in msg.get("tool_calls") or []:
            args = tc["function"]["arguments"]
            calls.append(
                ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=args if isinstance(args, str) else json.dumps(args),
                )
            )

        usage = response.usage or {}
        usage_dict = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
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
        responses = await AioGeneration.call(
            model=self._model,
            messages=messages,
            api_key=self._api_key,
            stream=True,
            incremental_output=True,
            **kwargs,
        )

        full_content: str = ""
        usage_dict: dict[str, Any] = {}
        calls: list[ToolCall] = []

        async for chunk in responses:
            if chunk.status_code != 200:
                raise RuntimeError(f"DashScope error: {chunk.code} - {chunk.message}")

            if chunk.usage:
                usage = chunk.usage
                usage_dict = {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

            for choice in chunk.output.choices:
                msg = choice.message

                # Reasoning chunks
                # Use .get() because SDK's DictMixin.__getattr__ raises KeyError, not AttributeError
                # (Mark Sze) Have raised a PR to fix: https://github.com/dashscope/dashscope-sdk-python/pull/115
                if rc := msg.get("reasoning_content"):
                    await ctx.send(ModelReasoning(content=rc))

                # Content chunks
                if c := msg.get("content"):
                    full_content += c
                    await ctx.send(ModelMessageChunk(content=c))

                # Tool calls (typically in the final chunk)
                for tc in msg.get("tool_calls") or []:
                    args = tc["function"]["arguments"]
                    calls.append(
                        ToolCall(
                            id=tc["id"],
                            name=tc["function"]["name"],
                            arguments=args if isinstance(args, str) else json.dumps(args),
                        )
                    )

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
