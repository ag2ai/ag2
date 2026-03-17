# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any, TypedDict

import dashscope
from dashscope.aigc.generation import AioGeneration

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
from autogen.beta.tools.schemas import ToolSchema

from .mappers import convert_messages, tool_to_api

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
        messages: Sequence[BaseEvent],
        context: Context,
        *,
        tools: Iterable[ToolSchema],
    ) -> ModelResponse:
        ds_messages = convert_messages(context.prompt, messages)
        tools_list = [tool_to_api(t) for t in tools]

        kwargs: dict[str, Any] = {
            **self._create_options,
            "result_format": "message",
        }

        if tools_list:
            kwargs["tools"] = tools_list

        # Set the base URL for this call (SDK uses a global)
        dashscope.base_http_api_url = self._base_url

        if self._streaming:
            return await self._call_streaming(ds_messages, kwargs, context)
        return await self._call_non_streaming(ds_messages, kwargs, context)

    async def _call_non_streaming(
        self,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
        context: Context,
    ) -> ModelResponse:
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

        # Use .get() because SDK's DictMixin.__getattr__ raises KeyError, not AttributeError
        # (Mark Sze) Have raised a PR to fix: https://github.com/dashscope/dashscope-sdk-python/pull/115
        if reasoning := msg.get("reasoning_content"):
            await context.send(ModelReasoning(content=reasoning))

        model_msg: ModelMessage | None = None
        if content := msg.get("content"):
            model_msg = ModelMessage(content=content)
            await context.send(model_msg)

        calls = []
        for tc in msg.get("tool_calls") or []:
            args = tc["function"]["arguments"]
            calls.append(
                ToolCallEvent(
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

        return ModelResponse(
            message=model_msg,
            tool_calls=ToolCallsEvent(calls=calls),
            usage=usage_dict,
        )

    async def _call_streaming(
        self,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
        context: Context,
    ) -> ModelResponse:
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
        calls: list[ToolCallEvent] = []

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

                # Use .get() because SDK's DictMixin.__getattr__ raises KeyError, not AttributeError
                # (Mark Sze) Have raised a PR to fix: https://github.com/dashscope/dashscope-sdk-python/pull/115
                if rc := msg.get("reasoning_content"):
                    await context.send(ModelReasoning(content=rc))

                if c := msg.get("content"):
                    full_content += c
                    await context.send(ModelMessageChunk(content=c))

                for tc in msg.get("tool_calls") or []:
                    args = tc["function"]["arguments"]
                    calls.append(
                        ToolCallEvent(
                            id=tc["id"],
                            name=tc["function"]["name"],
                            arguments=args if isinstance(args, str) else json.dumps(args),
                        )
                    )

        message: ModelMessage | None = None
        if full_content:
            message = ModelMessage(content=full_content)
            await context.send(message)

        return ModelResponse(
            message=message,
            tool_calls=ToolCallsEvent(calls=calls),
            usage=usage_dict,
        )
