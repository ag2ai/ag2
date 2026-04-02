# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable, Sequence
from itertools import chain
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
    Usage,
)
from autogen.beta.response import ResponseProto
from autogen.beta.tools.schemas import ToolSchema

from .mappers import convert_messages, response_proto_to_format, tool_to_api

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
        response_schema: ResponseProto | None,
    ) -> ModelResponse:
        if response_schema and response_schema.system_prompt:
            prompt: Iterable[str] = chain(context.prompt, (response_schema.system_prompt,))
        else:
            prompt = context.prompt

        ds_messages = convert_messages(prompt, messages)
        tools_list = [tool_to_api(t) for t in tools]

        kwargs: dict[str, Any] = {
            **self._create_options,
            "result_format": "message",
        }

        if tools_list:
            kwargs["tools"] = tools_list

        if r := response_proto_to_format(response_schema):
            kwargs["response_format"] = r

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

        u = response.usage or {}
        usage = Usage(
            prompt_tokens=float(u.get("input_tokens", 0)),
            completion_tokens=float(u.get("output_tokens", 0)),
            total_tokens=float(u.get("total_tokens", 0)),
        )

        return ModelResponse(
            message=model_msg,
            tool_calls=ToolCallsEvent(calls=calls),
            usage=usage,
            model=self._model,
            provider="dashscope",
            finish_reason=choice.get("finish_reason")
            if hasattr(choice, "get")
            else getattr(choice, "finish_reason", None),
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
        usage = Usage()
        calls: list[ToolCallEvent] = []
        finish_reason: str | None = None

        async for chunk in responses:
            if chunk.status_code != 200:
                raise RuntimeError(f"DashScope error: {chunk.code} - {chunk.message}")

            if chunk.usage:
                u = chunk.usage
                usage = Usage(
                    prompt_tokens=float(u.get("input_tokens", 0)),
                    completion_tokens=float(u.get("output_tokens", 0)),
                    total_tokens=float(u.get("total_tokens", 0)),
                )

            for choice in chunk.output.choices:
                fr = choice.get("finish_reason") if hasattr(choice, "get") else getattr(choice, "finish_reason", None)
                if fr:
                    finish_reason = fr

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
            usage=usage,
            model=self._model,
            provider="dashscope",
            finish_reason=finish_reason,
        )
