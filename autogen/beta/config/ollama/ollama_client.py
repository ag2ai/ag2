# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any, TypedDict

from ollama import AsyncClient

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

        ollama_messages = convert_messages(prompt, messages)
        tools_list = [tool_to_api(t) for t in tools]

        kwargs: dict[str, Any] = {}
        if self._create_options:
            kwargs["options"] = self._create_options

        if tools_list:
            kwargs["tools"] = tools_list

        if fmt := response_proto_to_format(response_schema):
            kwargs["format"] = fmt

        if self._streaming:
            return await self._call_streaming(ollama_messages, kwargs, context)
        return await self._call_non_streaming(ollama_messages, kwargs, context)

    async def _call_non_streaming(
        self,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
        context: Context,
    ) -> ModelResponse:
        response = await self._client.chat(
            model=self._model,
            messages=messages,
            **kwargs,
        )

        msg = response.message

        if msg.thinking:
            await context.send(ModelReasoning(msg.thinking))

        model_msg: ModelMessage | None = None
        if msg.content:
            model_msg = ModelMessage(msg.content)
            await context.send(model_msg)

        calls = [
            ToolCallEvent(
                id=f"call_{i}",
                name=tc.function.name,
                arguments=json.dumps(tc.function.arguments),
            )
            for i, tc in enumerate(msg.tool_calls or [])
        ]

        prompt_n = float(response.prompt_eval_count or 0)
        completion_n = float(response.eval_count or 0)
        usage = Usage(
            prompt_tokens=prompt_n,
            completion_tokens=completion_n,
            total_tokens=prompt_n + completion_n,
        )

        return ModelResponse(
            message=model_msg,
            tool_calls=ToolCallsEvent(calls),
            usage=usage,
            model=response.model,
            provider="ollama",
            finish_reason=getattr(response, "done_reason", None),
        )

    async def _call_streaming(
        self,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
        context: Context,
    ) -> ModelResponse:
        response_stream = await self._client.chat(
            model=self._model,
            messages=messages,
            stream=True,
            **kwargs,
        )

        full_content: str = ""
        usage = Usage()
        calls: list[ToolCallEvent] = []
        finish_reason: str | None = None
        resolved_model: str | None = None

        async for chunk in response_stream:
            msg = chunk.message

            if msg.thinking:
                await context.send(ModelReasoning(msg.thinking))

            if msg.content:
                full_content += msg.content
                await context.send(ModelMessageChunk(msg.content))

            for i, tc in enumerate(msg.tool_calls or []):
                calls.append(
                    ToolCallEvent(
                        id=f"call_{len(calls) + i}",
                        name=tc.function.name,
                        arguments=json.dumps(tc.function.arguments),
                    )
                )

            if chunk.done:
                p_n = float(chunk.prompt_eval_count or 0)
                c_n = float(chunk.eval_count or 0)
                usage = Usage(prompt_tokens=p_n, completion_tokens=c_n, total_tokens=p_n + c_n)
                finish_reason = getattr(chunk, "done_reason", None)
                resolved_model = chunk.model

        message: ModelMessage | None = None
        if full_content:
            message = ModelMessage(full_content)
            await context.send(message)

        return ModelResponse(
            message=message,
            tool_calls=ToolCallsEvent(calls),
            usage=usage,
            model=resolved_model,
            provider="ollama",
            finish_reason=finish_reason,
        )
