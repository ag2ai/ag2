# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
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
)
from autogen.beta.tools.schemas import ToolSchema

from .mappers import convert_messages, tool_to_api

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
    ) -> ModelResponse:
        ollama_messages = convert_messages(context.prompt, messages)
        tools_list = [tool_to_api(t) for t in tools]

        kwargs: dict[str, Any] = {}
        if self._create_options:
            kwargs["options"] = self._create_options

        if tools_list:
            kwargs["tools"] = tools_list

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
            await context.send(ModelReasoning(content=msg.thinking))

        model_msg: ModelMessage | None = None
        if msg.content:
            model_msg = ModelMessage(content=msg.content)
            await context.send(model_msg)

        calls = [
            ToolCallEvent(
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
        response_stream = await self._client.chat(
            model=self._model,
            messages=messages,
            stream=True,
            **kwargs,
        )

        full_content: str = ""
        usage_dict: dict[str, Any] = {}
        calls: list[ToolCallEvent] = []

        async for chunk in response_stream:
            msg = chunk.message

            if msg.thinking:
                await context.send(ModelReasoning(content=msg.thinking))

            if msg.content:
                full_content += msg.content
                await context.send(ModelMessageChunk(content=msg.content))

            for i, tc in enumerate(msg.tool_calls or []):
                calls.append(
                    ToolCallEvent(
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
            await context.send(message)

        return ModelResponse(
            message=message,
            tool_calls=ToolCallsEvent(calls=calls),
            usage=usage_dict,
        )
