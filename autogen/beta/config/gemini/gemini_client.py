# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any, TypedDict

from google import genai
from google.genai import types

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

from .mappers import build_system_instruction, build_tools, convert_messages, normalize_usage, response_proto_to_config


class CreateConfig(TypedDict, total=False):
    temperature: float | None
    top_p: float | None
    top_k: int | None
    max_output_tokens: int | None
    stop_sequences: list[str] | None
    presence_penalty: float | None
    frequency_penalty: float | None
    seed: int | None


class GeminiClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        streaming: bool = False,
        create_config: CreateConfig | None = None,
        cached_content: str | None = None,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model_name = model
        self._streaming = streaming
        self._create_config = create_config or {}
        self._cached_content = cached_content

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        *,
        tools: Iterable[ToolSchema],
        response_schema: ResponseProto | None,
    ) -> ModelResponse:
        contents = convert_messages(messages)

        if response_schema and response_schema.system_prompt:
            prompt: Iterable[str] = chain(context.prompt, (response_schema.system_prompt,))
        else:
            prompt = context.prompt

        system_instruction = build_system_instruction(prompt)
        gemini_tools = build_tools(list(tools))

        cache_kwargs: dict[str, Any] = {}
        if self._cached_content:
            cache_kwargs["cached_content"] = self._cached_content

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=gemini_tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True) if gemini_tools else None,
            **response_proto_to_config(response_schema),
            **self._create_config,
            **cache_kwargs,
        )

        if self._streaming:
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model_name,
                contents=contents,
                config=config,
            )
            return await self._process_stream(stream, context)

        response = await self._client.aio.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=config,
        )
        return await self._process_response(response, context)

    async def _process_response(
        self,
        response: types.GenerateContentResponse,
        context: Context,
    ) -> ModelResponse:
        model_msg: ModelMessage | None = None
        calls: list[ToolCallEvent] = []

        for candidate in response.candidates or ():
            if candidate.content:
                for part in candidate.content.parts or ():
                    if part.thought and part.text:
                        await context.send(ModelReasoning(part.text))
                    elif part.text is not None:
                        model_msg = ModelMessage(part.text)
                        await context.send(model_msg)
                    elif part.function_call:
                        fc = part.function_call
                        pdata: dict[str, Any] = {}
                        if part.thought_signature is not None:
                            pdata["thought_signature"] = part.thought_signature
                        calls.append(
                            ToolCallEvent(
                                id=fc.id or fc.name or "",
                                name=fc.name or "",
                                arguments=json.dumps(dict(fc.args)) if fc.args else "{}",
                                provider_data=pdata,
                            )
                        )

        usage = Usage()
        if response.usage_metadata:
            usage = normalize_usage(response.usage_metadata)

        finish_reason = None
        if response.candidates:
            fr = response.candidates[0].finish_reason
            if fr is not None:
                finish_reason = fr.name.lower() if hasattr(fr, "name") else str(fr)

        return ModelResponse(
            message=model_msg,
            tool_calls=ToolCallsEvent(calls),
            usage=usage,
            model=self._model_name,
            provider="google",
            finish_reason=finish_reason,
        )

    async def _process_stream(
        self,
        stream: Any,
        context: Context,
    ) -> ModelResponse:
        full_content: str = ""
        calls: list[ToolCallEvent] = []
        usage = Usage()
        finish_reason: str | None = None

        async for chunk in stream:
            for candidate in chunk.candidates or ():
                if candidate.content:
                    for part in candidate.content.parts or ():
                        if part.thought and part.text:
                            await context.send(ModelReasoning(part.text))
                        elif part.text is not None:
                            full_content += part.text
                            await context.send(ModelMessageChunk(part.text))
                        elif part.function_call:
                            fc = part.function_call
                            pdata: dict[str, Any] = {}
                            if part.thought_signature is not None:
                                pdata["thought_signature"] = part.thought_signature
                            calls.append(
                                ToolCallEvent(
                                    id=fc.id or fc.name or "",
                                    name=fc.name or "",
                                    arguments=json.dumps(dict(fc.args)) if fc.args else "{}",
                                    provider_data=pdata,
                                )
                            )

            if chunk.usage_metadata:
                usage = normalize_usage(chunk.usage_metadata)

            if chunk.candidates:
                fr = chunk.candidates[0].finish_reason
                if fr is not None:
                    finish_reason = fr.name.lower() if hasattr(fr, "name") else str(fr)

        message: ModelMessage | None = None
        if full_content:
            message = ModelMessage(full_content)
            await context.send(message)

        return ModelResponse(
            message=message,
            tool_calls=ToolCallsEvent(calls),
            usage=usage,
            model=self._model_name,
            provider="google",
            finish_reason=finish_reason,
        )
