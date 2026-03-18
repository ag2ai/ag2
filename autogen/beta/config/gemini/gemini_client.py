# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
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
)
from autogen.beta.tools.schemas import ToolSchema

from .mappers import build_tools, convert_messages


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
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model_name = model
        self._streaming = streaming
        self._create_config = create_config or {}

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        *,
        tools: Iterable[ToolSchema],
    ) -> ModelResponse:
        contents = convert_messages(messages)
        system_instruction = "\n\n".join(context.prompt) if context.prompt else None

        gemini_tools = build_tools(list(tools))

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=gemini_tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True) if gemini_tools else None,
            **self._create_config,
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
                        await context.send(ModelReasoning(content=part.text))
                    elif part.text is not None:
                        model_msg = ModelMessage(content=part.text)
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

        usage = {}
        if response.usage_metadata:
            usage = {
                "prompt_token_count": response.usage_metadata.prompt_token_count,
                "candidates_token_count": response.usage_metadata.candidates_token_count,
                "total_token_count": response.usage_metadata.total_token_count,
            }

        return ModelResponse(
            message=model_msg,
            tool_calls=ToolCallsEvent(calls=calls),
            usage=usage,
        )

    async def _process_stream(
        self,
        stream: Any,
        context: Context,
    ) -> ModelResponse:
        full_content: str = ""
        calls: list[ToolCallEvent] = []
        usage: dict[str, Any] = {}

        async for chunk in stream:
            for candidate in chunk.candidates or ():
                if candidate.content:
                    for part in candidate.content.parts or ():
                        if part.thought and part.text:
                            await context.send(ModelReasoning(content=part.text))
                        elif part.text is not None:
                            full_content += part.text
                            await context.send(ModelMessageChunk(content=part.text))
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
                usage = {
                    "prompt_token_count": chunk.usage_metadata.prompt_token_count,
                    "candidates_token_count": chunk.usage_metadata.candidates_token_count,
                    "total_token_count": chunk.usage_metadata.total_token_count,
                }

        message: ModelMessage | None = None
        if full_content:
            message = ModelMessage(content=full_content)
            await context.send(message)

        return ModelResponse(
            message=message,
            tool_calls=ToolCallsEvent(calls=calls),
            usage=usage,
        )
