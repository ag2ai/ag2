# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable
from typing import Any, TypedDict

from google import genai
from google.genai import types

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
        *messages: BaseEvent,
        ctx: Context,
        tools: Iterable[Tool],
    ) -> None:
        contents = self._convert_messages(messages)
        system_instruction = "\n\n".join(ctx.prompt) if ctx.prompt else None

        tool_declarations = [types.FunctionDeclaration(**self._tool_to_api(t)) for t in tools]
        gemini_tools = [types.Tool(function_declarations=tool_declarations)] if tool_declarations else None

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
            await self._process_stream(stream, ctx)
        else:
            response = await self._client.aio.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=config,
            )
            await self._process_response(response, ctx)

    @staticmethod
    def _tool_to_api(t: Tool) -> dict[str, Any]:
        return {
            "name": t.schema.function.name,
            "description": t.schema.function.description,
            "parameters": t.schema.function.parameters,
        }

    def _convert_messages(
        self,
        messages: tuple[BaseEvent, ...],
    ) -> list[types.Content]:
        result: list[types.Content] = []

        for message in messages:
            if isinstance(message, ModelRequest):
                result.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=message.content)],
                    )
                )
            elif isinstance(message, ModelResponse):
                parts: list[types.Part] = []
                if message.message:
                    parts.append(types.Part.from_text(text=message.message.content))
                for call in message.tool_calls.calls:
                    fc_part = types.Part.from_function_call(
                        name=call.name,
                        args=json.loads(call.arguments),
                    )
                    if "thought_signature" in call.provider_data:
                        fc_part.thought_signature = call.provider_data["thought_signature"]
                    parts.append(fc_part)
                if parts:
                    result.append(types.Content(role="model", parts=parts))
            elif isinstance(message, ToolResults):
                parts_list: list[types.Part] = []
                for r in message.results:
                    parts_list.append(
                        types.Part.from_function_response(
                            name=r.name if hasattr(r, "name") else "",
                            response={"result": r.content},
                        )
                    )
                result.append(types.Content(role="user", parts=parts_list))

        return result

    async def _process_response(
        self,
        response: types.GenerateContentResponse,
        ctx: Context,
    ) -> None:
        model_msg: ModelMessage | None = None
        calls: list[ToolCall] = []

        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.thought and part.text:
                    await ctx.send(ModelReasoning(content=part.text))
                elif part.text is not None:
                    model_msg = ModelMessage(content=part.text)
                    await ctx.send(model_msg)
                elif part.function_call:
                    fc = part.function_call
                    pdata: dict[str, Any] = {}
                    if part.thought_signature is not None:
                        pdata["thought_signature"] = part.thought_signature
                    calls.append(
                        ToolCall(
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
        usage: dict[str, Any] = {}

        async for chunk in stream:
            if chunk.candidates:
                for part in chunk.candidates[0].content.parts:
                    if part.thought and part.text:
                        await ctx.send(ModelReasoning(content=part.text))
                    elif part.text is not None:
                        full_content += part.text
                        await ctx.send(ModelMessageChunk(content=part.text))
                    elif part.function_call:
                        fc = part.function_call
                        pdata: dict[str, Any] = {}
                        if part.thought_signature is not None:
                            pdata["thought_signature"] = part.thought_signature
                        calls.append(
                            ToolCall(
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
            await ctx.send(message)

        await ctx.send(
            ModelResponse(
                message=message,
                tool_calls=ToolCalls(calls=calls),
                usage=usage,
            )
        )
