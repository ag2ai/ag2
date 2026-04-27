# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any, TypedDict
from uuid import uuid4

import google.auth
import google.genai as genai
from fast_depends.library.serializer import SerializerProto
from google.genai import types
from google.oauth2 import service_account

from autogen.beta.config.client import LLMClient
from autogen.beta.context import ConversationContext
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

from .events import GeminiServerToolCallEvent, GeminiServerToolResultEvent
from .mappers import (
    build_system_instruction,
    build_tools,
    convert_messages,
    grounding_tool_name,
    normalize_usage,
    response_proto_to_config,
)


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
        vertexai: bool | None = None,
        credentials: google.auth.credentials.Credentials | str | None = None,
        project: str | None = None,
        location: str | None = None,
        streaming: bool = False,
        create_config: CreateConfig | None = None,
        cached_content: str | None = None,
    ) -> None:
        if isinstance(credentials, str):
            # String indicates a json credentials file, load into credentials
            credentials = service_account.Credentials.from_service_account_file(
                credentials,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        self._client = genai.Client(
            vertexai=vertexai, api_key=api_key, credentials=credentials, project=project, location=location
        )
        self._model_name = model
        self._streaming = streaming
        self._create_config = create_config or {}
        self._cached_content = cached_content

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: "ConversationContext",
        *,
        tools: Iterable[ToolSchema],
        response_schema: ResponseProto | None,
        serializer: SerializerProto,
    ) -> ModelResponse:
        contents = convert_messages(messages, serializer)

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
        context: "ConversationContext",
    ) -> ModelResponse:
        model_msg: ModelMessage | None = None
        calls: list[ToolCallEvent] = []

        for candidate in response.candidates or ():
            pending_code_call_id: str | None = None
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
                                id=fc.id or str(uuid4()),
                                name=fc.name or "",
                                arguments=json.dumps(dict(fc.args)) if fc.args else "{}",
                                provider_data=pdata,
                            )
                        )
                    elif part.executable_code and (call_event := GeminiServerToolCallEvent.from_executable_code(part)):
                        pending_code_call_id = call_event.id
                        await context.send(call_event)
                    elif (
                        part.code_execution_result
                        and pending_code_call_id is not None
                        and (
                            result_event := GeminiServerToolResultEvent.from_code_execution_result(
                                part, parent_id=pending_code_call_id
                            )
                        )
                    ):
                        await context.send(result_event)
                        pending_code_call_id = None
            grounding = candidate.grounding_metadata if candidate.grounding_metadata else None
            if grounding:
                name = grounding_tool_name(grounding)
                gnd_call = GeminiServerToolCallEvent.from_grounding(grounding, name=name)
                await context.send(gnd_call)
                await context.send(
                    GeminiServerToolResultEvent.from_grounding(grounding, parent_id=gnd_call.id, name=name)
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
        context: "ConversationContext",
    ) -> ModelResponse:
        full_content: str = ""
        calls: list[ToolCallEvent] = []
        usage = Usage()
        finish_reason: str | None = None
        pending_code_call_id: str | None = None
        last_grounding_metadata: types.GroundingMetadata | None = None

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
                                    id=fc.id or str(uuid4()),
                                    name=fc.name or "",
                                    arguments=json.dumps(dict(fc.args)) if fc.args else "{}",
                                    provider_data=pdata,
                                )
                            )
                        elif part.executable_code and (
                            call_event := GeminiServerToolCallEvent.from_executable_code(part)
                        ):
                            pending_code_call_id = call_event.id
                            await context.send(call_event)
                        elif (
                            part.code_execution_result
                            and pending_code_call_id is not None
                            and (
                                result_event := GeminiServerToolResultEvent.from_code_execution_result(
                                    part, parent_id=pending_code_call_id
                                )
                            )
                        ):
                            await context.send(result_event)
                            pending_code_call_id = None
                grounding = candidate.grounding_metadata if candidate.grounding_metadata else None
                if grounding:
                    last_grounding_metadata = grounding

            if chunk.usage_metadata:
                usage = normalize_usage(chunk.usage_metadata)

            if chunk.candidates:
                fr = chunk.candidates[0].finish_reason
                if fr is not None:
                    finish_reason = fr.name.lower() if hasattr(fr, "name") else str(fr)

        if last_grounding_metadata is not None:
            name = grounding_tool_name(last_grounding_metadata)
            gnd_call = GeminiServerToolCallEvent.from_grounding(last_grounding_metadata, name=name)
            await context.send(gnd_call)
            await context.send(
                GeminiServerToolResultEvent.from_grounding(last_grounding_metadata, parent_id=gnd_call.id, name=name)
            )

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
