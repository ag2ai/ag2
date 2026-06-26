# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any, TypedDict

from fast_depends.library.serializer import SerializerProto
from typing_extensions import Required
from xai_sdk import AsyncClient
from xai_sdk.aio.chat import Chat as XAIChat
from xai_sdk.chat import Response as XAIResponse
from xai_sdk.proto import chat_pb2, sample_pb2
from xai_sdk.types.chat import IncludeOption, ReasoningEffort, ToolMode

from ag2.config.client import LLMClient
from ag2.context import ConversationContext
from ag2.events import (
    BaseEvent,
    BuiltinToolCallEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    Usage,
)
from ag2.response import ResponseProto
from ag2.tools.schemas import ToolSchema

from .events import XAIAssistantEvent
from .mappers import (
    PROVIDER,
    convert_messages,
    normalize_usage,
    response_proto_to_format,
    tool_to_api,
)

__all__ = ["CreateOptions", "IncludeOption", "ReasoningEffort", "XAIClient"]


class CreateOptions(TypedDict, total=False):
    model: Required[str]

    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    frequency_penalty: float | None
    presence_penalty: float | None
    seed: int | None
    stop: Sequence[str] | None
    user: str | None
    logprobs: bool | None
    top_logprobs: int | None
    tool_choice: ToolMode | None
    parallel_tool_calls: bool | None
    reasoning_effort: ReasoningEffort | None
    store_messages: bool | None
    previous_response_id: str | None
    use_encrypted_content: bool | None
    max_turns: int | None
    include: Sequence[IncludeOption] | None
    conversation_id: str | None


class XAIClient(LLMClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_host: str = "api.x.ai",
        timeout: float | None = None,
        metadata: tuple[tuple[str, str], ...] | None = None,
        channel_options: list[tuple[str, Any]] | None = None,
        streaming: bool = False,
        create_options: CreateOptions | None = None,
    ) -> None:
        self._client = AsyncClient(
            api_key=api_key,
            api_host=api_host,
            timeout=timeout,
            metadata=metadata,
            channel_options=channel_options,
        )
        self._create_options: dict[str, Any] = {k: v for k, v in (create_options or {}).items() if v is not None}
        self._streaming = streaming

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: "ConversationContext",
        *,
        tools: Iterable[ToolSchema],
        response_schema: ResponseProto | None,
        serializer: SerializerProto,
    ) -> ModelResponse:
        if response_schema and response_schema.system_prompt:
            prompt: Iterable[str] = chain(context.prompt, (response_schema.system_prompt,))
        else:
            prompt = context.prompt

        xai_messages, replay_responses = convert_messages(prompt, messages, serializer)
        xai_tools = [tool_to_api(t) for t in tools]
        response_format = response_proto_to_format(response_schema)

        create_kwargs: dict[str, Any] = dict(self._create_options)
        if xai_messages:
            create_kwargs["messages"] = xai_messages
        if xai_tools:
            create_kwargs["tools"] = xai_tools
        if response_format is not None:
            create_kwargs["response_format"] = response_format

        chat = self._client.chat.create(**create_kwargs)
        for resp in replay_responses:
            chat.append(resp)

        if self._streaming:
            return await self._call_streaming(chat, context)
        return await self._call_non_streaming(chat, context)

    async def _call_non_streaming(
        self,
        chat: XAIChat,
        context: "ConversationContext",
    ) -> ModelResponse:
        response = await chat.sample()

        if response.reasoning_content:
            await context.send(ModelReasoning(response.reasoning_content))

        model_msg: ModelMessage | None = None
        if response.content:
            model_msg = ModelMessage(response.content)
            await context.send(model_msg)

        calls: list[ToolCallEvent] = []
        for tc in response.tool_calls:
            cls = ToolCallEvent if tc.type == chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL else BuiltinToolCallEvent
            ev = cls(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            if isinstance(ev, BuiltinToolCallEvent):
                await context.send(ev)
            else:
                calls.append(ev)

        await context.send(XAIAssistantEvent.from_response(response))

        # xai-sdk returns finish_reason as either a string (e.g. "FINISH_REASON_STOP")
        # or a proto enum int — strip the prefix and lowercase to match openai's "stop".
        fr = response.finish_reason
        finish_reason: str | None = None
        if fr:
            name = sample_pb2.FinishReason.Name(fr) if isinstance(fr, int) else str(fr)
            finish_reason = name.removeprefix("FINISH_REASON_").removeprefix("REASON_").lower() or None

        return ModelResponse(
            message=model_msg,
            tool_calls=ToolCallsEvent(calls),
            usage=normalize_usage(response.usage),
            model=response.proto.model or None,
            provider=PROVIDER,
            finish_reason=finish_reason,
        )

    async def _call_streaming(
        self,
        chat: XAIChat,
        context: "ConversationContext",
    ) -> ModelResponse:
        full_content: str = ""
        usage: Usage = Usage()
        finish_reason_raw: str | int | sample_pb2.FinishReason.ValueType | None = None
        resolved_model: str | None = None
        last_response: XAIResponse | None = None
        # tool_calls accumulate by id; the SDK delivers whole calls per chunk.
        tool_calls_by_id: dict[str, ToolCallEvent] = {}
        builtin_emitted: set[str] = set()

        async for response, chunk in chat.stream():
            last_response = response

            if chunk.reasoning_content:
                await context.send(ModelReasoning(chunk.reasoning_content))

            if chunk.content:
                full_content += chunk.content
                await context.send(ModelMessageChunk(chunk.content))

            for tc in chunk.tool_calls:
                if not tc.id:
                    continue
                if tc.id in tool_calls_by_id or tc.id in builtin_emitted:
                    continue
                cls = ToolCallEvent if tc.type == chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL else BuiltinToolCallEvent
                ev = cls(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
                if isinstance(ev, BuiltinToolCallEvent):
                    await context.send(ev)
                    builtin_emitted.add(tc.id)
                else:
                    tool_calls_by_id[tc.id] = ev

            if response.usage:
                usage = normalize_usage(response.usage)
            if model := response.proto.model or None:
                resolved_model = model
            if fr := response.finish_reason:
                finish_reason_raw = fr

        message: ModelMessage | None = None
        if full_content:
            message = ModelMessage(full_content)
            await context.send(message)

        if last_response is not None:
            await context.send(XAIAssistantEvent.from_response(last_response))
            if not usage:
                usage = normalize_usage(last_response.usage)
            if not resolved_model:
                resolved_model = last_response.proto.model or None
            if finish_reason_raw is None:
                finish_reason_raw = last_response.finish_reason

        # xai-sdk returns finish_reason as either a string (e.g. "FINISH_REASON_STOP")
        # or a proto enum int — strip the prefix and lowercase to match openai's "stop".
        finish_reason: str | None = None
        if finish_reason_raw is not None:
            name = (
                sample_pb2.FinishReason.Name(finish_reason_raw)
                if isinstance(finish_reason_raw, int)
                else str(finish_reason_raw)
            )
            finish_reason = name.removeprefix("FINISH_REASON_").removeprefix("REASON_").lower() or None

        return ModelResponse(
            message=message,
            tool_calls=ToolCallsEvent(list(tool_calls_by_id.values())),
            usage=usage,
            model=resolved_model,
            provider=PROVIDER,
            finish_reason=finish_reason,
        )
