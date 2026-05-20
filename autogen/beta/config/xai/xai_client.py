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
from xai_sdk.proto import sample_pb2
from xai_sdk.types.chat import IncludeOption, ReasoningEffort, ToolMode

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
        # Defer AsyncClient construction to call time — it eagerly opens a gRPC
        # channel and validates XAI_API_KEY in its constructor, which would
        # otherwise turn ``XAIConfig.create()`` into a side-effecting call.
        self._api_key = api_key
        self._api_host = api_host
        self._timeout = timeout
        self._metadata = metadata
        self._channel_options = channel_options
        self._client: AsyncClient | None = None

        self._create_options: dict[str, Any] = {k: v for k, v in (create_options or {}).items() if v is not None}
        self._streaming = streaming

    def _get_client(self) -> AsyncClient:
        if self._client is None:
            self._client = AsyncClient(
                api_key=self._api_key,
                api_host=self._api_host,
                timeout=self._timeout,
                metadata=self._metadata,
                channel_options=self._channel_options,
            )
        return self._client

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

        chat = self._get_client().chat.create(**create_kwargs)
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

        if reasoning := getattr(response, "reasoning_content", None):
            await context.send(ModelReasoning(reasoning))

        model_msg: ModelMessage | None = None
        if content := getattr(response, "content", None):
            model_msg = ModelMessage(content)
            await context.send(model_msg)

        calls: list[ToolCallEvent] = []
        for tc in getattr(response, "tool_calls", None) or ():
            calls.append(
                ToolCallEvent(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments or "{}",
                )
            )

        await context.send(XAIAssistantEvent.from_response(response))

        return ModelResponse(
            message=model_msg,
            tool_calls=ToolCallsEvent(calls),
            usage=normalize_usage(getattr(response, "usage", None)),
            model=getattr(response, "model", None),
            provider=PROVIDER,
            finish_reason=_finish_reason_to_str(getattr(response, "finish_reason", None)),
        )

    async def _call_streaming(
        self,
        chat: XAIChat,
        context: "ConversationContext",
    ) -> ModelResponse:
        full_content: str = ""
        usage: Usage = Usage()
        finish_reason: str | None = None
        resolved_model: str | None = None
        last_response: XAIResponse | None = None
        # tool_calls accumulate by id; the SDK delivers whole calls per chunk.
        tool_calls_by_id: dict[str, ToolCallEvent] = {}

        async for response, chunk in chat.stream():
            last_response = response

            if reasoning := getattr(chunk, "reasoning_content", None):
                await context.send(ModelReasoning(reasoning))

            if content := getattr(chunk, "content", None):
                full_content += content
                await context.send(ModelMessageChunk(content))

            for tc in getattr(chunk, "tool_calls", None) or ():
                if tc.id and tc.id not in tool_calls_by_id:
                    tool_calls_by_id[tc.id] = ToolCallEvent(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments or "{}",
                    )

            if u := getattr(chunk, "usage", None):
                usage = normalize_usage(u)
            if model := getattr(chunk, "model", None):
                resolved_model = model
            if fr := getattr(chunk, "finish_reason", None):
                finish_reason = _finish_reason_to_str(fr)

        message: ModelMessage | None = None
        if full_content:
            message = ModelMessage(full_content)
            await context.send(message)

        if last_response is not None:
            await context.send(XAIAssistantEvent.from_response(last_response))
            if not usage:
                usage = normalize_usage(getattr(last_response, "usage", None))
            if not resolved_model:
                resolved_model = getattr(last_response, "model", None)
            if not finish_reason:
                finish_reason = _finish_reason_to_str(getattr(last_response, "finish_reason", None))

        return ModelResponse(
            message=message,
            tool_calls=ToolCallsEvent(list(tool_calls_by_id.values())),
            usage=usage,
            model=resolved_model,
            provider=PROVIDER,
            finish_reason=finish_reason,
        )


def _finish_reason_to_str(reason: "str | int | sample_pb2.FinishReason.ValueType | None") -> str | None:
    """Normalise xai-sdk finish_reason (string or proto enum) to a plain string.

    xai-sdk emits ``FinishReason`` proto-enum names like
    ``FINISH_REASON_STOP`` / ``FINISH_REASON_TOOL_CALLS``. Strip the prefix so
    downstream consumers see ``stop`` / ``tool_calls`` (consistent with openai).
    """
    if reason is None:
        return None
    if isinstance(reason, str):
        s = reason
    else:
        name = sample_pb2.FinishReason.Name(reason) if isinstance(reason, int) else str(reason)
        s = name
    if s.startswith("FINISH_REASON_"):
        s = s[len("FINISH_REASON_") :]
    return s.lower() or None
