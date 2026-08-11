# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from typing import Any, Literal, TypedDict
from urllib.parse import urlparse

import httpx
from fast_depends.library.serializer import SerializerProto
from mistralai.client import Mistral
from mistralai.client.models import ChatCompletionResponse

from ag2.config.client import LLMClient
from ag2.context import ConversationContext
from ag2.events import (
    BaseEvent,
    BinaryResult,
    BinaryType,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    UrlInput,
    Usage,
)
from ag2.response import ResponseProto
from ag2.tools.schemas import ToolSchema

from .mappers import (
    PROVIDER,
    convert_messages,
    json_arguments,
    normalize_usage,
    response_proto_to_format,
    server_tool_name,
    server_tool_result,
    split_content,
    tool_to_api,
)

MISTRAL_DEFAULT_SERVER_URL = "https://api.mistral.ai"

IMAGE_FETCH_TIMEOUT_S = 30.0

_IMAGE_EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif"}
_EXTENSION_MEDIA_TYPES = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


def _image_media_type(url: str, content_type: str | None) -> str:
    """Blob storage serves these as octet-stream, so fall back to the URL suffix."""
    declared = (content_type or "").split(";")[0].strip()
    if declared.startswith("image/"):
        return declared
    suffix = urlparse(url).path.rsplit(".", 1)[-1].lower()
    return _EXTENSION_MEDIA_TYPES.get(suffix, "image/jpeg")


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]

__all__ = ["CreateOptions", "MistralClient", "ReasoningEffort"]


class CreateOptions(TypedDict, total=False):
    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    stop: str | list[str] | None
    random_seed: int | None
    presence_penalty: float | None
    frequency_penalty: float | None
    n: int | None
    parallel_tool_calls: bool | None
    tool_choice: str | dict[str, Any] | None
    reasoning_effort: ReasoningEffort | None
    prompt_mode: Literal["reasoning"] | None
    prompt_cache_key: str | None
    safe_prompt: bool | None
    metadata: dict[str, Any] | None
    timeout_ms: int | None
    http_headers: dict[str, str] | None


class MistralClient(LLMClient):
    """Mistral client adapter for the ``mistralai`` chat-completions API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        server_url: str | None = None,
        timeout_ms: int | None = None,
        async_client: httpx.AsyncClient | None = None,
        streaming: bool = False,
        create_options: CreateOptions | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._server_url = server_url
        self._timeout_ms = timeout_ms
        self._async_client = async_client
        self._streaming = streaming
        self._create_options: dict[str, Any] = {k: v for k, v in (create_options or {}).items() if v is not None}
        self._client: Mistral | None = None

    def _get_client(self) -> Mistral:
        if self._client is None:
            kwargs: dict[str, Any] = {}
            if self._api_key is not None:
                kwargs["api_key"] = self._api_key
            if self._server_url is not None:
                kwargs["server_url"] = self._server_url
            if self._timeout_ms is not None:
                kwargs["timeout_ms"] = self._timeout_ms
            if self._async_client is not None:
                kwargs["async_client"] = self._async_client
            self._client = Mistral(**kwargs)
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

        mistral_messages = convert_messages(prompt, messages, serializer)
        tools_list = [tool_to_api(t) for t in tools]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": mistral_messages,
            **self._create_options,
        }

        if tools_list:
            kwargs["tools"] = tools_list
        if response_format := response_proto_to_format(response_schema):
            kwargs["response_format"] = response_format

        client = self._get_client()

        if self._streaming:
            return await self._call_streaming(client, kwargs, context)
        return await self._call_non_streaming(client, kwargs, context)

    async def _fetch_image(self, url: str) -> BinaryResult | None:
        """Download a generated image so it can land on ``reply.files``.

        Mistral returns a short-lived signed URL where other providers inline the
        bytes; fetching keeps ``reply.files`` uniform. Failures are not fatal —
        the URL is still on the tool-result event.
        """
        try:
            client = self._async_client or httpx.AsyncClient()
            try:
                response = await client.get(url, timeout=IMAGE_FETCH_TIMEOUT_S)
                response.raise_for_status()
            finally:
                if self._async_client is None:
                    await client.aclose()
        except httpx.HTTPError:
            return None

        media_type = _image_media_type(url, response.headers.get("content-type"))
        return BinaryResult(
            response.content,
            metadata={
                "media_type": media_type,
                "url": url,
                "filename": f"generated.{_IMAGE_EXTENSIONS.get(media_type, 'jpg')}",
            },
        )

    async def _emit_tool_calls(
        self,
        raw_calls: Sequence[tuple[str, str, Any]],
        results: Mapping[str, Any],
        context: "ConversationContext",
    ) -> tuple[list[ToolCallEvent], list[BinaryResult]]:
        """Split calls into server-executed and client-side, and emit the former.

        A call whose id has a result in the same response was run by Mistral (the
        ``image_generation`` tool); anything else is ours to execute.
        """
        pending: list[ToolCallEvent] = []
        files: list[BinaryResult] = []

        for call_id, name, arguments in raw_calls:
            if call_id not in results:
                pending.append(ToolCallEvent(id=call_id, name=name, arguments=json_arguments(arguments)))
                continue

            tool_name = server_tool_name(name)
            result = server_tool_result(results[call_id])

            for part in result.parts:
                if not (isinstance(part, UrlInput) and part.kind is BinaryType.IMAGE):
                    continue
                if binary := await self._fetch_image(part.url):
                    files.append(binary)

            await context.send(BuiltinToolCallEvent(id=call_id, name=tool_name, arguments=json_arguments(arguments)))
            await context.send(BuiltinToolResultEvent(parent_id=call_id, name=tool_name, result=result))

        return pending, files

    async def _call_non_streaming(
        self,
        client: Mistral,
        kwargs: dict[str, Any],
        context: "ConversationContext",
    ) -> ModelResponse:
        response: ChatCompletionResponse = await client.chat.complete_async(**kwargs)

        choices = response.choices or []
        choice = choices[0] if choices else None

        # A server-side tool run replaces `message` with the whole exchange in
        # `messages`: the tool call, its result, then the final answer.
        turns = [choice.message] if choice and choice.message else list(choice.messages or []) if choice else []

        text = reasoning = ""
        raw_calls: list[tuple[str, str, Any]] = []
        results: dict[str, Any] = {}

        for turn in turns:
            if getattr(turn, "tool_call_id", None):
                results[turn.tool_call_id] = turn.content
                continue
            turn_text, turn_reasoning = split_content(turn.content)
            text += turn_text
            reasoning += turn_reasoning
            for tc in turn.tool_calls or []:
                if tc.id and tc.function is not None:
                    raw_calls.append((tc.id, tc.function.name, tc.function.arguments))

        if reasoning:
            await context.send(ModelReasoning(reasoning))

        model_msg: ModelMessage | None = None
        if text:
            model_msg = ModelMessage(text)
            await context.send(model_msg)

        calls, files = await self._emit_tool_calls(raw_calls, results, context)

        return ModelResponse(
            message=model_msg,
            tool_calls=ToolCallsEvent(calls),
            usage=normalize_usage(response.usage),
            model=response.model or self._model,
            provider=PROVIDER,
            finish_reason=choice.finish_reason if choice else None,
            files=files,
        )

    async def _call_streaming(
        self,
        client: Mistral,
        kwargs: dict[str, Any],
        context: "ConversationContext",
    ) -> ModelResponse:
        stream = await client.chat.stream_async(**kwargs)

        full_content = ""
        usage = Usage()
        finish_reason: str | None = None
        model: str | None = None
        # Keyed by index: Mistral sends whole calls today, fragments are allowed.
        tool_accs: dict[int, dict[str, str]] = {}
        results: dict[str, Any] = {}

        async for event in stream:
            chunk = event.data

            if chunk.model:
                model = chunk.model
            if chunk.usage:
                usage = normalize_usage(chunk.usage)

            for choice in chunk.choices or []:
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta
                if delta is None:
                    continue

                # A server-executed tool reports back mid-stream on its own delta.
                if tool_call_id := getattr(delta, "tool_call_id", None):
                    results[tool_call_id] = delta.content
                    continue

                text, reasoning = split_content(delta.content)
                if reasoning:
                    await context.send(ModelReasoning(reasoning))
                if text:
                    full_content += text
                    await context.send(ModelMessageChunk(text))

                for position, tc in enumerate(delta.tool_calls or []):
                    index = tc.index if isinstance(tc.index, int) else len(tool_accs) + position
                    acc = tool_accs.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        acc["id"] = tc.id
                    if tc.function is None:
                        continue
                    if tc.function.name:
                        acc["name"] = tc.function.name
                    if tc.function.arguments:
                        acc["arguments"] += json_arguments(tc.function.arguments)

        calls, files = await self._emit_tool_calls(
            [
                (acc["id"], acc["name"], acc["arguments"] or "{}")
                for _, acc in sorted(tool_accs.items())
                if acc["id"] and acc["name"]
            ],
            results,
            context,
        )

        message: ModelMessage | None = None
        if full_content:
            message = ModelMessage(full_content)
            await context.send(message)

        return ModelResponse(
            message=message,
            tool_calls=ToolCallsEvent(calls),
            usage=usage,
            model=model or self._model,
            provider=PROVIDER,
            finish_reason=finish_reason,
            files=files,
        )
