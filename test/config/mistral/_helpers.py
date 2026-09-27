# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import httpx
from mistralai.client.models import (
    ArgumentsTypedDict,
    ChatCompletionChoiceFinishReason,
    ChatCompletionResponseTypedDict,
    CompletionChunkTypedDict,
    CompletionResponseStreamChoiceFinishReason,
    DeltaMessageContentTypedDict,
    DeltaMessageTypedDict,
    ToolCallTypedDict,
    UsageInfoTypedDict,
)

from test.config._helpers import WireRecorder

CHAT_URL = "https://api.mistral.ai/v1/chat/completions"


def make_usage(prompt_tokens: int = 1, completion_tokens: int = 1, total_tokens: int = 2) -> UsageInfoTypedDict:
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens}


def make_tool_call(
    call_id: str = "tc_1",
    name: str = "search_docs",
    arguments: ArgumentsTypedDict = '{"query": "x"}',
    index: int | None = None,
) -> ToolCallTypedDict:
    call: ToolCallTypedDict = {"id": call_id, "function": {"name": name, "arguments": arguments}}
    if index is not None:
        call["index"] = index
    return call


def make_turn(
    content: DeltaMessageContentTypedDict = "",
    *,
    tool_call_id: str | None = None,
    tool_calls: list[tuple[str, str, str]] | None = None,
) -> DeltaMessageTypedDict:
    """One turn of `ChatCompletionChoice.messages`."""
    turn: DeltaMessageTypedDict = {"role": "assistant", "content": content}
    if tool_call_id is not None:
        turn["tool_call_id"] = tool_call_id
    if tool_calls:
        turn["tool_calls"] = [make_tool_call(i, n, a) for i, n, a in tool_calls]
    return turn


def make_server_tool_turns(
    call_id: str = "gen_1",
    name: str = "generate_image",
    arguments: str = '{"prompt": "a red circle"}',
    url: str = "https://example.com/generated.jpg",
    text: str = "Here is your image.",
) -> list[DeltaMessageTypedDict]:
    """The `messages` trace a server-executed tool produces: call, result, answer."""
    return [
        make_turn(tool_calls=[(call_id, name, arguments)]),
        make_turn(f'{{"url": "{url}"}}', tool_call_id=call_id),
        make_turn(text),
    ]


def make_agentic_response(
    turns: list[DeltaMessageTypedDict] | None = None,
    finish_reason: ChatCompletionChoiceFinishReason = "stop",
    usage: UsageInfoTypedDict | None = None,
    model: str = "mistral-test",
) -> ChatCompletionResponseTypedDict:
    """A completion with no `message`, whose exchange is in `messages`."""
    return {
        "id": "cmpl_1",
        "object": "chat.completion",
        "model": model,
        "created": 0,
        "usage": usage if usage is not None else make_usage(),
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "messages": turns if turns is not None else make_server_tool_turns(),
            }
        ],
    }


def make_response(
    content: DeltaMessageContentTypedDict | None = "ok",
    tool_calls: list[ToolCallTypedDict] | None = None,
    finish_reason: ChatCompletionChoiceFinishReason = "stop",
    usage: UsageInfoTypedDict | None = None,
    model: str = "mistral-test",
) -> ChatCompletionResponseTypedDict:
    return {
        "id": "cmpl_1",
        "object": "chat.completion",
        "model": model,
        "created": 0,
        "usage": usage if usage is not None else make_usage(),
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, "tool_calls": tool_calls or []},
            }
        ],
    }


def make_stream_chunk(
    content: DeltaMessageContentTypedDict | None = None,
    tool_calls: list[ToolCallTypedDict] | None = None,
    tool_call_id: str | None = None,
    finish_reason: CompletionResponseStreamChoiceFinishReason | None = None,
    usage: UsageInfoTypedDict | None = None,
    model: str = "mistral-test",
) -> CompletionChunkTypedDict:
    """One SSE `data:` payload; the SDK wraps it in a `CompletionEvent` itself."""
    delta: DeltaMessageTypedDict = {"content": content, "tool_calls": tool_calls or []}
    if tool_call_id is not None:
        delta["tool_call_id"] = tool_call_id
    chunk: CompletionChunkTypedDict = {
        "id": "cmpl_1",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def image_response(data: bytes = b"\xff\xd8image", content_type: str = "image/jpeg") -> httpx.Response:
    """What the blob store answers a generated-image fetch with."""
    return httpx.Response(200, content=data, headers={"content-type": content_type})


class FailingFetches:
    """Answers the chat API from `wire` and fails every other request with `error`."""

    def __init__(self, wire: WireRecorder, error: Exception) -> None:
        self.wire = wire
        self.error = error

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.mistral.ai":
            return self.wire(request)
        raise self.error


def make_call_context(prompt: list[str] | None = None) -> AsyncMock:
    ctx = AsyncMock()
    ctx.send = AsyncMock()
    ctx.prompt = prompt or []
    return ctx
