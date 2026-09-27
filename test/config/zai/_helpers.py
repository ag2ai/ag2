# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock

import httpx
from typing_extensions import Unpack
from zai.types.chat.chat_completion import (
    Completion,
    CompletionChoice,
    CompletionMessage,
    CompletionMessageToolCall,
    CompletionTokensDetails,
    CompletionUsage,
    Function,
    PromptTokensDetails,
)
from zai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from zai.types.chat.chat_completion_chunk import CompletionUsage as ChunkUsage

from ag2.config.zai import ZAIConfig
from ag2.config.zai.config import ZAIConfigOverrides
from test.config._helpers import WireRecorder


def make_usage(
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> CompletionUsage:
    # The SDK's own `construct`, the unvalidated path every response it parses takes, so a
    # `null` the API sends for a count arrives as `None` here too.
    return CompletionUsage.construct(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens) if cached_tokens is not None else None,
        completion_tokens_details=(
            CompletionTokensDetails(reasoning_tokens=reasoning_tokens) if reasoning_tokens is not None else None
        ),
    )


def completion_json(
    content: str | None = "ok",
    reasoning_content: str | None = None,
    tool_calls: list[CompletionMessageToolCall] | None = None,
    finish_reason: str = "stop",
    usage: CompletionUsage | None = None,
    model: str = "glm-test",
) -> dict[str, Any]:
    """A `/chat/completions` reply body, built from the SDK's own `Completion`."""
    completion = Completion(
        id="cmpl-1",
        created=1,
        model=model,
        choices=[
            CompletionChoice(
                index=0,
                finish_reason=finish_reason,
                message=CompletionMessage(
                    role="assistant", content=content, reasoning_content=reasoning_content, tool_calls=tool_calls
                ),
            )
        ],
        usage=usage or CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    return completion.model_dump(exclude_none=True)


def chunk_json(
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list[ChoiceDeltaToolCall] | None = None,
    finish_reason: str | None = None,
    usage: ChunkUsage | None = None,
) -> dict[str, Any]:
    """One streamed `chat.completion.chunk` event, built from the SDK's own `ChatCompletionChunk`."""
    chunk = ChatCompletionChunk(
        id="cmpl-1",
        created=1,
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(content=content, reasoning_content=reasoning_content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
        extra_json={},
    )
    # `extra_json` is the SDK's slot for unknown keys, not a wire field.
    return chunk.model_dump(exclude_none=True, exclude={"extra_json"})


def tool_call_json(
    call_id: str = "tc_1", name: str = "search_docs", arguments: str = '{"query": "x"}'
) -> CompletionMessageToolCall:
    return CompletionMessageToolCall(id=call_id, type="function", function=Function(name=name, arguments=arguments))


def with_object_arguments(body: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """`body` with its first tool call's `arguments` as a JSON object, as Z.AI has sent it.

    The SDK types the field `str` but parses replies unvalidated, so the object arrives as-is.
    """
    body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = arguments
    return body


def tool_call_delta_json(
    index: int, call_id: str | None = None, name: str | None = None, arguments: str | None = None
) -> ChoiceDeltaToolCall:
    return ChoiceDeltaToolCall(
        index=index,
        id=call_id,
        type="function" if call_id is not None else None,
        function=ChoiceDeltaToolCallFunction(name=name, arguments=arguments),
    )


@contextmanager
def wire_config(recorder: WireRecorder, /, **overrides: Unpack[ZAIConfigOverrides]) -> Generator[ZAIConfig]:
    """A `ZAIConfig` whose SDK client talks to `recorder` through the public `http_client` seam."""
    with httpx.Client(transport=httpx.MockTransport(recorder)) as http_client:
        yield ZAIConfig(model="glm-test", api_key="id.secret", http_client=http_client).copy(**overrides)


def make_call_context(prompt: list[str] | None = None) -> AsyncMock:
    ctx = AsyncMock()
    ctx.send = AsyncMock()
    ctx.prompt = prompt or []
    return ctx
