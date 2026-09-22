# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from zai.core import StreamResponse


class FakeStreamResponse(StreamResponse[Any]):
    """A `StreamResponse` over a fixed list of chunks.

    `ZAIClient` narrows on the SDK's own class to tell a stream from a completion — the SDK
    answers both from one `create` call — so the double has to be one. Iteration is overridden
    rather than fed through the SSE machinery, so nothing here depends on the SDK's internals.
    """

    def __init__(self, chunks: Iterable[Any]) -> None:
        self._chunks: Iterator[Any] = iter(chunks)

    def __iter__(self) -> Iterator[Any]:
        return self._chunks

    def __next__(self) -> Any:
        return next(self._chunks)


def make_usage(
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens) if cached_tokens is not None else None,
        completion_tokens_details=(
            SimpleNamespace(reasoning_tokens=reasoning_tokens) if reasoning_tokens is not None else None
        ),
    )


def make_response(
    content: str | None = "ok",
    reasoning_content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    usage: Any | None = None,
    model: str = "glm-test",
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls or [],
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage if usage is not None else make_usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model=model,
    )


def make_tool_call(
    call_id: str = "tc_1",
    name: str = "search_docs",
    arguments: Any = '{"query": "x"}',
) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


def make_stream_chunk(
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
    usage: Any | None = None,
    model: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls or [],
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
        model=model,
    )


def make_stream_tool_call(
    index: int, call_id: str | None = None, name: str | None = None, arguments: Any = None
) -> SimpleNamespace:
    return SimpleNamespace(index=index, id=call_id, function=SimpleNamespace(name=name, arguments=arguments))


class FakeCompletions:
    def __init__(self, response: Any | None = None, stream_chunks: Iterable[Any] = ()) -> None:
        self.response = response if response is not None else make_response()
        self.stream_chunks = list(stream_chunks)
        self.kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if kwargs.get("stream"):
            return FakeStreamResponse(self.stream_chunks)
        return self.response


class FakeZAIClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def make_call_context(prompt: list[str] | None = None) -> AsyncMock:
    ctx = AsyncMock()
    ctx.send = AsyncMock()
    ctx.prompt = prompt or []
    return ctx
