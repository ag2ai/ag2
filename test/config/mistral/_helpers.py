# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock


def make_usage(
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cached_tokens: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_tokens_details={"cached_tokens": cached_tokens} if cached_tokens is not None else None,
    )


def make_tool_call(
    call_id: str = "tc_1",
    name: str = "search_docs",
    arguments: Any = '{"query": "x"}',
    index: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        index=index,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def make_server_tool_turns(
    call_id: str = "gen_1",
    name: str = "generate_image",
    arguments: str = '{"prompt": "a red circle"}',
    url: str = "https://example.com/generated.jpg",
    text: str = "Here is your image.",
) -> list[SimpleNamespace]:
    """The `messages` trace a server-executed tool produces: call, result, answer."""
    return [
        SimpleNamespace(content="", tool_call_id=None, tool_calls=[make_tool_call(call_id, name, arguments)]),
        SimpleNamespace(content=f'{{"url": "{url}"}}', tool_call_id=call_id, tool_calls=None),
        SimpleNamespace(content=text, tool_call_id=None, tool_calls=None),
    ]


def make_agentic_response(
    turns: list[Any] | None = None,
    finish_reason: str = "stop",
    usage: Any | None = None,
    model: str = "mistral-test",
) -> SimpleNamespace:
    """A response whose `message` is None and whose exchange is in `messages`."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=None,
                messages=turns if turns is not None else make_server_tool_turns(),
                finish_reason=finish_reason,
            )
        ],
        usage=usage if usage is not None else make_usage(1, 1, 2),
        model=model,
    )


def make_response(
    content: Any = "ok",
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    usage: Any | None = None,
    model: str = "mistral-test",
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls or [], tool_call_id=None),
                messages=None,
                finish_reason=finish_reason,
            )
        ],
        usage=usage if usage is not None else make_usage(1, 1, 2),
        model=model,
    )


def make_stream_chunk(
    content: Any = None,
    tool_calls: list[Any] | None = None,
    tool_call_id: str | None = None,
    finish_reason: str | None = None,
    usage: Any | None = None,
    model: str | None = None,
) -> SimpleNamespace:
    """One ``CompletionEvent`` — the SDK wraps each chunk in a ``.data`` envelope."""
    return SimpleNamespace(
        data=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=content, tool_calls=tool_calls or [], tool_call_id=tool_call_id),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
            model=model,
        )
    )


class _AsyncIterator:
    def __init__(self, items: Iterable[Any]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> "_AsyncIterator":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration from None


class FakeChat:
    """Stands in for ``Mistral.chat``, capturing the kwargs sent to the API."""

    def __init__(self, response: Any | None = None, stream_chunks: Iterable[Any] = ()) -> None:
        self.response = response if response is not None else make_response()
        self.stream_chunks = list(stream_chunks)
        self.kwargs: dict[str, Any] | None = None

    async def complete_async(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response

    async def stream_async(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return _AsyncIterator(self.stream_chunks)


class FakeHttpResponse:
    def __init__(self, data: bytes, content_type: str = "image/jpeg") -> None:
        self.content = data
        self.headers = {"content-type": content_type}

    def raise_for_status(self) -> None:
        return None


class FakeHttpClient:
    """Stands in for httpx.AsyncClient so image fetches never touch the network."""

    def __init__(
        self, data: bytes = b"\xff\xd8image", content_type: str = "image/jpeg", error: Exception | None = None
    ) -> None:
        self.data = data
        self.content_type = content_type
        self.error = error
        self.urls: list[str] = []

    async def get(self, url: str, **kwargs: Any) -> FakeHttpResponse:
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        return FakeHttpResponse(self.data, self.content_type)

    async def aclose(self) -> None:
        return None


class FakeMistralClient:
    def __init__(self, chat: FakeChat) -> None:
        self.chat = chat


def make_call_context(prompt: list[str] | None = None) -> AsyncMock:
    ctx = AsyncMock()
    ctx.send = AsyncMock()
    ctx.prompt = prompt or []
    return ctx
