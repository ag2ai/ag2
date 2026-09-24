# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
from mistralai.client.models import DeltaMessage, FunctionCall, ToolCall

from ag2.config.mistral import MistralClient


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


def make_turn(
    content: str = "",
    *,
    tool_call_id: str | None = None,
    tool_calls: list[tuple[str, str, str]] | None = None,
) -> DeltaMessage:
    """One turn of `ChatCompletionChoice.messages`, which the SDK parses as a `DeltaMessage`."""
    return DeltaMessage(
        content=content,
        tool_call_id=tool_call_id,
        tool_calls=[ToolCall(id=i, function=FunctionCall(name=n, arguments=a)) for i, n, a in tool_calls]
        if tool_calls
        else None,
    )


def make_server_tool_turns(
    call_id: str = "gen_1",
    name: str = "generate_image",
    arguments: str = '{"prompt": "a red circle"}',
    url: str = "https://example.com/generated.jpg",
    text: str = "Here is your image.",
) -> list[DeltaMessage]:
    """The `messages` trace a server-executed tool produces: call, result, answer."""
    return [
        make_turn(tool_calls=[(call_id, name, arguments)]),
        make_turn(f'{{"url": "{url}"}}', tool_call_id=call_id),
        make_turn(text),
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
                message=SimpleNamespace(content=content, tool_calls=tool_calls or []),
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
        self._kwargs: dict[str, Any] | None = None

    @property
    def kwargs(self) -> dict[str, Any]:
        """What the last call was made with; fails the test if none was."""
        assert self._kwargs is not None, "the chat API was never called"
        return self._kwargs

    async def complete_async(self, **kwargs: Any) -> Any:
        self._kwargs = kwargs
        return self.response

    async def stream_async(self, **kwargs: Any) -> Any:
        self._kwargs = kwargs
        return _AsyncIterator(self.stream_chunks)


class FakeHttpClient(httpx.AsyncClient):
    """An httpx client answered by a local transport, so image fetches never touch the network."""

    def __init__(
        self, data: bytes = b"\xff\xd8image", content_type: str = "image/jpeg", error: Exception | None = None
    ) -> None:
        self.data = data
        self.content_type = content_type
        self.error = error
        self.urls: list[str] = []
        super().__init__(transport=httpx.MockTransport(self._answer))

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        if self.error is not None:
            raise self.error
        return httpx.Response(200, content=self.data, headers={"content-type": self.content_type})


class FakeMistralClient:
    def __init__(self, chat: FakeChat) -> None:
        self.chat = chat


def install_fake_sdk(client: MistralClient, chat: FakeChat) -> None:
    """Stand `chat` in for the SDK client `client` would otherwise build."""
    client._client = FakeMistralClient(chat)  # type: ignore[assignment]  # the tests read what the chat API was called with, which the SDK turns into a request body


def make_call_context(prompt: list[str] | None = None) -> AsyncMock:
    ctx = AsyncMock()
    ctx.send = AsyncMock()
    ctx.prompt = prompt or []
    return ctx
