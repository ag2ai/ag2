# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import httpx
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config.ollama import OllamaConfig
from ag2.events import BaseEvent, ModelRequest, ModelResponse, TextInput


def make_chunk(
    *,
    content: str = "",
    thinking: str | None = None,
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
    done: bool = False,
) -> dict[str, Any]:
    """One `/api/chat` payload: a stream chunk, or the whole reply when not streaming."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if thinking:
        message["thinking"] = thinking
    if tool_calls:
        message["tool_calls"] = [{"function": {"name": name, "arguments": args}} for name, args in tool_calls]
    chunk: dict[str, Any] = {"model": "m1", "message": message, "done": done}
    if done:
        chunk |= {"done_reason": "stop", "prompt_eval_count": 4, "eval_count": 6}
    return chunk


class FakeOllama:
    """Scripts `/api/chat` replies on an `httpx.MockTransport` and records the requests it gets.

    `chunks` are sent as newline-delimited JSON when the request asks to stream; otherwise the
    last chunk, which carries the whole reply, is sent as one JSON document.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.chunks: list[dict[str, Any]] = [make_chunk(content="ok", done=True)]

    def client(self, **kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle), **kwargs)

    def config(self, http_client: httpx.AsyncClient | None = None, **overrides: Any) -> OllamaConfig:
        return OllamaConfig(**{"model": "m1", "http_client": http_client or self.client(), **overrides})

    @property
    def request(self) -> httpx.Request:
        [request] = self.requests
        return request

    @property
    def body(self) -> dict[str, Any]:
        return json.loads(self.request.content)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not json.loads(request.content).get("stream"):
            return httpx.Response(200, json=self.chunks[-1])
        body = b"".join(json.dumps(chunk).encode() + b"\n" for chunk in self.chunks)
        return httpx.Response(200, headers={"content-type": "application/x-ndjson"}, content=body)


def recording(stream: MemoryStream) -> list[BaseEvent]:
    """Every event sent on `stream`, transient ones included — history keeps none of those."""
    captured: list[BaseEvent] = []

    async def capture(event: BaseEvent) -> None:
        captured.append(event)

    stream.subscribe(capture)
    return captured


async def ask(
    config: OllamaConfig,
    messages: list[BaseEvent] | None = None,
    tools: list[Any] | None = None,
) -> tuple[ModelResponse, list[BaseEvent]]:
    """One client call; returns the response and every event the client sent."""
    stream = MemoryStream()
    events = recording(stream)
    response = await config.create()(
        messages=messages or [ModelRequest([TextInput("hello")])],
        context=Context(stream=stream),
        tools=tools or [],
        response_schema=None,
        serializer=SerializerCls,
    )
    return response, events
