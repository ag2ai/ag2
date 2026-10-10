# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from typing import Any

from aiohttp import web
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config.ollama import OllamaConfig
from ag2.events import BaseEvent, ModelRequest, ModelResponse, TextInput


def make_chunk(
    *,
    content: str = "",
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
    done: bool = False,
) -> dict[str, Any]:
    """One `/api/chat` payload: a stream chunk, or the whole reply when not streaming."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [{"function": {"name": name, "arguments": args}} for name, args in tool_calls]
    chunk: dict[str, Any] = {"model": "m1", "message": message, "done": done}
    if done:
        chunk |= {"done_reason": "stop", "prompt_eval_count": 4, "eval_count": 6}
    return chunk


@dataclass(slots=True)
class OllamaRequest:
    path: str
    body: dict[str, Any]


class FakeOllama:
    """Local Ollama endpoint: records requests, answers with scripted payloads.

    `chunks` are sent as newline-delimited JSON when the request asks to stream; otherwise the
    last chunk, which carries the whole reply, is sent as one JSON document.
    """

    def __init__(self) -> None:
        self.url = ""
        self.requests: list[OllamaRequest] = []
        self.chunks: list[dict[str, Any]] = [make_chunk(content="ok", done=True)]

    def config(self, **overrides: Any) -> OllamaConfig:
        options: dict[str, Any] = {"model": "m1", "host": self.url, **overrides}
        return OllamaConfig(**options)

    @property
    def body(self) -> dict[str, Any]:
        [request] = self.requests
        return request.body

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/api/chat", self._chat)
        return app

    async def _chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests.append(OllamaRequest(request.path, body))
        if not body.get("stream"):
            return web.json_response(self.chunks[-1])
        response = web.StreamResponse(headers={"content-type": "application/x-ndjson"})
        await response.prepare(request)
        for chunk in self.chunks:
            await response.write(json.dumps(chunk).encode() + b"\n")
        await response.write_eof()
        return response


def recording(stream: MemoryStream) -> list[BaseEvent]:
    """Every event sent on `stream`, transient ones included — history keeps none of those."""
    captured: list[BaseEvent] = []

    async def capture(event: BaseEvent) -> None:
        captured.append(event)

    stream.subscribe(capture)
    return captured


async def ask(config: OllamaConfig) -> tuple[ModelResponse, list[BaseEvent]]:
    """One client call; returns the response and every event the client sent."""
    stream = MemoryStream()
    events = recording(stream)
    response = await config.create()(
        messages=[ModelRequest([TextInput("hello")])],
        context=Context(stream=stream),
        tools=[],
        response_schema=None,
        serializer=SerializerCls,
    )
    return response, events
