# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import binascii
import json
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from aiohttp import web
from fast_depends.use import SerializerCls
from multidict import CIMultiDictProxy

from ag2 import Context, MemoryStream
from ag2.config import BedrockConfig
from ag2.events import BaseEvent, ModelRequest, ModelResponse, TextInput
from ag2.response import ResponseProto
from ag2.tools.schemas import ToolSchema


def make_converse_response(
    content: list[dict[str, Any]] | None = None,
    stop_reason: str = "end_turn",
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canned Converse API response."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": content if content is not None else [{"text": "ok"}],
            },
        },
        "stopReason": stop_reason,
        "usage": usage if usage is not None else {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    }


@dataclass(slots=True)
class BedrockRequest:
    path: str
    headers: CIMultiDictProxy[str]
    body: dict[str, Any]


class FakeBedrock:
    """Local Bedrock Runtime endpoint: records requests, answers with scripted payloads.

    ``stream_events`` are ``(event_type, payload)`` pairs, sent as AWS eventstream frames.
    ``status`` other than 200 answers every call with an error; ``hang`` holds every answer
    until ``release()``.
    """

    def __init__(self) -> None:
        self.url = ""
        self.requests: list[BedrockRequest] = []
        self.response = make_converse_response()
        self.stream_events: list[tuple[str, dict[str, Any]]] = []
        self.status = 200
        self.hang = False
        self._released = asyncio.Event()

    def release(self) -> None:
        self._released.set()

    def config(self, **overrides: Any) -> BedrockConfig:
        options: dict[str, Any] = {
            "model": "m1",
            "region_name": "us-east-1",
            "aws_access_key_id": "AKIDTEST",
            "aws_secret_access_key": "secret",
            "endpoint_url": self.url,
            **overrides,
        }
        return BedrockConfig(**options)

    @property
    def body(self) -> dict[str, Any]:
        [request] = self.requests
        return request.body

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/model/{model}/converse", self._converse)
        app.router.add_post("/model/{model}/converse-stream", self._converse_stream)
        return app

    async def _record(self, request: web.Request) -> web.StreamResponse | None:
        self.requests.append(BedrockRequest(request.path, request.headers, await request.json()))
        if self.hang:
            await self._released.wait()
        if self.status != 200:
            return web.json_response({"message": "boom"}, status=self.status)
        return None

    async def _converse(self, request: web.Request) -> web.StreamResponse:
        return await self._record(request) or web.json_response(self.response)

    async def _converse_stream(self, request: web.Request) -> web.StreamResponse:
        if error := await self._record(request):
            return error
        response = web.StreamResponse(headers={"content-type": "application/vnd.amazon.eventstream"})
        await response.prepare(request)
        for event_type, payload in self.stream_events:
            await response.write(eventstream_frame(event_type, payload))
        await response.write_eof()
        return response


def eventstream_frame(event_type: str, payload: dict[str, Any]) -> bytes:
    """One AWS eventstream message: prelude, headers, JSON payload, each CRC32-guarded."""
    headers = b"".join(
        _string_header(name, value)
        for name, value in (
            (":event-type", event_type),
            (":content-type", "application/json"),
            (":message-type", "event"),
        )
    )
    body = json.dumps(payload).encode()
    prelude = struct.pack("!II", 16 + len(headers) + len(body), len(headers))
    message = prelude + struct.pack("!I", binascii.crc32(prelude)) + headers + body
    return message + struct.pack("!I", binascii.crc32(message))


def _string_header(name: str, value: str) -> bytes:
    encoded_name, encoded_value = name.encode(), value.encode()
    # Header value type 7 is a string
    return (
        struct.pack("!B", len(encoded_name))
        + encoded_name
        + b"\x07"
        + struct.pack("!H", len(encoded_value))
        + encoded_value
    )


def recording(stream: MemoryStream) -> list[BaseEvent]:
    """Every event sent on `stream`, transient ones included — history keeps none of those."""
    captured: list[BaseEvent] = []

    async def capture(event: BaseEvent) -> None:
        captured.append(event)

    stream.subscribe(capture)
    return captured


async def ask(
    config: BedrockConfig,
    *,
    prompt: Sequence[str] = (),
    tools: Iterable[ToolSchema] = (),
    response_schema: ResponseProto | None = None,
) -> tuple[ModelResponse, list[BaseEvent]]:
    """One client call; returns the response and every event the client sent."""
    stream = MemoryStream()
    events = recording(stream)
    response = await config.create()(
        messages=[ModelRequest([TextInput("hello")])],
        context=Context(stream=stream, prompt=list(prompt)),
        tools=tools,
        response_schema=response_schema,
        serializer=SerializerCls,
    )
    return response, events
