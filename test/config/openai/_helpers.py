# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Responses API scaffolding shared by the hosted-tool tests."""

import json
from typing import Any

import httpx2
from fast_depends.use import SerializerCls

from ag2 import Context, MemoryStream
from ag2.config import OpenAIResponsesConfig
from ag2.events import (
    BaseEvent,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    ModelRequest,
    ModelResponse,
    TextInput,
)
from ag2.tools.schemas import ToolSchema

USAGE = {
    "input_tokens": 1,
    "output_tokens": 1,
    "total_tokens": 2,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0},
}


MCP_CALL = {
    "id": "mcp_1",
    "type": "mcp_call",
    "name": "ask_question",
    "server_label": "deepwiki",
    "arguments": '{"question": "what is ag2?"}',
    "output": "an agent framework",
    "status": "completed",
}

SHELL_CALL = {
    "id": "sh_1",
    "type": "shell_call",
    "call_id": "call_1",
    "status": "completed",
    "action": {"commands": ["echo hi", "ls"], "timeout_ms": 1000},
}

SHELL_OUTPUT = {
    "id": "sho_1",
    "type": "shell_call_output",
    "call_id": "call_1",
    "status": "completed",
    "output": [{"stdout": "hi\n", "stderr": "warning\n", "outcome": {"type": "exit", "exit_code": 0}}],
}


def message(text: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def response(*output: dict[str, Any], response_id: str = "resp_1", **extra: Any) -> dict[str, Any]:
    """One Responses payload. `extra` adds fields the SDK may not model, such as diagnostics."""
    return {
        "id": response_id,
        "object": "response",
        "created_at": 0,
        "model": "gpt-5",
        "status": "completed",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": list(output),
        "usage": USAGE,
        **extra,
    }


def sse(events: list[dict[str, Any]]) -> bytes:
    """`events` as one SSE stream body."""
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def recording(stream: MemoryStream) -> list[BaseEvent]:
    """Every event sent on `stream`, transient ones included — history keeps none of those."""
    captured: list[BaseEvent] = []

    async def capture(event: BaseEvent) -> None:
        captured.append(event)

    stream.subscribe(capture)
    return captured


def capturing_config(
    *turns: dict[str, Any],
    stream: bool = False,
) -> tuple[OpenAIResponsesConfig, list[dict[str, Any]]]:
    """A config replaying one payload per call, and the list its request bodies land in."""
    remaining = list(turns)
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        payload = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        if stream:
            events = [
                {"type": "response.created", "sequence_number": 0, "response": payload},
                {"type": "response.completed", "sequence_number": 1, "response": payload},
            ]
            return httpx2.Response(200, content=sse(events), headers={"content-type": "text/event-stream"})
        return httpx2.Response(200, json=payload)

    config = OpenAIResponsesConfig(
        model="gpt-5",
        api_key="test",
        streaming=stream,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    return config, bodies


def config(*turns: dict[str, Any]) -> OpenAIResponsesConfig:
    """A config whose transport replays one crafted Responses payload per call."""
    remaining = list(turns)

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=remaining.pop(0) if len(remaining) > 1 else remaining[0])

    return OpenAIResponsesConfig(
        model="gpt-5",
        api_key="test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


def streaming_config(events: list[dict[str, Any]]) -> OpenAIResponsesConfig:
    """A streaming config whose transport replays `events` as one SSE stream."""
    body = sse(events)

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})

    return OpenAIResponsesConfig(
        model="gpt-5",
        api_key="test",
        streaming=True,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


async def ask(
    model_config: OpenAIResponsesConfig,
    *,
    stream: MemoryStream,
    tools: list[ToolSchema] | None = None,
) -> ModelResponse:
    """One turn against `model_config`, with everything it emits landing on `stream`."""
    return await ask_client(model_config.create(), stream=stream, tools=tools)


async def ask_client(
    client: Any,
    *,
    stream: MemoryStream | None = None,
    tools: list[ToolSchema] | None = None,
) -> ModelResponse:
    """One turn against an already-built `client`, so several turns can share its state."""
    return await client(
        messages=[ModelRequest([TextInput("go")])],
        context=Context(stream=stream or MemoryStream()),
        tools=tools or [],
        response_schema=None,
        serializer=SerializerCls,
    )


async def events_of(*output: dict[str, Any]) -> list[BaseEvent]:
    """The history of one turn whose response carried `output`."""
    stream = MemoryStream()
    await ask(config(response(*output)), stream=stream)
    return list(await stream.history.get_events())


def calls(events: list[BaseEvent]) -> list[BuiltinToolCallEvent]:
    return [e for e in events if isinstance(e, BuiltinToolCallEvent)]


def results(events: list[BaseEvent]) -> list[BuiltinToolResultEvent]:
    return [e for e in events if isinstance(e, BuiltinToolResultEvent)]
