# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Responses API scaffolding shared by the hosted-tool tests.

One place to build a crafted Responses payload and drive the real
`OpenAIResponsesConfig` over it, so the tests that read the resulting events say
only what they are about.
"""

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


def response(*output: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-5",
        "status": "completed",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": list(output),
        "usage": USAGE,
    }


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
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()

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
) -> None:
    """One turn against `model_config`, with everything it emits landing on `stream`."""
    await model_config.create()(
        messages=[ModelRequest([TextInput("go")])],
        context=Context(stream=stream),
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
