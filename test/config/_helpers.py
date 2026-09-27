# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from ag2.tools.final import FunctionDefinition, FunctionToolSchema


@dataclass
class ToolStub:
    name: str
    schema: FunctionToolSchema


def make_parameterless_tool() -> ToolStub:
    return ToolStub(
        name="ask_human",
        schema=FunctionToolSchema(
            function=FunctionDefinition(
                name="ask_human",
                description="Ask the human for input.",
                parameters={"type": "null"},
            )
        ),
    )


def make_tool() -> ToolStub:
    return ToolStub(
        name="search_docs",
        schema=FunctionToolSchema(
            function=FunctionDefinition(
                name="search_docs",
                description="Search documentation by query.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                    "required": ["query"],
                },
            )
        ),
    )


class WireRecorder:
    """An `httpx.MockTransport` handler that records each request and answers with the scripted responses in order.

    The provider SDK builds the request and parses the reply, so a test asserts on what reaches the API.
    """

    def __init__(self, *responses: httpx.Response) -> None:
        self.requests: list[httpx.Request] = []
        self._responses = list(responses)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        """Each request's JSON body, in order."""
        return [json.loads(r.content) for r in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert self._responses, f"unscripted request: {request.method} {request.url}"
        return self._responses.pop(0)


def json_response(body: Mapping[str, Any], status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json=body)


def sse_response(*events: Mapping[str, Any], done: bool = True) -> httpx.Response:
    """A `text/event-stream` reply carrying each event as one `data:` line, then `[DONE]`."""
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    if done:
        lines.append("data: [DONE]\n\n")
    return httpx.Response(200, content="".join(lines).encode(), headers={"content-type": "text/event-stream"})
