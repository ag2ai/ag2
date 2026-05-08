# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any

from autogen.beta.events import (
    Input,
    TextInput,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResult,
    ToolResultEvent,
)
from autogen.beta.tools.final.function_tool import FunctionDefinition, FunctionToolSchema


def schemas_to_payload(schemas: Iterable[FunctionToolSchema]) -> dict[str, Any]:
    """Serialize tool schemas for the ``tool-schemas+json`` extension Part.

    Sent by the client to the server when opening a Task — tells the server
    which tools live on the calling side.
    """
    return {
        "tools": [
            {
                "name": s.function.name,
                "description": s.function.description,
                "parameters": s.function.parameters,
            }
            for s in schemas
        ]
    }


def payload_to_schemas(payload: dict[str, Any]) -> list[FunctionToolSchema]:
    return [
        FunctionToolSchema(
            function=FunctionDefinition(
                name=t["name"],
                description=t.get("description", "") or "",
                parameters=t.get("parameters", {}) or {},
            )
        )
        for t in payload.get("tools", [])
    ]


def call_to_payload(call: ToolCallEvent) -> dict[str, Any]:
    """Serialize a single tool invocation for the ``tool-call+json`` Part."""
    return {
        "id": call.id,
        "name": call.name,
        "arguments": call.arguments,
    }


def payload_to_call(payload: dict[str, Any]) -> ToolCallEvent:
    return ToolCallEvent(
        id=str(payload["id"]),
        name=str(payload["name"]),
        arguments=str(payload.get("arguments", "{}")),
    )


def results_to_payload(results: Iterable[ToolResultEvent]) -> dict[str, Any]:
    """Serialize tool results for the ``tool-result+json`` Part.

    Sent by the client back to the server after locally executing the tools
    that the server requested via ``tool-call+json`` artifacts. ``name`` is
    threaded through so the stateless server can rebuild a
    ``ToolResultEvent`` without consulting any session state.
    """
    return {
        "results": [
            {
                "id": r.parent_id,
                "name": r.name,
                "content": _result_to_text(r.result),
                "error": str(r.error) if isinstance(r, ToolErrorEvent) else None,
            }
            for r in results
        ]
    }


def payload_to_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Decode a ``tool-result+json`` payload to a list of result records.

    Returns raw dicts (``id``/``name``/``content``/``error``) rather than
    ``ToolResultEvent`` objects — the executor uses ``name`` directly to
    construct events without any pending-call lookup.
    """
    return [
        {
            "id": str(r["id"]),
            "name": r.get("name"),
            "content": r.get("content", "") or "",
            "error": r.get("error"),
        }
        for r in payload.get("results", [])
    ]


def _result_to_text(result: ToolResult) -> str:
    chunks: list[str] = []
    for part in result.parts:
        chunks.append(_input_to_text(part))
    return "".join(chunks)


def _input_to_text(part: Input) -> str:
    if isinstance(part, TextInput):
        return part.content
    return str(part)
