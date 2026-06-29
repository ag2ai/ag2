# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Mapping
from typing import Any

from ag2.events import (
    ToolCallEvent,
    ToolErrorEvent,
    ToolResult,
    ToolResultEvent,
)
from ag2.tools.final.function_tool import FunctionDefinition, FunctionToolSchema

from ..errors import RehydratedToolError
from .parts import tool_result_to_text


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


def payload_to_call(payload: Mapping[str, Any]) -> ToolCallEvent:
    return ToolCallEvent(
        id=str(payload["id"]),
        name=str(payload["name"]),
        arguments=str(payload.get("arguments", "{}")),
    )


def result_event_to_payload(ev: ToolResultEvent) -> dict[str, Any]:
    """Serialize one ``ToolResultEvent`` to a wire dict.

    Mirrors :func:`payload_to_result_event`. Wire key is ``id`` to stay
    aligned with ``tool-call+json`` (server thinks in call ids on the
    wire, even though AG2 internally calls the field ``parent_id``).
    """
    return {
        "id": ev.parent_id,
        "name": ev.name,
        "content": tool_result_to_text(ev.result),
        "error": str(ev.error) if isinstance(ev, ToolErrorEvent) else None,
    }


def payload_to_result_event(entry: Mapping[str, Any]) -> ToolResultEvent:
    """Build a ``ToolResultEvent`` (or ``ToolErrorEvent``) from a wire dict.

    Accepts both ``id`` (``tool-result+json`` Part) and ``parent_id``
    (``ag2.history+json`` event) — the two wire flavours diverged
    historically; this helper normalises both. ``error`` flips the
    return type to ``ToolErrorEvent`` carrying ``RehydratedToolError``.
    """
    parent_id = str(entry.get("parent_id") or entry.get("id") or "")
    name = entry.get("name")
    content = str(entry.get("content", "") or "")
    result = ToolResult(content)
    error = entry.get("error")
    if error:
        return ToolErrorEvent(
            parent_id=parent_id,
            name=name,
            error=RehydratedToolError(error),
            result=result,
        )
    return ToolResultEvent(parent_id=parent_id, name=name, result=result)


def results_to_payload(results: Iterable[ToolResultEvent]) -> dict[str, Any]:
    """Serialize tool results for the ``tool-result+json`` Part.

    Sent by the client back to the server after locally executing the tools
    that the server requested via ``tool-call+json`` artifacts. ``name`` is
    threaded through so the stateless server can rebuild a
    ``ToolResultEvent`` without consulting any session state.
    """
    return {"results": [result_event_to_payload(r) for r in results]}


def payload_to_results(payload: Mapping[str, Any]) -> list[ToolResultEvent]:
    """Decode a ``tool-result+json`` payload to a list of ``ToolResultEvent``s.

    Returns ready-to-use events (with ``ToolErrorEvent`` for entries
    carrying ``error``), so the executor doesn't have to hand-roll the
    rebuild and can't lose the error branch by accident.
    """
    return [payload_to_result_event(r) for r in payload.get("results", []) if isinstance(r, Mapping)]
