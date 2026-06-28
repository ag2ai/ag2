# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from mcp.types import Tool as MCPTool

from ag2.agent import Agent

if TYPE_CHECKING:
    from ag2.response import ResponseProto


def build_ask_tool(
    agent: Agent,
    *,
    tool_name: str = "ask",
    tool_description: str | None = None,
    response_schema: "ResponseProto[Any] | None" = None,
) -> MCPTool:
    """Build the single conversational MCP tool that fronts ``agent.ask()``.

    The tool takes a required ``message`` and an optional ``context`` string —
    mirroring :meth:`Agent.as_tool`'s ``objective`` / ``context`` shape. When
    ``response_schema`` is an object schema, it is advertised as the tool's
    ``outputSchema`` so MCP clients receive validated ``structuredContent``
    (see :mod:`ag2.mcp.executor`).
    """
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message or task to send to the agent.",
            },
            "context": {
                "type": "string",
                "description": "Optional additional context to prepend to the message.",
            },
        },
        "required": ["message"],
    }
    kwargs: dict[str, Any] = {
        "name": tool_name,
        "description": tool_description or f"Send a message to the '{agent.name}' AG2 agent and receive its reply.",
        "inputSchema": input_schema,
    }
    output_schema = object_output_schema(response_schema)
    if output_schema is not None:
        kwargs["outputSchema"] = output_schema
    return MCPTool(**kwargs)


def object_output_schema(response_schema: "ResponseProto[Any] | None") -> dict[str, Any] | None:
    """Return the JSON schema iff it is an object schema, else ``None``.

    MCP ``outputSchema`` / ``structuredContent`` must be objects, so non-object
    response schemas (scalars, unions) are not advertised — those replies still
    flow back as plain text content.
    """
    json_schema = response_schema.json_schema if response_schema is not None else None
    if isinstance(json_schema, dict) and json_schema.get("type") == "object":
        return json_schema
    return None
