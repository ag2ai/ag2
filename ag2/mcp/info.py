# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from mcp.types import Tool as MCPTool

from ag2.agent import Agent

from .sessions import ConversationBounds

if TYPE_CHECKING:
    from ag2.response import ResponseProto


def build_ask_tool(
    agent: Agent,
    *,
    tool_name: str = "ask",
    tool_description: str | None = None,
    response_schema: "ResponseProto[Any] | None" = None,
    conversation_bounds: ConversationBounds | None = None,
) -> MCPTool:
    """Build the single conversational MCP tool that fronts ``agent.ask()``.

    The tool takes a required ``message`` and an optional ``context`` string —
    mirroring :meth:`Agent.as_tool`'s ``objective`` / ``context`` shape. When
    ``response_schema`` is an object schema, it is advertised as the tool's
    ``outputSchema`` so MCP clients receive validated ``structuredContent``
    (see :mod:`ag2.mcp.executor`).

    ``conversation_bounds`` — the registry's bound and idle expiry — adds the
    optional ``conversation`` argument that continues a conversation, and is
    worded into its description because the protocol requires a stateful
    handle's lifetime to be documented there. Pass ``None`` when conversations
    are off: advertising an argument that cannot work would invite a guaranteed
    error, and suppressing it mirrors how ``outputSchema`` appears only when the
    agent has a response schema. The variation is by server configuration, fixed
    for the process — not by connection state, which the protocol forbids.
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
    if conversation_bounds is not None:
        input_schema["properties"]["conversation"] = {
            "type": "string",
            "description": (
                "Opaque handle naming the conversation to continue, as returned by an earlier "
                "call to this tool. Omit it to start a new conversation; the handle for it comes "
                f"back with the reply. {_lifetime_sentence(conversation_bounds)}"
            ),
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


def _lifetime_sentence(bounds: ConversationBounds) -> str:
    """How long a conversation lives, in a sentence a client can read."""
    # A fragment rather than a clause, so it reads as the tail of either branch.
    bound = f"once it is not among the {bounds.max_conversations} most recently used"
    if bounds.ttl is None:
        return f"Lifetime: a conversation is dropped {bound}."
    return f"Lifetime: a conversation is dropped after {bounds.ttl:g} seconds idle, or {bound}."


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
