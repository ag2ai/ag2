# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from mcp.types import Tool as MCPTool

from autogen.beta.agent import Agent

_DEFAULT_VERSION = "0.1.0"


def build_server_info(
    agent: Agent,
    *,
    name: str | None = None,
    version: str | None = None,
    instructions: str | None = None,
) -> tuple[str, str, str | None]:
    """Resolve the ``(name, version, instructions)`` triple for an MCP server.

    Defaults: ``name`` -> ``agent.name``, ``version`` -> ``"0.1.0"``,
    ``instructions`` -> the agent's first static system prompt (if any). The
    ``instructions`` string is surfaced to MCP clients as server-level guidance.
    """
    resolved_name = name or agent.name
    resolved_version = version or _DEFAULT_VERSION
    resolved_instructions = instructions if instructions is not None else _agent_instructions(agent)
    return resolved_name, resolved_version, resolved_instructions


def build_ask_tool(
    agent: Agent,
    *,
    tool_name: str = "ask",
    tool_description: str | None = None,
) -> MCPTool:
    """Build the single conversational MCP tool that fronts ``agent.ask()``.

    The tool takes a required ``message`` and an optional ``context`` string —
    mirroring :meth:`Agent.as_tool`'s ``objective`` / ``context`` shape. When the
    agent declares a structured ``response_schema`` whose JSON schema is an object,
    that schema is advertised as the tool's ``outputSchema`` so MCP clients receive
    validated ``structuredContent`` (see :mod:`autogen.beta.mcp.executor`).
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
        "description": tool_description or _ask_description(agent),
        "inputSchema": input_schema,
    }
    output_schema = _output_schema(agent)
    if output_schema is not None:
        kwargs["outputSchema"] = output_schema
    return MCPTool(**kwargs)


def _agent_instructions(agent: Agent) -> str | None:
    return agent._system_prompt[0] if agent._system_prompt else None


def _ask_description(agent: Agent) -> str:
    base = f"Send a message to the '{agent.name}' AG2 agent and receive its reply."
    instructions = _agent_instructions(agent)
    return f"{base} {instructions}" if instructions else base


def _output_schema(agent: Agent) -> dict[str, Any] | None:
    """Return the agent's structured-output JSON schema iff it is an object schema.

    MCP ``outputSchema`` / ``structuredContent`` must be objects, so non-object
    response schemas (scalars, unions) are not advertised — those replies still
    flow back as plain text content.
    """
    schema = agent._response_schema
    json_schema = schema.json_schema if schema is not None else None
    if isinstance(json_schema, dict) and json_schema.get("type") == "object":
        return json_schema
    return None
