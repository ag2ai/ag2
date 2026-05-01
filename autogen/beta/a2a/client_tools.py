# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from typing import Any
from uuid import uuid4

from autogen.beta.annotations import Context
from autogen.beta.events import (
    BaseEvent,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
)
from autogen.beta.middleware import BaseMiddleware
from autogen.beta.tools.final.function_tool import FunctionDefinition, FunctionToolSchema
from autogen.beta.tools.schemas import ToolSchema
from autogen.beta.tools.tool import Tool

from .mappers import TOOL_CALL_REQUEST_KEY, TOOL_CALL_RESULT_KEY
from .streams import StreamOutcome

# Marker prefix that brands a ``HumanInputRequest.content`` as an encoded
# client-tool call rather than a real HITL prompt. The executor's HITL hook
# uses this to decide between the two paths. Lives here (next to encode/decode)
# rather than in executor.py because both sides of the wire need it.
_CLIENT_TOOL_CONTENT_PREFIX = "__ag2_client_tool_call__:"

PENDING_TOOL_CALL_ID_VAR_KEY = "ag:a2a:pending_tool_call_id"
"""``Context.variables`` key tracking the in-flight client-side tool call ID.

Set when ``A2AClient`` returns a ``ModelResponse(tool_calls=[...])`` driven by
a server ``input_required + tool_call_request``. Cleared once the matching
result has been forwarded back to the server.
"""


def schema_to_wire(schema: ToolSchema) -> dict[str, Any] | None:
    """Encode a beta ``ToolSchema`` into the wire dict the server expects.

    Only ``FunctionToolSchema`` is sendable — builtin tools (web search,
    code execution, …) are server-side capabilities that don't make sense as
    client-declared stubs and are silently dropped.
    """
    if not isinstance(schema, FunctionToolSchema):
        return None
    fn = schema.function
    return {
        "name": fn.name,
        "description": fn.description,
        "parameters": fn.parameters,
    }


def schemas_to_wire(schemas: Sequence[ToolSchema]) -> list[dict[str, Any]]:
    return [w for s in schemas if (w := schema_to_wire(s)) is not None]


def parse_tool_call_request(outcome: StreamOutcome) -> ToolCallEvent | None:
    """Read a ``ToolCallEvent`` out of an ``input_required`` outcome, if present.

    The server stamps ``metadata[TOOL_CALL_REQUEST_KEY]`` on the status message
    when the LLM picked a client-side tool. Plain HITL prompts have no such
    marker.
    """
    if not outcome.input_required or not outcome.input_metadata:
        return None
    payload = outcome.input_metadata.get(TOOL_CALL_REQUEST_KEY)
    if not isinstance(payload, dict):
        return None
    call_id = payload.get("id")
    name = payload.get("name")
    if not isinstance(call_id, str) or not isinstance(name, str):
        return None
    arguments = payload.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments) if arguments is not None else "{}"
    return ToolCallEvent(id=call_id, name=name, arguments=arguments)


def find_pending_tool_result(messages: Sequence[BaseEvent], pending_id: str) -> ToolResultEvent | ToolErrorEvent | None:
    """Walk the in-memory history for the result of a previously requested call."""
    for event in reversed(messages):
        if isinstance(event, (ToolResultEvent, ToolErrorEvent)) and event.parent_id == pending_id:
            return event
    return None


def tool_result_payload(event: ToolResultEvent | ToolErrorEvent) -> dict[str, Any]:
    """Encode a beta ``ToolResultEvent`` for the wire (server reads this on follow-up)."""
    if isinstance(event, ToolErrorEvent):
        return {"id": event.parent_id, "error": str(event.error)}
    parts = event.result.parts
    if len(parts) == 1 and hasattr(parts[0], "content"):
        output = parts[0].content
    else:
        output = "\n".join(getattr(p, "content", str(p)) for p in parts)
    return {"id": event.parent_id, "output": output}


def parse_tool_result_request(message_metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Read the server-side stub view of an inbound tool-call result.

    Used by the executor when the client posts a follow-up Message answering a
    previously issued ``tool_call_request``.
    """
    if not message_metadata:
        return None
    payload = message_metadata.get(TOOL_CALL_RESULT_KEY)
    return payload if isinstance(payload, dict) else None


def encode_client_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Encode an LLM tool-call invocation as a marker-prefixed JSON string.

    Counterpart to :func:`decode_client_tool_call`. The result is fed into
    ``context.input(...)`` so the executor's HITL hook routes it through the
    A2A ``tool_call_request`` flow instead of treating it as a real HITL prompt.
    """
    payload = json.dumps(
        {"id": uuid4().hex, "name": name, "arguments": json.dumps(arguments)},
        sort_keys=True,
    )
    return _CLIENT_TOOL_CONTENT_PREFIX + payload


def decode_client_tool_call(content: str) -> dict[str, Any] | None:
    """Reverse of :func:`encode_client_tool_call`.

    Returns the decoded ``{id, name, arguments}`` dict, or ``None`` if
    ``content`` is a plain HITL prompt (no marker, malformed JSON, or missing
    required keys).
    """
    if not content.startswith(_CLIENT_TOOL_CONTENT_PREFIX):
        return None
    raw = content[len(_CLIENT_TOOL_CONTENT_PREFIX) :]
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, dict) or "id" not in decoded or "name" not in decoded:
        return None
    return decoded


def validate_client_tool_parameters(parameters: Any) -> None:
    """Strict structural check for a client-declared JSON Schema.

    Raises ``ValueError`` on any deviation. We only validate the shape that
    matters for an LLM function-tool parameters block — not the full JSON
    Schema spec — so the failure is loud and the diagnostic is concrete.

    A missing or ``null`` ``type`` is accepted (defaults to ``"object"``) since
    schemas going through proto ``Struct`` round-trips can lose explicit fields.
    """
    if not isinstance(parameters, dict):
        raise ValueError(f"client tool parameters must be a JSON object, got {type(parameters).__name__}")
    declared_type = parameters.get("type")
    if declared_type is not None and declared_type != "object":
        raise ValueError(f"client tool parameters.type must be 'object', got {declared_type!r}")
    if "properties" in parameters and not isinstance(parameters["properties"], (dict, type(None))):
        raise ValueError(
            f"client tool parameters.properties must be a JSON object, got {type(parameters['properties']).__name__}"
        )
    if "required" in parameters:
        required = parameters["required"]
        if required is not None and (
            not isinstance(required, list) or not all(isinstance(name, str) for name in required)
        ):
            raise ValueError("client tool parameters.required must be a list of property-name strings")


class ClientToolStub(Tool):
    """Server-side proxy for a tool declared on the A2A client.

    The client sends the JSON Schema over the wire (``Message.metadata`` →
    ``CLIENT_TOOLS_KEY``); the server materialises one ``ClientToolStub`` per
    schema and exposes it to its own LLM. When the LLM picks the tool, the
    stub forwards the call to the client via ``context.input(...)`` keyed
    with the marker-prefixed payload, then wraps whatever the client posts
    back as a ``ToolResultEvent``.

    No ``fast_depends`` / ``inspect.Signature`` plumbing — argument shape is
    whatever the client declared, the model picks values, and the stub
    forwards them verbatim. ``_InputRequiredSignal`` (a ``BaseException``)
    propagates through the bare ``__call__`` so the executor can translate
    it into an A2A ``requires_input`` task transition.
    """

    __slots__ = ("name", "schema")

    def __init__(self, name: str, description: str, parameters: dict[str, Any]) -> None:
        self.name = name
        self.schema = FunctionToolSchema(
            function=FunctionDefinition(
                name=name,
                description=description,
                parameters=parameters,
            )
        )

    async def schemas(self, context: "Context") -> list[FunctionToolSchema]:
        return [self.schema]

    def register(
        self,
        stack: "ExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        async def execute(event: "ToolCallEvent", context: "Context") -> None:
            result = await self(event, context)
            await context.send(result)

        stack.enter_context(
            context.stream.where(ToolCallEvent.name == self.name).sub_scope(execute),
        )

    async def __call__(
        self,
        event: "ToolCallEvent",
        context: "Context",
    ) -> "ToolResultEvent | ToolErrorEvent":
        try:
            forwarded = await context.input(
                encode_client_tool_call(self.name, event.serialized_arguments),
            )
        except Exception as exc:
            return ToolErrorEvent.from_call(event, error=exc)
        return ToolResultEvent.from_call(event, result=forwarded)


def build_client_tool_stubs(schemas: list[dict[str, Any]]) -> list[ClientToolStub]:
    """Materialise stub ``ClientToolStub``s from client-supplied JSON schemas.

    Tools the client declares with no arguments come over the wire with a
    degenerate ``{"type": "null"}`` schema (artefact of beta's ``tool()``
    builder for arg-less functions). Any non-``"object"`` ``type`` is
    coerced to a canonical empty-object schema before validation so the
    server LLM sees a well-formed JSON Schema.
    """
    stubs: list[ClientToolStub] = []
    for schema in schemas:
        name = schema.get("name") or "client_tool"
        description = schema.get("description") or f"Client-side tool {name}."
        parameters = schema.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            parameters = {"type": "object", "properties": {}}
        validate_client_tool_parameters(parameters)
        stubs.append(ClientToolStub(name=name, description=description, parameters=parameters))
    return stubs
