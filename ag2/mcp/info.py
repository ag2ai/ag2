# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from types import GenericAlias
from typing import TYPE_CHECKING, Any, cast

from mcp.types import Tool as MCPTool
from pydantic import BaseModel, TypeAdapter
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema

from ag2.agent import Agent
from ag2.response import ResponseSchema

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

    The tool takes a required ``message`` and an optional ``context`` string. An
    object ``response_schema`` is advertised as the tool's ``outputSchema``, so
    clients receive validated ``structuredContent``.

    ``conversation_bounds`` — the registry's bound and idle expiry — adds the
    optional ``conversation`` argument that continues a conversation and words
    its lifetime into the description, as the protocol requires. Pass ``None``
    when conversations are off, so no argument that cannot work is advertised.
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

    MCP ``outputSchema`` must be an object, so scalar or union response schemas
    are not advertised — those replies flow back as plain text content.
    """
    json_schema = response_schema.json_schema if response_schema is not None else None
    if isinstance(json_schema, dict) and isinstance(response_schema, ResponseSchema):
        model = response_schema.types
        if isinstance(model, type) and not isinstance(model, GenericAlias) and issubclass(model, BaseModel):
            # The LLM schema describes validation input; MCP describes the value
            # after to_structured_dict calls model_dump(mode="json").
            output_schema = TypeAdapter(model).json_schema(mode="serialization", schema_generator=_OutputSchema)
            # ResponseSchema lifts these into its name/description unless the
            # caller supplied them explicitly. Keep that presentation unchanged.
            for key in ("title", "description"):
                if key not in json_schema:
                    output_schema.pop(key, None)
            json_schema = output_schema
    if isinstance(json_schema, dict) and json_schema.get("type") == "object":
        return json_schema
    return None


class _OutputSchema(GenerateJsonSchema):
    """Match model_dump's default, per-type serialize_by_alias policy.

    Passing by_alias=False or True to the generator would override every nested
    type's policy. Core configs also capture the policy inherited by dataclasses.
    TypedDicts use the enclosing serializer's policy instead.
    """

    def generate_inner(
        self,
        schema: core_schema.CoreSchema
        | core_schema.ModelField
        | core_schema.DataclassField
        | core_schema.TypedDictField
        | core_schema.ComputedField,
    ) -> JsonSchemaValue:
        if schema["type"] not in ("model", "dataclass"):
            return super().generate_inner(schema)
        previous = self.by_alias
        config = cast(core_schema.CoreConfig, schema.get("config", {}))
        self.by_alias = config.get("serialize_by_alias", False)
        try:
            return super().generate_inner(schema)
        finally:
            self.by_alias = previous
