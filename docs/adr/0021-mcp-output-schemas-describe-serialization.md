---
status: accepted
date: 2026-09-29
---

# 0021. MCP model output schemas describe serialization, not LLM validation

## Context

`ResponseSchema.json_schema` describes what an LLM must produce for validation.
MCP's `structuredContent`, however, contains the validated model's
`model_dump(mode="json")`. Aliases, excluded fields and serializers can make
these representations differ. Advertising the validation schema makes MCP
clients reject otherwise valid tool results.

## Decision

For a `ResponseSchema` backed by a Pydantic model, MCP derives an object output
schema in serialization mode. The schema follows each type's effective
`serialize_by_alias` policy, just as the existing dump does. This applies to
the served agent's `ask` and automatically derived `@mcp_tool` return schemas.
The served tool's declaration is built once and reused when shaping replies.

`ResponseSchema.json_schema` and LLM validation do not change. Neither do the
returned field names. Explicit tool output schemas, custom `ResponseProto`
implementations and standalone dataclasses retain their existing behavior.
Manually constructed results are not rewritten: a handler annotated with a
model still promises that model's output shape. A different manual shape needs
an explicit output schema or a `CallToolResult` return annotation.

This supersedes only ADR 0015's clause 3 claim that a Pydantic response schema
is advertised verbatim. Conversation handles still travel in text and `_meta`,
never in `structuredContent`.

## Consequences

- MCP clients validate the representation they actually receive.
- Forcing one alias policy across all models is avoided: it would rename fields
  for callers that already rely on a different default or nested configuration.
- Serialization schema generation stays in the MCP layer; providers and the
  `ResponseProto` interface need no new output-schema API.
