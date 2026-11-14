---
status: accepted
date: 2026-11-13
---

# 0038. A decision `response_schema` class is several questions in one request

Supersedes the "one `ask()` is one question" part of [ADR 0018](0018-typesafe-provider-is-decision-only.md)
(decision 1) and [ADR 0023](0023-openai-decisions-config-is-decision-only.md) (decision 1 and the
"Only one question per call" consequence). Everything else in those ADRs stands.

## Context

`TypeSafeConfig` and `OpenAIDecisionsConfig` sent one question, named `answer`, per call. Both APIs
are built for several independent questions over one shared input. A live check (2026-11-12, one
ticket, three questions) showed that one request gives the same answers as separate requests, using
fewer input tokens (Jev 450 vs 1068, OpenAI 438 vs 508). OpenAI can refuse one question and answer
the others.

## Decision

1. **The shape of `response_schema` decides; there is no new config or flag.** A scalar stays one
   question named `answer`. A flat class (dataclass, `TypedDict`, Pydantic model) is one question
   per field, named after the field (not its alias), all sent in one request.
2. **Every field is a required decision type with its own question** (`Question(...)`,
   `Field(description=...)` or the `Enum` docstring). The class itself takes no question; the agent
   prompt is what the fields share.
3. **A refusal of any field fails the whole reply.** A partial object would look like a complete one.
4. **Config option descriptions stay scalar-only.** `descriptions` / `criteria` are keyed by option
   value, which two fields may share; a class describes options per field with `Question(options=...)`.

## Consequences

- `metadata` is `{field: answer}` for a class, in each provider's raw shape. Normalising the two
  shapes is left to a separate change.
- Dependent questions still need separate `ask()` calls, as both providers advise.
- A class with a single field named `data` has the same JSON schema as `ResponseSchema`'s
  `{"data": ...}` envelope; `ResponseSchema` knows which one it built, but a raw
  `ResponseSchema.from_schema` object with a sole `data` property is read as the envelope.
