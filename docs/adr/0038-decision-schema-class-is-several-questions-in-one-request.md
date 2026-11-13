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
are built around several independent questions over one shared input: OpenAI accepts 1–400
questions and recommends batching independent ones; Jev takes a map of named questions of mixed
types. A live check (2026-11-12, one ticket, three questions) showed batching gives the same
answers as separate requests and uses fewer input tokens (Jev 450 vs 1068, OpenAI 438 vs 508).
OpenAI can refuse one question while answering the others.

Without batching, "is it spam / which department / how urgent" for one ticket needed three agents
and a hand-assembled result.

## Decision

1. **The shape of `response_schema` decides; there is no new config or flag.** A scalar (`bool`,
   string `Enum`, `IntEnum`, `0..1` number) stays one question named `answer`. A flat class (a
   dataclass, a `TypedDict` or a Pydantic model) becomes one question per field, named after the
   field, all sent in one request. `await reply.content()` is an instance of the class.
2. **Every field is required and a decision type.** Nested classes, `Optional`, lists, fields
   with defaults and `additionalProperties` raise `NotADecisionError` (the provider's subclass)
   before a request. Mutually exclusive options belong in one `Enum` field, not several `bool`s.
3. **Each field states its own question** through `Question(...)`, `Field(description=...)` or the
   `Enum` docstring, read with `get_type_hints(include_extras=True)` because the marker does not
   reach the JSON schema. A field without one is a local `ValueError`, because OpenAI requires
   instructions on every question and the prompt is shared by all of them. The agent prompt (and
   `question_instructions` on OpenAI) is prepended to each field's question.
4. **A refusal of any field fails the whole reply** with `DecisionRefusedError(field)`; a missing
   answer raises `ValueError` naming the field. A partial object would be indistinguishable from
   a complete one.
5. **`metadata` is `{field: answer}`** for a class, each entry in that provider's raw shape;
   unchanged for a scalar. Normalising the two providers' shapes is left to a separate change.
6. **Config option descriptions stay scalar-only.** `descriptions` (OpenAI) and `criteria`
   (TypeSafe) are keyed by option value, which two fields may share, so passing them with a class
   is a `ValueError`; options are described per field with `Question(options=...)`.
   `boolean_threshold` applies to every predicate field.
7. **The schema is still read once, provider-neutrally**, by `DecisionSet` in
   `ag2/response/decision.py`: a mapping of question name to `DecisionSpec`, with `render(values)`
   building the JSON the schema validates. The mappers only translate it.

## Consequences

- An object `response_schema` on a decision agent is no longer an error by itself; only the
  field rules above are.
- `DecisionSet` is public in `ag2.response`; existing scalar mapper functions keep their
  signatures, and the clients go through the new set-based ones.
- Dependent questions (one only meaningful after another's answer) still need separate `ask()`
  calls, as both providers advise.
- A class with a single field named `data` has the same JSON schema as `ResponseSchema`'s
  `{"data": ...}` envelope. `ResponseSchema` knows which one it built, so the class is still read
  as a class; only a raw `ResponseSchema.from_schema` object with a sole `data` property is read
  as the envelope.
- An empty class is rejected locally, since both APIs require at least one question.
