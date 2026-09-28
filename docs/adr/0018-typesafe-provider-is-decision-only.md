---
status: accepted
date: 2026-09-23
---

# 0018. The TypeSafe provider is decision-only: the response schema is the question

## Context

TypeSafe AI's Jev model does not generate text. Its System One API takes a
`state` (text or JSON) and a map of named **questions**, each one of three
primitives, and returns one **answer** per question with a confidence and the
probability of every option:

| primitive | asks | answers with |
| :--- | :--- | :--- |
| noul | yes or no | a yes-probability in `0..1` |
| choice | one label out of several | the label, plus a probability per label |
| score | a level on an ordered rubric | the expected value on `0..n-1`, plus a probability per level |

Every other provider in `ag2/config/*` turns messages into a completion and
turns tool declarations into a tool list. Jev has neither: there is no
completion to stream, no tool to call, and nothing to say unless a question was
asked. Fitting it behind `LLMClient` therefore needed a rule for where the
question comes from and what the agent loop is told about the parts of the
protocol Jev cannot serve.

## Decision

### 1. One `ask()` is one question, and `response_schema` is that question

`TypeSafeClient` sends exactly one question per call, under the fixed name
`answer`, and derives it from the agent's `response_schema` JSON schema:

- `boolean`, or `number` with `minimum: 0` and `maximum: 1` → noul
- `enum` of strings → choice
- `enum` of integers → score, provided the levels are exactly `0..n-1`,
  `2 <= n <= 10`, and every level is described

An agent with no `response_schema`, or with any other shape (`str`, `int`, a
dataclass, a Pydantic model, a `PromptedSchema`), raises
`UnsupportedResponseSchemaError` before a request is sent. `ResponseSchema`'s
`{"data": ...}` envelope is unwrapped for the decision and re-applied to the
answer, so the usual `await reply.content()` validation path is unchanged.

The alternative — inventing a text mode by asking a choice over the model's
own words — was rejected as pretending to be something the API is not.

### 2. The prompt frames the question; the schema's description asks it

`context.prompt` becomes the leading part of the question's `instructions`,
followed by the schema's own description: an `Enum`'s class docstring, or an
explicit `ResponseSchema(..., description=...)`. `ResponseSchema` falls back to
the type's docstring for its `description`, so `bool.__doc__` would otherwise
be sent as a question; the mapper ignores a description equal to the type's
docstring.

A noul with no instructions and no criteria is rejected by the API. The client
raises `ValueError` locally with the three ways out (prompt, description,
`criteria`) rather than surfacing an HTTP 4xx.

### 3. Option descriptions are read from `Enum` member docstrings in the source

The string literal under an `Enum` member (`BILLING = "billing"` followed by
`"""Payments, invoicing, refunds."""`) is the natural place to describe that
option, and it is what PEP 257 calls an attribute docstring. Python discards
it at runtime, so `_member_docstrings` parses the class source with `ast` and
maps each member's value to the literal below its assignment. The result is
cached per type.

Where the source is unavailable (REPL, notebook, frozen app) the options are
simply undescribed. `TypeSafeConfig.criteria` overrides or supplies
descriptions without touching the type, keyed by choice label, by score level
as a string, or by `"true"` / `"false"` for a noul.

This is the reason `ResponseSchema` now keeps the `types` it was built from:
the mapper needs the `Enum` class itself, and the JSON schema no longer carries
it.

### 4. Tools are rejected at the client, not at request time

`tool_to_api` raises `UnsupportedToolError` for every `ToolSchema`, so an
agent that lists a tool fails on its first call with the provider named, rather
than sending tools Jev would ignore. Tools belong on the generative agent that
a Jev agent routes to; the subtask and `as_tool` machinery composes the two.

### 5. Answers are normalised to the schema, and the raw answer is kept

- A `bool` schema is `True` when the yes-probability is at or above
  `boolean_threshold` (default `0.5`); a probability schema gets the raw value.
- A score is the expected value on the rubric, which is rarely an integer, so
  it is rounded to the nearest level and clamped to the rubric.
- The whole SDK answer minus its `type` (`confidence`, `probabilities`,
  `choice` / `noul` / `score`, `legend`) is kept on `ModelMessage.metadata`,
  reachable as `reply.response.metadata`, because the distribution is usually
  the reason to use a decision model at all.

### 6. Non-streaming only

The API has no streaming form, so `TypeSafeConfig` has no `streaming` field and
the client emits a single `ModelMessage`. The docs' streaming-first tip does not
apply to this provider.

## Consequences

- **A `TypeSafeConfig` agent without a decision `response_schema` is a
  configuration error**, raised locally on the first `ask()`. This is the
  opposite of every other provider, where `response_schema` is optional.
- **`reply.body` is JSON, not prose.** It is whatever `answer_to_content`
  rendered for the schema (`{"data": "technical"}`), and only exists so the
  standard validation path can parse it.
- **Member docstrings are a source-level feature.** A type defined where
  `inspect.getsource` fails is silently undescribed for a choice and rejected
  for a score. `criteria` is the documented fallback.
- **`Literal[...]` is only reachable through `ResponseSchema.from_schema`.**
  `make_adapter` does not accept a bare `Literal` or `Annotated` type today, so
  the docs advertise `Enum` for choices and `from_schema` for a probability;
  the mapper itself keys on the JSON schema and needs no change if that lands.
- **`ResponseSchema.types` is now public state.** It was added for this mapper
  and is a one-line change to core; anything else that needs the original type
  back from a schema can use it too.
- **Mocked unit tests carry no `typesafe` mark.** Like every other
  `test/config/<provider>` suite they run in the default `just test`; the mark
  is reserved for `test/providers/typesafe`, which needs `TYPESAFE_API_KEY` and
  is excluded by `_llm_filter`.
