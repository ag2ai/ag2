---
status: accepted
date: 2026-10-09
---

# 0023. `OpenAIDecisionsConfig` is decision-only and follows the TypeSafe mapping

## Context

OpenAI's [Decisions API](https://developers.openai.com/api/docs/guides/decisions)
(`POST /v1/decisions`, model `gpt-6-luna`, public beta) answers typed questions
instead of generating text. A request carries shared `input` (text and inline
images, user role only) and a list of named `questions`, each one of:

| type | answers with |
| :--- | :--- |
| `predicate` | `probability` that a condition is true |
| `choice` | one supplied `value`, plus `probabilities` and `confidence` |
| `score` | the probability-weighted mean of 0-based level indices, plus `probabilities` and `confidence` |

Any question may instead come back as a `refusal`. There are no tools, no
streaming and no free text. The shape is the same as TypeSafe's Jev, already
integrated under [ADR 0018](0018-typesafe-provider-is-decision-only.md).

## Decision

### 1. Reuse ADR 0018's contract, not invent a second one

`OpenAIDecisionsConfig` is a separate config next to `OpenAIConfig` and
`OpenAIResponsesConfig`, with the same rules as `TypeSafeConfig`: one `ask()` is
one question named `answer`, and the agent's `response_schema` is that question:

- `bool`, or a `number` bounded to `0..1` → `predicate`
- an `Enum` of strings → `choice`
- an `IntEnum` numbered `0..n-1` → `score`

Anything else raises `UnsupportedResponseSchemaError` before a request. The
prompt and the schema description join into the question's `instructions`;
member docstrings describe options; `descriptions` on the config overrides them.
Answers are rendered back through the `ResponseSchema` envelope, and the raw
answer (minus `type` and `name`) lands on `reply.response.metadata`.

The schema-reading helpers (`decision_node`, `member_docstrings`, …) moved from
`ag2/config/typesafe/mappers.py` into the provider-neutral
`ag2/config/decision_schema.py` so both mappers share them without the OpenAI
extra depending on `typesafe-sdk`.

### 2. Where the APIs differ, follow OpenAI

- **Score levels are labelled, descriptions optional.** The API requires a
  `label` per level; it is taken from the `IntEnum` member name
  (`WORKAROUND_AVAILABLE` → `Workaround available`). Unlike Jev, a level
  without a docstring is accepted.
- **Every question needs `instructions`,** including choices and scores, so an
  empty prompt plus an undescribed schema raises `ValueError` locally.
- **No `true` / `false` criteria.** A predicate has no outcome descriptions, so
  the config field is `descriptions` (OpenAI's word), not `criteria`.
- **Refusal is an error.** A `refusal` answer raises `DecisionRefusedError`,
  because no value of the schema represents "declined", and a silent default
  would be indistinguishable from a real answer.

### 3. History is flattened into user messages

The endpoint accepts only `user` messages. Earlier assistant turns, tool
results and compaction summaries are sent as user messages tagged
`[Assistant]`, `[Tool result]`, `[Summary of earlier conversation]`, so a
decision agent can sit on a shared stream without losing context.

### 4. Images only inline; no files

`BinaryInput` images become data URLs (honouring `vendor_metadata["detail"]`),
and a `data:` `UrlInput` passes through. Hosted URLs, `file_id`s and non-image
binaries raise `UnsupportedInputError`; `create_files_client()` raises
`NotImplementedError`.

### 5. The SDK floor moves to `openai>=3.27.0`

The provider requires `openai>=3.27.0` for `client.decisions.create` and the
typed `Decision` models. Calling the endpoint with `client.post` would avoid
the bump, but would lose those models;
the floor is raised for every `ag2[openai]` user instead.

## Consequences

- **An agent on `OpenAIDecisionsConfig` without a decision `response_schema`
  is a configuration error**, as with `TypeSafeConfig`.
- **Only one question per call.** The API can batch independent questions in
  one request; ag2 does not expose that, since one `ask()` returns one typed
  value. Several decision agents over the same input cost several requests.
- **A refusal aborts the turn.** Callers that expect refusals catch
  `DecisionRefusedError`.
- **Changes to the shared schema reading affect both providers.**
  `test/config/typesafe` and `test/config/openai/test_decisions_mappers.py`
  cover it from both sides.
- **Live tests reuse the `openai` mark** under `test/providers/openai`, so no
  justfile or CI matrix change is needed.
