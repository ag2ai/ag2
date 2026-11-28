---
status: accepted
date: 2026-11-28
---

# 0041. `GeminiDecisionsConfig` follows ADR 0023; `ag2[gemini]` requires `google-genai>=3.2`

## Context

Gemini's Decisions API (`POST /v1beta/decisions`, model `gemini-decision-1`) answers
the same three kinds of typed questions as OpenAI's: yes/no (`noul`), choice, and
score. It is only on the Gemini Developer API, not Vertex AI.

The SDK gained `client.aio.decisions` in `google-genai` 3.2. Until now the
`ag2[gemini]` extra allowed `>=1.68.0,<4`, and `google-genai` 2.x was deprecated, to be
removed in 1.4.0.

Two options were weighed:

- **Keep 2.x and guard the import.** `GeminiDecisionsConfig` would become a
  missing-dependency stand-in on older SDKs.
- **Raise the floor.** This is what ADR 0023 §5 did for OpenAI: it moved `ag2[openai]`
  to the SDK version that had the Decisions types.

## Decision

1. **`ag2[gemini]` requires `google-genai>=3.2,<4` now, ahead of 1.4.0.** This
   follows the OpenAI precedent. A config that installs with the extra but fails on
   use would be surprising, and keeping 2.x for a few more releases was not worth the
   guard. The rest of the 2.x removal planned for 1.4.0 goes with it:
   - the `SDK_3X` branches, `warn_if_sdk_2x` and `HTTPX2_NEEDS_SDK_3X` in `_compat.py`
   - the 2.x-only tests
   - the `test-gemini-2x` CI job
   - the 2.x notes in the docs
2. **Same contract as `OpenAIDecisionsConfig`.** `GeminiDecisionsConfig` is a
   separate, decision-only config. It reads the `response_schema` through the shared
   `DecisionSet` / `DecisionSpec` and only translates it. Where the APIs differ,
   follow Gemini:
   - Questions are keyed by name.
   - Score levels are a list of rubric strings, `"Label: description"`.
   - Input is one flat list of content blocks. History is tagged as text, and adjacent
     text blocks are separated, because the API joins them with nothing in between.
   - Images, audio, video and documents are accepted inline, by HTTPS URI, or as a
     Files API upload (`FileIdInput`). The MIME type for a URI or upload comes from its
     file extension, using the types the API lists for that block kind.
   - The API's limits are checked before the request: 1 to 1,000 questions, keys
     matching `[A-Za-z0-9_-]{1,64}`, 2 to 255 options, and valid option IDs.
   - A blocked request (HTTP 400 with `error.code` `safety`, `prohibited_content`,
     `blocklist`, `image_safety` or `content_blocked`) raises `DecisionBlockedError`,
     because Gemini has no per-question refusal.
   - A choice outside the schema raises instead of being returned.
3. **`api_version` defaults to `v1beta`.** The SDK's default `v1` returns 404 for
   this endpoint (checked against the live API on 2026-11-28).

## Consequences

- Users on `google-genai` 2.x must upgrade to `>=3.2` when they upgrade ag2. The
  1.4.0 removal note for 2.x is gone from `AGENTS.md`.
- `GeminiConfig`'s Developer API default is `v1`, as it already was on 3.x.
- When Google moves the endpoint to `v1`, change the `api_version` default.
