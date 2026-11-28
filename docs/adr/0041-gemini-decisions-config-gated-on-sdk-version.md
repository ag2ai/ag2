---
status: accepted
date: 2026-11-28
---

# 0041. `GeminiDecisionsConfig` follows ADR 0023 and is gated on `google-genai>=3.2`

## Context

Gemini's Decisions API (`POST /v1beta/decisions`, model `gemini-decision-1`) answers
the same three kinds of typed questions as OpenAI's: yes/no (`noul`), choice, and
score. It is only on the Gemini Developer API, not Vertex AI.

The SDK gained `client.aio.decisions` in `google-genai` 3.2. The `ag2[gemini]` extra
still allows `>=1.68.0,<4`, because 2.x stays supported until 1.4.0
(`ag2/config/gemini/_compat.py`). Raising the floor to 3.2 would drop 2.x early for
every Gemini user, just to add one new config.

## Decision

1. **Same contract as `OpenAIDecisionsConfig`.** `GeminiDecisionsConfig` is a
   separate, decision-only config. It reads the `response_schema` through the shared
   `DecisionSet` / `DecisionSpec` and only translates it. Where the APIs differ,
   follow Gemini:
   - Questions are keyed by name.
   - Score levels are a list of rubric strings, `"Label: description"`.
   - A yes/no question takes `"true"` / `"false"` outcome descriptions.
   - Input is one flat list of content blocks, with history tagged as text.
   - Images, audio, video and documents are accepted.
   - A filter block (`prompt_feedback.block_reason`) raises `DecisionBlockedError`,
     because Gemini has no per-question refusal.
2. **Gated on the SDK version, not the extra's floor.** The config's modules import
   `google.genai.decisions` at the top. `ag2/config/gemini/__init__.py` imports them
   inside `try/except ImportError`. On an older SDK, `GeminiDecisionsConfig` becomes a
   `missing_additional_dependency` stand-in that tells the user to
   `pip install "google-genai>=3.2,<4"`. The errors live in `decisions_errors.py`,
   which does not import the SDK, so callers can `except` them on any version.
3. **`api_version` defaults to `v1beta`.** The SDK's default `v1` returns 404 for
   this endpoint (checked against the live API on 2026-11-28).

## Consequences

- On `google-genai` 2.x or 3.0–3.1, `GeminiConfig` and `VertexAIConfig` keep working.
  Only `GeminiDecisionsConfig` raises, and only when it is used.
- The decisions tests `importorskip("google.genai.decisions")`, so the
  `test-gemini-2x` CI job skips them.
- When 1.4.0 raises the floor to `google-genai>=3`, set it to `>=3.2` and drop the
  guard.
- When Google moves the endpoint to `v1`, change the `api_version` default.
