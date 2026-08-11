---
status: accepted
date: 2026-06-29
---

# History sizing and prompt rendering use event content, not `str(event)`

Surfaced from a report that a `max_tokens` compaction trigger "never fires" —
traced to history-management code reusing the truncated `__repr__` as a stand-in
for both token size and prompt content.

## Context

`BaseEvent.__repr__` renders each field through `truncate_repr`, which caps
`str`/`bytes` fields at 80 chars so logs stay readable. `str(event)` falls back
to that repr (there is no `__str__`). Five history-management call sites reused
`str(event)` for one of two things it was never meant to do:

- **Size estimation:** the compaction trigger (`agent.py`) and `TokenLimiter`
  middleware sized history as `len(str(event)) // chars_per_token`.
- **Prompt content:** `SummarizeCompact` and both aggregation strategies built
  their summarizer/memory prompts as `"\n".join(str(e) for e in events)`.

Both are misuses of a display helper. Measured with a 10,000-char payload:

| Event | real content | `len(str(e))` |
|---|---|---|
| `ModelRequest` (user) | 10,000 | 137 |
| `ToolResultsEvent` (tool) | 10,000 | 237 |
| `ModelResponse` (assistant) | 10,000 | 10,023 |

`ModelResponse` is the lone exception — its own `__repr__` emits full content.
So the estimate undercounts user/tool turns ~70× and is wildly asymmetric: a
`max_tokens` trigger effectively never fires on request/tool-heavy histories, and
the summarizer/memory LLM only ever sees the first 80 chars of each non-assistant
turn.

## Decision

**Size and render from event *content*, never from the repr. Two helpers, because
the two needs diverge on multimodal.** Both live in `ag2/events/sizing.py`.

- `estimated_tokens(event, chars_per_token=4) -> int` — text counts as
  `len // chars_per_token`; each non-text part counts as a flat per-modality
  budget (`_MODALITY_TOKENS`). Used by the compaction trigger and `TokenLimiter`.
- `render_for_prompt(event) -> str` — full, untruncated text, with non-text parts
  as `[modality]` placeholders and a role label. Used by the three
  summarizer/memory prompts.

They share one content walk (`_content_pieces`), so size and rendering can't
disagree about what an event contains.

**Multimodal is sized by a flat per-modality constant, deliberately.** A single
text rendering cannot serve both needs: an image carries ~0 text but real token
cost, so a text-derived size would count it as ~0 (the same bug), and a
placeholder like `[image]` is ~4 tokens though the image costs ~1,000. So
estimation is modality-aware and returns a number; rendering is text + placeholder.
The constant is a *heuristic* — image tokens track pixel dimensions, not bytes or
count, and the exact figure is provider-specific — so anything finer needs a real
tokenizer, which is out of scope (see below).

`TokenLimiter` is migrated from char math (`max_chars`) to token math
(`max_tokens` + `estimated_tokens`); its trimming semantics are unchanged, only
the per-event size is now content-based.

## Consequences / things that look wrong but are deliberate

- **`truncate_repr` / `__repr__` stay exactly as they are.** They are correct for
  their job (readable logs). The fix is to stop *other* code from borrowing them
  as a size/content proxy — not to change the repr.

- **Both helpers skip non-conversational events.** `estimated_tokens` returns 0
  and `render_for_prompt` returns `""` for telemetry (`UsageEvent` — see
  [0010](./0010-history-management-keys-on-conversational-not-transient.md)),
  since it never reaches the model. Putting the guard in the helpers — not just at
  each caller — means every budget calc (trigger, `TokenLimiter`) and every prompt
  is telemetry-free by construction, with no way for a new caller to forget it.

- **The size estimate is a coarse heuristic, and that is the right altitude.**
  Text is already "chars ÷ 4" — no real tokenizer is wired in anywhere — so
  precise provider-specific image tokenization would be inconsistent precision.
  The bug being fixed is that media counted as ~0; a non-zero flat budget fixes
  that. The principled long-term answer is a pluggable tokenizer seam (text +
  media); `TokenLimiter`'s docstring already gestured at "unless a custom
  tokenizer is provided," though none was ever wired up. That seam is a larger
  design, intentionally deferred.

- **`TokenLimiter`'s effective budget shifts.** It now sizes message *content*,
  not the repr (which included `ModelResponse(content=…)` scaffolding and
  truncated the payload). For the same `max_tokens` a caller may now retain a
  different number of turns — a behavior change, but the previous numbers were an
  artifact of repr length, not content. Its tests were recalibrated against
  `estimated_tokens`, not re-pinned to repr lengths.

- **`render_for_prompt` is intentionally text-only today.** Non-text parts become
  `[image]` / `[audio]` placeholders rather than being passed to the summarizer as
  real media. Feeding actual image/audio to a multimodal summarizer is an additive
  extension (the summarizer here is a plain `ModelConfig` call), deferred until
  there's a need.
