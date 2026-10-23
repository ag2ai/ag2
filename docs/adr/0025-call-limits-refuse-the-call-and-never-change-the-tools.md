---
status: accepted
date: 2026-10-23
---

# 0025. Call limits refuse the call and never change the tools

## Context

An agent with a budget has to stop calling tools at some point and answer. The
obvious way to stop it is to change the request for the final calls: send fewer
tools, no tools, or a `tool_choice` of `"none"`. AG2 built that as a per-call tool
choice carried on the context (draft PR #3468) and dropped it before merging:

- Changing the request breaks the provider's prompt cache. On Anthropic a change to
  `tool_choice` stops the cached conversation from being reused, and a narrower
  tool list changes the cached prefix itself.
- The model still sees its own tool calls in the history, and calls again. Claude
  Haiku 4.5 named a tool it was not offered. Under `"none"` it answered with no
  content in 6 of 6 tries, and Gemini 3.8 Flash with a malformed function call.
  MiniMax-M3 ignored `"required"` and named tools entirely. A call dropped silently
  leaves an empty reply, and the model never learns why.

A downstream budget middleware built on refusing the call, rather than changing the
request, had run in production since October 2026.

## Decision

- `CallLimiter` (`ag2.middleware`) refuses an over-limit tool call with a structured
  `call_limit_exceeded` result instead of running it. The tools, their schemas and
  `tool_choice` are never changed.
- A single tool's cap refuses that tool only and the turn goes on. A used-up total
  (tool calls, model calls or a soft clock) exhausts the turn: the model gets a final
  call asking it to answer, and at most one more for text, and then the turn ends
  with no model call.
  Exempt tools are never counted or refused.
- Model-call limits are checked before the call is sent, so a model that ignores the
  refusals cannot spend past the limit.
- A used-up tool total is also checked before the next model call. Exempt tools
  remain executable, but cannot keep an exhausted turn alive indefinitely.
- Counts live in an injectable `CallCounter`. The default is one per agent and per
  turn. A caller shares one, for example through `context.dependencies`, for limits
  across a run.
- Stacked limiters coordinate their policies for the duration of a turn. They check
  all tool limits before reserving a call, and count each inner synthesis request
  against the outer budget. Multiple policies sharing a counter reserve it once.
  All total limits are checked before per-tool caps, so a per-tool refusal cannot
  hide another policy's exhausted total.
- Optional remaining-budget notes are appended to history and replayed unchanged.
  A temporary note would move Anthropic's message cache breakpoint onto content
  missing from the next request, preventing reuse of that conversation cache.
- Exhaustion is announced when the turn actually finishes, including when an
  exempt tool returns a final result without another model call.
  A final tool completing a batch with a total refusal has ending `"final_tool"`;
  metadata is attached before the final response is published, regardless of
  which parallel tool settles first.
- Discarded synthesis responses emit their own `UsageEvent`; the agent reports
  usage for the returned response, preserving accounting without duplication.
- A discarded response also emits a durable `ModelResponseDiscarded` marker.
  Provider replay must discard its preceding native turn state too. For a kept
  response, xAI's mapper retains only its accepted client calls in the native
  turn, preserving server-side calls and opaque state. Otherwise a discarded
  call would reappear in the next request without a tool result.
- `RetryMiddleware` must wrap all limiters so every provider attempt is charged.
  The reverse order is rejected. Input budget notes do not disarm retries.
- A limiter cannot be wrapped in `ConditionalMiddleware`: per-hook gates cannot
  preserve the coordinated turn policy. This combination is rejected explicitly.
- The `last_text` fallback observes responses during the turn. A request drained
  from enqueued input can continue the same turn, so request types in history
  cannot reliably identify its boundary.

## Consequences

- Each refused call costs a model call, because the model has to read the refusal.
- A turn the limiter ends can come back empty. It is marked
  `finish_reason="call_limit"` and announced by a `CallLimitReached` event, so a
  caller can tell it from a model that chose to say nothing.
- There is no per-call `tool_choice` in AG2. A middleware that needs one changes
  the request itself and pays the cache cost knowingly.
