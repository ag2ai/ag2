---
status: accepted
date: 2026-10-22
---

# 0025. A per-call tool choice rides on the context, and each client maps it

## Context

A middleware sometimes needs one LLM call made with fewer tools, or none. For example,
a budget guard's final "answer from what you have" call should not be able to come back
as tool calls only. `on_llm_call` can change the events it passes to `call_next`, but
the agent binds the tools into the innermost call (`partial(client, tools=...)`).
`tool_choice` exists only as a whole-agent config field on some providers, and each
of them types it differently.

The obvious shape is extra keyword arguments on `call_next`. But `call_next` is
`partial(next_middleware.on_llm_call, ...)`, and every existing `on_llm_call` takes
`(call_next, events, context)` only. The first middleware that has not been updated,
including built-ins like `TelemetryMiddleware`, would raise `TypeError`, and widening
`LLMCall` breaks every third-party middleware.

Dropping the tool schemas is also not a portable "none". Bedrock's Converse API
requires `toolConfig` when the messages hold tool-use blocks. Elsewhere, dropping the
schemas changes the request's cached prefix and the model's view of its own tool
history. Keeping them is not free either: measured on Claude (2026-10-22), a model that
wanted a tool can answer `tool_choice: "none"` with no content at all. Claude Haiku 4.5
in a "stop and answer now" synthesis call gave 0 of 6 text replies with the tools kept,
and 2 of 6 with them dropped (5 of 6 when the instruction shared the tool-result
message). Claude Sonnet 4.6 and gpt-5.4-nano answered 8 of 8 either way in the same flow.

## Decision

- `ToolChoice` (`ag2/tool_choice.py`, exported from `ag2.middleware`) is a provider-neutral
  value: a mode, `"auto"`, `"none"` or `"required"`, plus an optional allow-list of tool
  names (`ToolChoice.only(...)`).
- It is carried on `ConversationContext.tool_choice`. A middleware passes
  `context.with_tool_choice(...)` to `call_next`. The copy flows through the inner
  middlewares unchanged and applies to that one call only, the way
  `replace(context, tool_call_id=...)` already scopes a context to one tool call.
- `with_tool_choice` narrows and never widens. Allow-lists intersect, and the stricter
  mode wins (`"none"` over `"required"` over `"auto"`). A choice set further out
  therefore holds whatever an inner middleware asks for.
- The agent's innermost `LLMCall` applies the allow-list for every client. The mode is
  left to each client: Anthropic, OpenAI, Gemini, Mistral, xAI and DashScope keep the
  schemas and send their native setting. Ollama and Z.ai, which have none, express
  `"none"` by leaving the tools out. A mode a provider cannot honour (Bedrock `"none"`,
  `"required"` on Ollama, Z.ai and DashScope) raises `UnsupportedToolChoiceError` instead
  of being dropped silently. `OpenAIConfig(drop_tools_for_none=True)` and
  `AnthropicConfig(drop_tools_for_none=True)` send `"none"` as no tools instead: for a
  gateway that rejects `tool_choice: "none"`, and for a model that goes silent under it.
- The choice is enforced on the reply too. A tool call that the call's choice did not
  allow is dropped with a warning rather than executed: the agent's tools all stay
  registered, and a model can name a tool it was not offered (seen on Claude with an
  allow-list), or a gateway can ignore `"none"`.

## Consequences

- No middleware or `LLMClient` signature changes. A third-party client that does not
  read `context.tool_choice` still gets the filtered tools, but it ignores the mode;
  the reply-side check still keeps a forbidden call from executing.
- `"none"` guarantees no tool runs, not that the model answers. A reply can come back
  empty, so a caller that needs an answer still has to handle that case.
- A per-call choice overrides a config-level `tool_choice` for that call.
- `replace(context, tool_choice=...)` would bypass narrowing. `with_tool_choice` is the
  documented way to set it.
