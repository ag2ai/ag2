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

Dropping the tool schemas is also not a portable "none". Anthropic and Bedrock reject
a request whose history holds tool-use blocks when no tools are defined, and dropping
them changes Anthropic's prompt-cache prefix.

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
  of being dropped silently. `OpenAIConfig(drop_tools_for_none=True)` sends `"none"` as no
  tools, for gateways that reject `tool_choice: "none"`.

## Consequences

- No middleware or `LLMClient` signature changes. A third-party client that does not
  read `context.tool_choice` still gets the filtered tools, but it ignores the mode.
- A per-call choice overrides a config-level `tool_choice` for that call.
- `replace(context, tool_choice=...)` would bypass narrowing. `with_tool_choice` is the
  documented way to set it.
