---
name: ag2-middleware
description: Intercept the AG2 beta agent loop with `BaseMiddleware` — wrap full turns (`on_turn`), each LLM call (`on_llm_call`), each tool execution (`on_tool_execution`), or each human-input request (`on_human_input`). Use for retry, logging, history trimming, request mutation, tool auditing, guardrails, or rate limiting. Built-ins: `LoggingMiddleware`, `RetryMiddleware`, `HistoryLimiter`, `TokenLimiter`, `TelemetryMiddleware` (see `ag2-telemetry`). For per-tool hooks see also `ag2-add-custom-tool` tool-middleware section.
license: Apache-2.0
---

# Middleware

## When to use

Middleware is for **cross-cutting behaviour** that should apply consistently across many runs without changing the agent, model client, or tools themselves. Common use cases:

- Logging, tracing, timing
- Retry on transient failures
- Trim history before it reaches the model
- Cap or estimate token usage
- Rewrite tool arguments / results
- Enforce policies before a tool runs
- Audit human-input requests

## Four hooks

`BaseMiddleware` exposes four async hooks. Implement only the ones you need:

| Hook | Wraps | Use for |
|---|---|---|
| `on_turn(call_next, event, context) → ModelResponse` | The whole agent turn | Total latency, request/response inspection, turn-level policies |
| `on_llm_call(call_next, events, context) → ModelResponse` | Each LLM API call | Retry, logging, history trim, request mutation, caching |
| `on_tool_execution(call_next, event, context) → ToolResultType` | Each tool invocation | Validate args, redact results, fallback on failure, access control |
| `on_human_input(call_next, event, context) → HumanMessage` | Each `context.input()` | Audit, rewrite prompts, automated short-circuit, rate limit |

Each instance is created **once per turn** and can hold per-turn state on `self`. The same instance can implement multiple hooks.

## Built-in middleware

Importable from `autogen.beta.middleware`:

| Middleware | Purpose | Constructor |
|---|---|---|
| `LoggingMiddleware` | Logs turn start/end, each LLM call, each tool execution | no args |
| `RetryMiddleware` | Retries failed LLM calls | `max_retries=N`, `retry_on=ExceptionClass` |
| `HistoryLimiter` | Cap event count before LLM call | `max_events=N` |
| `TokenLimiter` | Char-based token-budget cap before LLM call | `max_tokens=N`, `chars_per_token=4` |
| `TelemetryMiddleware` | OpenTelemetry GenAI spans (see `ag2-telemetry`) | see telemetry skill |

## Registration — agent-level

Apply to every turn:

```python
from autogen.beta import Agent
from autogen.beta.config import OpenAIConfig
from autogen.beta.middleware import LoggingMiddleware, RetryMiddleware

agent = Agent(
    "assistant",
    config=OpenAIConfig(model="gpt-4o-mini"),
    middleware=[
        LoggingMiddleware(),
        RetryMiddleware(max_retries=2),
    ],
)
```

## Registration — call-level

Add temporary middleware for one turn. Both `agent.ask(...)` and `reply.ask(...)` accept it:

```python
from autogen.beta.middleware import TokenLimiter

reply = await agent.ask("Summarise the latest messages.", middleware=[LoggingMiddleware()])
next_turn = await reply.ask("Now answer in one paragraph.", middleware=[TokenLimiter(max_tokens=4000)])
```

Call-level middleware is **appended after** the agent's middleware list.

## Ordering

Middleware runs in registration order, like nested `with` blocks. Registering `[A, B, C]` enters `A → B → C` and unwinds `C → B → A`:

```
enter A
  enter B
    enter C
      <LLM call>
    exit C
  exit B
exit A
```

This matters when you mix logging, mutation, retry. If `RetryMiddleware` should retry mutated requests, mutation goes **inside** retry; if you want each retry attempt logged separately, logging goes **inside** retry.

## Writing your own

Subclass `BaseMiddleware`, implement the hooks you need:

```python
import logging
from collections.abc import Sequence
from autogen.beta import Agent, Context
from autogen.beta.config import OpenAIConfig
from autogen.beta.events import BaseEvent, ModelResponse, ToolCallEvent
from autogen.beta.middleware import BaseMiddleware, LLMCall, Middleware, ToolExecution

class AuditMiddleware(BaseMiddleware):
    def __init__(self, event: BaseEvent, context: Context, logger: logging.Logger) -> None:
        super().__init__(event, context)
        self.logger = logger

    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        self.logger.info("Calling model with %d events", len(events))
        response = await call_next(events, context)
        self.logger.info("Model returned: %s", response)
        return response

    async def on_tool_execution(self, call_next: ToolExecution, event: ToolCallEvent, context: Context):
        self.logger.info("Executing tool: %s", event.name)
        return await call_next(event, context)

agent = Agent(
    "assistant",
    config=OpenAIConfig(model="gpt-4o-mini"),
    middleware=[
        Middleware(AuditMiddleware, logger=logging.getLogger("ag2.audit")),
    ],
)
```

If your middleware needs constructor args beyond `event` and `context`, **wrap with `Middleware(YourClass, ...)`** when registering. Zero-config middleware can be passed bare (`middleware=[LoggingMiddleware()]`).

## Tool-scoped vs agent-scoped

For behaviour that applies to **one tool only** (validation, redaction for that tool's output, approval gates), use **tool middleware** instead — `middleware=[hook]` on `@tool`, `@agent.tool`, or `Toolkit`. See `ag2-add-custom-tool` for the syntax. The `approval_required()` built-in (see `ag2-hitl`) is a tool middleware.

Agent middleware runs **outside** tool middleware: `BaseMiddleware.on_tool_execution()` sees the full execution including tool-scoped hooks.

## Picking the right hook

- `on_turn` → behaviour about the whole request/response lifecycle.
- `on_llm_call` → behaviour about what goes into / comes out of the model.
- `on_tool_execution` → tool safety / auditing / result shaping across many tools.
- Tool-scoped middleware (not `BaseMiddleware`) → behaviour for a single tool's definition.
- `on_human_input` → intercept HITL requests/responses.

## Going deeper

- `references/builtin_middleware.md` — every built-in's params, common-case recipes, when each fits.
- `website/docs/beta/middleware.mdx` — full reference, ordering examples, custom-middleware guidelines.
- `website/docs/beta/tools/tool_middleware.mdx` — per-tool hooks (different mental model — plain async callables, not `BaseMiddleware`).
- For OpenTelemetry instrumentation specifically, see `ag2-telemetry`.

## Common pitfalls

- **Forgetting `Middleware(...)` for constructor args** — `middleware=[AuditMiddleware]` (no wrapper) only works if the class needs only `event` and `context`. Otherwise wrap: `middleware=[Middleware(AuditMiddleware, logger=...)]`.
- **Mutation order surprises** — middleware runs in registration order. If middleware A trims history and middleware B logs it, register `[A, B]` so B sees the trimmed view.
- **Per-call middleware doesn't replace agent middleware** — it's appended. Agent middleware still runs.
- **One big middleware doing five things** — keep hooks focused. Logging + retry + mutation + policy in one class is hard to reason about and order. Split into multiple instances.
- **`on_tool_execution` branching on `event.name` for a single tool** — that's a smell; use tool-scoped middleware for one-tool behaviour and reserve `on_tool_execution` for cross-cutting policies.
- **Putting OpenTelemetry instrumentation in custom code** — there's a `TelemetryMiddleware` for that; see `ag2-telemetry`.
