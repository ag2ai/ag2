---
name: ag2-telemetry
description: Add OpenTelemetry traces to an AG2 beta `Agent` via `TelemetryMiddleware` (`autogen.beta.middleware.builtin`). Emits spans for the full turn, each LLM call, each tool execution, and each human-input request, following the OpenTelemetry GenAI semantic conventions. Compatible with any OTLP backend — Jaeger, Grafana Tempo, Datadog, Honeycomb, Langfuse. Use when the user wants production-grade traces, latency analysis, token-usage attribution, or to ship telemetry into an existing observability stack.
license: Apache-2.0
---

# Telemetry — OpenTelemetry instrumentation

## When to use

The user wants to:

- See per-turn / per-call latency breakdowns
- Attribute token usage across operations
- Push traces to Jaeger, Grafana Tempo, Datadog, Honeycomb, Langfuse, etc.
- Debug a slow agent end-to-end with structured spans rather than print statements

If they just want quick stdout debugging, point them at `LoggingMiddleware` instead (see `ag2-middleware`).

## Installation

```bash
pip install "ag2[openai,tracing]"
```

## 60-second recipe

```python
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

from autogen.beta import Agent
from autogen.beta.config import OpenAIConfig
from autogen.beta.middleware.builtin import TelemetryMiddleware

# 1. Configure OpenTelemetry
resource = Resource.create({"service.name": "ag2-beta-quickstart"})
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(tracer_provider)

# 2. Wire the middleware
agent = Agent(
    "assistant",
    prompt="You are a helpful assistant.",
    config=OpenAIConfig(model="gpt-4o-mini"),
    middleware=[
        TelemetryMiddleware(
            tracer_provider=tracer_provider,
            agent_name="assistant",
        ),
    ],
)

# 3. Run — spans emit automatically
import asyncio
asyncio.run(agent.ask("What is the capital of France?"))
```

For production, swap `ConsoleSpanExporter` for `OTLPSpanExporter` (or your backend's exporter) and `SimpleSpanProcessor` for `BatchSpanProcessor`.

## Span hierarchy

Each `ask()` produces a root span with children:

```
invoke_agent assistant
  ├── chat gpt-4o-mini              # LLM API call
  ├── execute_tool get_weather      # tool execution
  ├── chat gpt-4o-mini              # LLM call after tool result
  └── await_human_input assistant   # human-in-the-loop
```

## Span types

Every span has an `ag2.span.type` attribute:

| `ag2.span.type` | Operation name | Hook |
|---|---|---|
| `agent` | `invoke_agent` | `on_turn` — full turn |
| `llm` | `chat` | `on_llm_call` — each LLM call |
| `tool` | `execute_tool` | `on_tool_execution` — each tool |
| `human_input` | `await_human_input` | `on_human_input` — HITL |

## Semantic attributes (GenAI semconv)

Spans carry standard [OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) attributes:

| Attribute | Spans | Description |
|---|---|---|
| `gen_ai.operation.name` | All | `invoke_agent` / `chat` / `execute_tool` / `await_human_input` |
| `gen_ai.agent.name` | agent, human_input | Agent name |
| `gen_ai.provider.name` | agent, llm | Auto-detected (`openai`, `anthropic`, …) |
| `gen_ai.request.model` | agent, llm | e.g. `gpt-4o-mini` |
| `gen_ai.response.model` | llm | Resolved from response |
| `gen_ai.response.finish_reasons` | llm | e.g. `["stop"]`, `["tool_calls"]` |
| `gen_ai.usage.input_tokens` | llm | Prompt tokens |
| `gen_ai.usage.output_tokens` | llm | Completion tokens |
| `gen_ai.usage.cache_creation_input_tokens` | llm | Prompt-cache writes (Anthropic) |
| `gen_ai.usage.cache_read_input_tokens` | llm | Prompt-cache reads (Anthropic, OpenAI, Gemini) |
| `gen_ai.tool.name` | tool | Tool function name |
| `gen_ai.tool.call.id` | tool | Tool call ID |
| `gen_ai.tool.type` | tool | Always `function` |

## Content capture (default ON)

By default, message content, tool args, and results are included on spans. Useful for debugging but can leak sensitive data:

```python
TelemetryMiddleware(
    tracer_provider=tracer_provider,
    agent_name="assistant",
    capture_content=False,   # omit messages, tool args, results
)
```

When enabled, additional attributes appear:

| Attribute | Span | Content |
|---|---|---|
| `gen_ai.input.messages` | llm | JSON request messages |
| `gen_ai.output.messages` | llm | JSON response messages |
| `gen_ai.tool.call.arguments` | tool | Tool args (JSON) |
| `gen_ai.tool.call.result` | tool | Tool result |
| `ag2.human_input.prompt` | human_input | Prompt shown to human |
| `ag2.human_input.response` | human_input | Human's response |

For privacy-sensitive backends (or anywhere telemetry leaves your infra), set `capture_content=False`.

## Constructor reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tracer_provider` | `TracerProvider \| None` | Global provider | OpenTelemetry TracerProvider |
| `capture_content` | `bool` | `True` | Include message/tool content in spans |
| `agent_name` | `str \| None` | `"unknown"` | Agent name for span attributes |
| `provider_name` | `str \| None` | `None` | Provider override (auto-detected if unset) |
| `model_name` | `str \| None` | `None` | Model override (auto-detected if unset) |

## Backend integration

`TelemetryMiddleware` uses standard OpenTelemetry, so any OTLP-compatible backend works:

- **Jaeger** — `OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")`
- **Grafana Tempo** — same OTLP exporter, point at the Tempo gateway
- **Langfuse, Honeycomb, Datadog** — vendor-specific exporters; the agent-side setup is identical

For container-orchestrated stacks, this repo includes a `tracing/` directory with Docker-Compose for otel-collector + Grafana Tempo.

## Going deeper

- `website/docs/beta/telemetry.mdx` — full attribute table, configuration, example.
- `tracing/` — Docker setup for local otel-collector + Tempo + Grafana.
- For sibling middleware (logging, retry, history limits), see `ag2-middleware`.

## Common pitfalls

- **`SimpleSpanProcessor` + `ConsoleSpanExporter` in production** — synchronous, blocks every span emit. Use `BatchSpanProcessor` and a real exporter (OTLP / Jaeger / vendor) outside of dev.
- **Leaking content into telemetry** — `capture_content=True` is the default. For privacy-sensitive prompts (PII, credentials), set `capture_content=False` *and* audit what your backend retains.
- **Forgetting `trace.set_tracer_provider(...)`** — without it, `tracer_provider` you pass to the middleware is fine, but third-party libraries that auto-instrument may use a different provider.
- **Token usage missing** — `gen_ai.usage.*` requires the provider client to surface usage in the response. Streaming providers may emit usage only at the end; if you don't see them, check the provider's response shape.
- **Span hierarchy doesn't show parent-child** — your exporter or backend may need the OTLP/HTTP path enabled, not just OTLP/gRPC. Check both.
- **Comparing to V1 tracing docs** — the semantic-attribute format is the same; only the agent instrumentation method differs (V1 uses `instrument_agent()` / `instrument_llm_wrapper()` / `instrument_pattern()`; beta uses `TelemetryMiddleware`).
