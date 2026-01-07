# AG2 Tracing & Telemetry

This directory contains the development playground for AG2's OpenTelemetry instrumentation. The production implementation lives in `autogen/instrumentation.py` and `autogen/tracing/utils.py`.

## Architecture Overview

AG2 tracing uses **OpenTelemetry** to provide distributed tracing of multi-agent conversations. The approach follows the [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) with AG2-specific extensions.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Trace Hierarchy                          │
├─────────────────────────────────────────────────────────────────┤
│  initiate_chats (multi-chat workflow)                           │
│    └── conversation (initiate_chat / a_initiate_chat)           │
│          ├── invoke_agent (generate_reply / a_generate_reply)   │
│          │     ├── chat (LLM API call)                          │
│          │     ├── execute_tool (execute_function)              │
│          │     ├── execute_code (code execution)                │
│          │     └── speaker_selection (a_auto_select_speaker)    │
│          │           └── invoke_agent (internal speaker)        │
│          │                 └── chat (LLM API call)              │
│          └── await_human_input (get_human_input)                │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Setup (Tracer Provider)
Create and configure the OpenTelemetry tracer provider with OTLP exporter:

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

resource = Resource.create(attributes={"service.name": "my-agent-service"})
tracer_provider = TracerProvider(resource=resource)
exporter = OTLPSpanExporter(endpoint="http://127.0.0.1:4317")  # OTLP gRPC endpoint
processor = BatchSpanProcessor(exporter)
tracer_provider.add_span_processor(processor)
trace.set_tracer_provider(tracer_provider)
```

### 2. Agent Instrumentation (`instrument_agent`)
Wraps agent methods to emit spans:

- `generate_reply` / `a_generate_reply` → `invoke_agent` span
- `initiate_chat` / `a_initiate_chat` → `conversation` span
- `execute_function` / `a_execute_function` → `execute_tool` span
- `a_generate_remote_reply` → `invoke_agent` span (with trace propagation)
- `get_human_input` / `a_get_human_input` → `await_human_input` span (captures HITL wait time)
- `_generate_code_execution_reply_using_executor` → `execute_code` span

```python
from autogen.instrumentation import instrument_agent

instrument_agent(my_agent, tracer_provider=tracer_provider)
```

### 3. LLM Instrumentation (`instrument_llm_wrapper`)
Instruments `OpenAIWrapper.create()` to emit `chat` spans for each LLM API call:

```python
from autogen.instrumentation import instrument_llm_wrapper

instrument_llm_wrapper(tracer_provider=tracer_provider)

# Or with message capture enabled (for debugging)
instrument_llm_wrapper(tracer_provider=tracer_provider, capture_messages=True)
```

This captures:
- Provider name (`gen_ai.provider.name`)
- Model name (`gen_ai.request.model`, `gen_ai.response.model`)
- Token usage (`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`)
- Request parameters (temperature, max_tokens, etc.)
- Finish reasons and cost

### 4. Pattern Instrumentation (`instrument_pattern`)
For group chats, instruments the pattern which auto-instruments all agents and the GroupChatManager:

```python
from autogen.instrumentation import instrument_pattern

instrument_pattern(pattern, tracer_provider=tracer_provider)
```

### 5. Multi-Chat Instrumentation (`instrument_chats`)
For sequential/parallel multi-chat workflows using `initiate_chats` or `a_initiate_chats`:

```python
from autogen.instrumentation import instrument_chats

instrument_chats(tracer_provider=tracer_provider)

# Now initiate_chats calls are traced with a parent span
results = agent.initiate_chats([
    {"recipient": agent1, "message": "...", "max_turns": 1},
    {"recipient": agent2, "message": "...", "max_turns": 1},
])
```

This creates a parent `initiate_chats` span that groups all child conversation spans together.

### 6. A2A Server Instrumentation (`instrument_a2a_server`)
For remote agents using A2A protocol, adds middleware to extract trace context from incoming requests:

```python
from autogen.instrumentation import instrument_a2a_server

instrument_a2a_server(server, tracer_provider=tracer_provider)
```

## Span Types

| Span Type | Operation | Instrumented Method |
|-----------|-----------|---------------------|
| `multi_conversation` | `initiate_chats` | `initiate_chats`, `a_initiate_chats` |
| `conversation` | `conversation` | `initiate_chat`, `a_initiate_chat`, `run_chat`, `a_run_chat`, `a_resume` |
| `agent` | `invoke_agent` | `generate_reply`, `a_generate_reply`, `a_generate_remote_reply` |
| `llm` | `chat` | `OpenAIWrapper.create` |
| `tool` | `execute_tool` | `execute_function`, `a_execute_function` |
| `speaker_selection` | `speaker_selection` | `a_auto_select_speaker`, `_auto_select_speaker` |
| `human_input` | `await_human_input` | `get_human_input`, `a_get_human_input` |
| `code_execution` | `execute_code` | `_generate_code_execution_reply_using_executor` |

**Note**: The `run()` and `a_run()` methods on `ConversableAgent` internally call `initiate_chat`/`a_initiate_chat`, so their traces appear as conversation spans. Similarly, `run_group_chat()`/`a_run_group_chat()` call `initiate_group_chat()`/`a_initiate_group_chat()` which trace via the agent's `initiate_chat`.

## Semantic Conventions

See [OTEL_GENAI_CONVENTION_AG2.md](./OTEL_GENAI_CONVENTION_AG2.md) for the full attribute reference.

### Standard OTEL GenAI Attributes
- `gen_ai.operation.name` - Operation type
- `gen_ai.agent.name` - Agent name
- `gen_ai.input.messages` / `gen_ai.output.messages` - Message payloads
- `gen_ai.tool.name`, `gen_ai.tool.call.id`, etc.

### AG2 Custom Attributes
- `ag2.span.type` - AG2 span classification
- `ag2.speaker_selection.candidates` / `ag2.speaker_selection.selected`
- `ag2.human_input.prompt` / `ag2.human_input.response` - Human-in-the-loop input
- `ag2.code_execution.exit_code` / `ag2.code_execution.output` - Code execution results
- `ag2.chats.count`, `ag2.chats.mode`, `ag2.chats.recipients`, `ag2.chats.summaries` - Multi-chat workflow
- `gen_ai.conversation.turns`, `gen_ai.usage.cost`

## Message Format

Messages are converted from AG2/OpenAI format to OTEL GenAI format:

```python
# AG2 format
{"role": "user", "content": "Hello", "name": "user_proxy"}

# OTEL format
{"role": "user", "parts": [{"type": "text", "content": "Hello"}]}
```

See `autogen/tracing/utils.py` for conversion functions.

## Local Development

### Prerequisites
```bash
pip install -r requirements.txt
```

### Start Tempo + Grafana
```bash
cd tracing
docker-compose up -d
```

- **Grafana**: http://localhost:3333 (anonymous auth enabled)
- **Tempo OTLP gRPC**: localhost:14317
- **Tempo OTLP HTTP**: localhost:14318

### Run Examples

```bash
# Sync two-agent conversation using initiate_chat()
python -m tracing.agents.local_initiate_chat

# Sequential multi-chat using initiate_chats()
python -m tracing.agents.local_initiate_chats

# Async two-agent conversation using a_initiate_chat()
python -m tracing.agents.local_agents

# Single-agent with tools using run()
python -m tracing.agents.local_run

# Single-agent with tools using a_run()
python -m tracing.agents.local_tools

# Group chat using a_initiate_group_chat() with pattern
python -m tracing.agents.group_chat

# Group chat using run_group_chat() (requires human input)
python -m tracing.agents.local_run_group_chat

# Group chat using a_run_group_chat() (requires human input)
python -m tracing.agents.local_a_run_group_chat

# Human-in-the-loop example
python -m tracing.agents.local_hitl

# Code execution example
python -m tracing.agents.local_code_execution
```

### View Traces
1. Open Grafana at http://localhost:3333
2. Go to Explore → Select Tempo data source
3. Search by service name or trace ID

## Exporting to ClickHouse

The tracing stack includes an OpenTelemetry Collector that can export traces to ClickHouse (including ClickHouse Cloud) for analytics and long-term storage.

### Architecture

```
AG2 App (OTLP gRPC :14317)
         │
         ▼
┌─────────────────────┐
│  OTel Collector     │
│  (receives OTLP)    │
└─────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
ClickHouse  Tempo
  Cloud     (local)
```

### Setup

1. **Create a `.env` file** in the `tracing/` directory with your ClickHouse credentials:

```bash
CLICKHOUSE_HOST=your-instance.clickhouse.cloud
CLICKHOUSE_PORT=8443
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your-password
CLICKHOUSE_DATABASE=default
```

2. **Start the stack**:

```bash
cd tracing
docker-compose up -d
```

The OTel Collector will automatically:
- Create the `otel_traces` table in ClickHouse if it doesn't exist
- Export traces to both ClickHouse and local Tempo simultaneously

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `CLICKHOUSE_HOST` | ClickHouse server hostname | `abc123.us-west-2.aws.clickhouse.cloud` |
| `CLICKHOUSE_PORT` | HTTPS port (usually 8443 for Cloud) | `8443` |
| `CLICKHOUSE_USER` | Database username | `default` |
| `CLICKHOUSE_PASSWORD` | Database password | `your-password` |
| `CLICKHOUSE_DATABASE` | Target database | `default` |

### Querying Traces in ClickHouse

```sql
-- List recent traces
SELECT Timestamp, SpanName, ServiceName, SpanAttributes
FROM otel_traces
ORDER BY Timestamp DESC
LIMIT 10;

-- Find traces by service name
SELECT * FROM otel_traces
WHERE ServiceName = 'my-agent-service'
ORDER BY Timestamp DESC;

-- Get conversation IDs
SELECT
    Timestamp,
    SpanName,
    SpanAttributes['gen_ai.conversation.id'] as conversation_id
FROM otel_traces
WHERE SpanName LIKE 'conversation%'
ORDER BY Timestamp DESC;

-- Analyze token usage
SELECT
    ServiceName,
    sum(toFloat64OrZero(SpanAttributes['gen_ai.usage.input_tokens'])) as total_input_tokens,
    sum(toFloat64OrZero(SpanAttributes['gen_ai.usage.output_tokens'])) as total_output_tokens,
    sum(toFloat64OrZero(SpanAttributes['gen_ai.usage.cost'])) as total_cost
FROM otel_traces
WHERE SpanAttributes['ag2.span.type'] = 'conversation'
GROUP BY ServiceName;
```

### Disabling ClickHouse Export

To export only to local Tempo, edit `otel-collector-config.yaml` and remove `clickhouse` from the exporters:

```yaml
service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/tempo]  # Remove clickhouse
```

## Distributed Tracing (A2A)

For remote agents using A2A protocol, trace context is automatically propagated via W3C Trace Context headers:

1. **Client side**: `instrument_agent` wraps the httpx client factory to inject `traceparent` header
2. **Server side**: `instrument_a2a_server` adds middleware to extract trace context from incoming requests

This allows traces to span across multiple services/processes.

## Implementation Notes

### Monkey-Patching Approach
Instrumentation works by replacing agent methods with traced wrappers. The original method is captured and called within a span context:

```python
old_method = agent.some_method

def traced_method(*args, **kwargs):
    with tracer.start_as_current_span("span_name") as span:
        span.set_attribute("key", "value")
        return old_method(*args, **kwargs)

agent.some_method = traced_method
```

### GroupChat Considerations
GroupChatManager creates a shallow copy of GroupChat during initialization. The instrumentation handles this by also instrumenting the copy stored in `manager._reply_func_list`.

### Async Support
All instrumentation supports both sync and async methods where applicable.

## Future Work

- [x] LLM invocation spans (`gen_ai.operation.name: chat`) - Implemented via `instrument_llm_wrapper()`
- [ ] Handoff spans
- [ ] Span events for streaming responses
