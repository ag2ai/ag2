## Context

AG2 Beta has a robust OTel-based tracing model via `TelemetryMiddleware`. This change adds a first-party `MetricsMiddleware` using `prometheus_client` for metrics that are critical for operations (especially token tracking).

The framework provides the necessary building blocks:
- `ModelResponse` carries `usage`, `provider`, `model`, `finish_reason` fields
- `BaseMiddleware` provides hooks: `on_turn`, `on_llm_call`, `on_tool_execution`, `on_human_input`
- `Context.dependencies` allows dependency injection across middleware

## Goals / Non-Goals

**Goals:**
- Implement full metrics surface from RFC #2686
- Use `prometheus_client` for direct Prometheus exposition
- Emit counters and histograms for all runtime phases
- Handle missing values by normalizing to `"unknown"`
- Support all token types: `prompt`, `completion`, `cache_read_input`, `cache_creation_input`
- Extract agent name from context via internal dependency key

**Non-Goals:**
- OTel metrics integration (traces remain the OTel surface)
- Exporter lifecycle management (application responsibility)
- HTTP `/metrics` endpoint setup (application responsibility)

## Decisions

### 1. Use `prometheus_client` directly (not OTel metrics)

**Rationale:** The RFC explicitly rejected full OTel metrics to avoid duplicating count/duration signals already in traces. Prometheus is the primary target. Direct `prometheus_client` integration is simpler and matches user expectations.

**Alternatives considered:**
- OTel metrics with Prometheus exporter: Rejected per RFC — adds complexity without benefit since we already have traces.

### 2. Custom histogram buckets per operation type

Use custom buckets aligned with expected latencies for each operation type:

- **Agent turns**: `[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0, +Inf]` (0.1s → 10m)
- **LLM calls**: `[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, +Inf]` (50ms → 30s)
- **Tool execution**: `[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, +Inf]` (10ms → 10s)
- **Human input**: `[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 1800.0, 3600.0, +Inf]` (1s → 1h)

### 3. Agent name and ModelConfig extraction from context

Agent name is extracted from `context.dependencies` using an internal key:
```python
def _get_agent_name(self, context: Context) -> str:
    agent = context.dependencies.get("__ag2_agent__")
    return agent.name if agent else "unknown"
```

The `Agent.ask()` method will inject the agent instance into context dependencies with an internal key
(`"__ag2_agent__"`) so user dependencies named `"agent"` remain available to tools and hooks.

**ModelConfig injection:** In addition to the agent instance, `Agent.ask()` will inject `ModelConfig` into
`context.dependencies` using an internal key (`"__ag2_model_config__"`). This allows `MetricsMiddleware` to determine
provider and model name without reserving a broad user dependency name like `"model_config"`.

To ensure type safety, `ModelConfig` protocol will expose `provider` and `model` as properties:

```python
class ModelConfig(Protocol):
    @property
    def provider(self) -> str:
        ...

    @property
    def model(self) -> str:
        ...
```

Each `ModelConfig` implementation must provide these properties, ensuring that mypy will report an error if a new config implementation is added without implementing them.

Provider values for each implementation:
- `AnthropicConfig` → `"anthropic"`
- `OpenAIConfig` → `"openai"`
- `GeminiConfig` → `"gemini"`
- `OllamaConfig` → `"ollama"`
- `DashScopeConfig` → `"dashscope"`

### 4. Middleware architecture: single `MetricsMiddleware` class

**Rationale:** All metrics share the same lifecycle (middleware factory pattern). A single class is simpler than multiple middleware classes.

**Structure:**
```python
class MetricsMiddleware(MiddlewareFactory):
    def __init__(self, registry: CollectorRegistry) -> None:
        # Initialize all counters and histograms

    def __call__(self, event: BaseEvent, context: Context) -> _MetricsMiddleware:
        return _MetricsMiddleware(event, context, ...)
```

`prometheus_client` does not expose a public API for retrieving already registered collectors from a
`CollectorRegistry`. Because of that, `MetricsMiddleware` does not try to reuse collectors across middleware instances.
Creating more than one `MetricsMiddleware` for the same `CollectorRegistry` raises a clear `ValueError`; callers who want
multiple agents to emit into one registry should create one middleware instance and share it across those agents.

### 5. Label extraction strategy

Extract labels from:
- Agent name via `context.dependencies.get("__ag2_agent__")`
- `ModelConfig.provider`, `ModelConfig.model`, and `ModelResponse.finish_reason` for LLM metrics
- `ToolCallEvent.name` for tool name
- Exception type for `outcome` on error, `"success"` otherwise

### 6. Outcome labeling strategy

Single metric with `outcome` label and optional `error_type` label:
- `outcome`: `"success"` or `"error"`
- `error_type`: exception type name (only when `outcome="error"`)

This enables simple success rate calculations and flexible error analysis.

### 7. Token metric emission strategy

Zero token values are omitted from emission to reduce cardinality:
```python
if usage.prompt_tokens:  # 0 or None - skip
    counter.labels(..., token_type="input").inc(usage.prompt_tokens)
```

Missing token fields are not synthesized as zero.

### 8. Duration tracking for streaming LLM responses

Duration is measured from the first `ModelMessageChunk` to the final `ModelResponse`. The middleware tracks the start time on the first chunk and records duration on the final response.

### 9. Retry handling

Retry middleware interactions are transparent to metrics. Each middleware invocation (whether retry or not) produces separate metric increments. No special retry handling is needed.

### 10. Optional dependency handling

`prometheus_client` is an optional dependency with proper ImportError handling:
```python
try:
    from prometheus_client import Counter, Histogram, CollectorRegistry
except ImportError as _err:
    raise ImportError(
        "prometheus_client is required for MetricsMiddleware. Install with: pip install ag2[metrics]"
    ) from _err
```

## Risks / Trade-offs

- **Label cardinality explosion** → Mitigation: Document that users should avoid high-cardinality values (e.g., user IDs) in agent/tool names. The label set is fixed and bounded.

- **Missing provider/model info** → Mitigation: Normalize to `"unknown"` per RFC. All metrics have fallback handling.

- **Performance overhead** → Mitigation: `prometheus_client` is optimized for high-throughput. Counter/histogram operations are O(1) with label caching.

- **Histogram memory usage** → Mitigation: Each unique label combination creates a histogram instance. Document cardinality implications.

- **Streaming duration complexity** → Mitigation: Track start time per request ID or context to handle streaming correctly.
