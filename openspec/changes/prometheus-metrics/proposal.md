## Why

Operators need Prometheus-compatible metrics to monitor AG2 Beta workloads in production. While AG2 Beta has OTel-based tracing, it lacks a stable first-party metrics contract for LLM token usage. This RFC implements the full metrics surface defined in issue #2686, enabling operators to track agent turns, LLM calls, tool execution, human input requests, and token consumption via Prometheus and Grafana.

## What Changes

- Add `MetricsMiddleware` that emits the following metrics:

### Counters
- `ag2_agent_turns_total` — agent turn count with labels: `agent`, `outcome`
- `ag2_llm_calls_total` — LLM call count with labels: `agent`, `provider`, `model`, `outcome`, `finish_reason`, `error_type` (when applicable)
- `ag2_tool_calls_total` — tool call count with labels: `agent`, `tool`, `outcome`, `error_type` (when applicable)
- `ag2_human_input_requests_total` — human input request count with labels: `agent`, `outcome`, `error_type` (when applicable)
- `ag2_llm_tokens_total` — LLM token count with labels: `agent`, `provider`, `model`, `token_type`

### Histograms
- `ag2_agent_turn_duration_seconds` — agent turn duration with custom buckets: `[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0, +Inf]`
- `ag2_llm_call_duration_seconds` — LLM call duration with custom buckets: `[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, +Inf]`
- `ag2_tool_duration_seconds` — tool execution duration with custom buckets: `[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, +Inf]`
- `ag2_human_input_duration_seconds` — human input duration with custom buckets: `[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 1800.0, 3600.0, +Inf]`

### Token Types
Supported `token_type` values: `input`, `output`, `cache_read_input`, `cache_creation_input`

### Label Strategy
- **Outcome labeling**: Single `outcome` label with values `"success"` or `"error"`, plus optional `error_type` label with exception name when `outcome="error"`
- **Agent name extraction**: Retrieved from `context.dependencies` using internal key `"__ag2_agent__"`
- **Provider/model extraction**: Retrieved from `ModelConfig` via `context.dependencies` using internal key `"__ag2_model_config__"`. `ModelConfig` protocol exposes `provider` and `model` properties.
- **Missing value normalization**: Missing values normalized to `"unknown"` label value
- **Zero value handling**: Zero token values omitted from emission to reduce cardinality
- **Missing values**: MUST NOT be synthesized as zero
- **CollectorRegistry ownership**: A `CollectorRegistry` can be used by only one `MetricsMiddleware` instance. To share
  a registry across agents, share the middleware instance as well.

### Streaming Support
- Duration tracking supports streaming LLM responses by measuring from first chunk to final response
- Retries counted as separate calls if they produce separate responses (transparent to middleware)

## Capabilities

### New Capabilities

- `llm-metrics`: Counter and histogram metrics for LLM calls (total count, duration, tokens) with provider/model/outcome/error_type dimensions
- `tool-metrics`: Counter and histogram metrics for tool execution (total count, duration) with tool name and outcome/error_type dimensions
- `agent-metrics`: Counter and histogram metrics for agent turns (total count, duration) with agent name and outcome/error_type dimensions
- `human-input-metrics`: Counter and histogram metrics for human input requests (total count, duration) with agent name and outcome/error_type dimensions

### Modified Capabilities

None — this is a new observability capability.

## Impact

- **Module**: `autogen/beta/middleware/builtin/metrics.py`
- **Tests**: `test/beta/middleware/builtins/metrics/`
- **Documentation**: `website/docs/beta/metrics.mdx` — comprehensive user guide
- **Dependencies**: `prometheus_client` (optional dependency, installable via `pip install "ag2[metrics]"`)
- **Public API**: `MetricsMiddleware` class exposed via `autogen.beta.middleware`
- **Agent modification**: `Agent.ask()` modified to inject agent and ModelConfig into context dependencies
- **ModelConfig modification**: Add `provider` and `model` properties to `ModelConfig` protocol

## Documentation

A comprehensive documentation page at `website/docs/beta/metrics.mdx` following the style of `telemetry.mdx`. The documentation should include:

- **Quick Start**: Complete working example with Prometheus registry setup
- **Installation**: Required dependencies (`pip install "ag2[metrics]"`)
- **Metrics Reference**: Table of all emitted metrics with names, types, labels, and descriptions
- **Configuration**: All `MetricsMiddleware` parameters with defaults and explanations
- **CollectorRegistry lifecycle**: Explain that `prometheus_client` has no public collector lookup API, so users must
  create one `MetricsMiddleware` per `CollectorRegistry` and reuse it across agents instead of constructing multiple
  middleware instances for the same registry
- **Label Reference**: All label values and their meanings (outcome values, token_type values, error_type values)
- **Prometheus Integration**: How to expose the `/metrics` endpoint
- **Grafana Dashboard**: Example queries and dashboard setup
- **Best Practices**: Label cardinality considerations, naming conventions, streaming considerations
