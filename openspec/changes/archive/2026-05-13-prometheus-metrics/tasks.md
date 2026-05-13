## 1. Agent Context Enhancement

- [x] 1.1 Modify `Agent.ask()` to inject agent instance into `context.dependencies` with internal key `"__ag2_agent__"`

## 2. ModelConfig Enhancement

- [x] 2.1 Add `provider` property to `ModelConfig` protocol
- [x] 2.2 Add `model` property to `ModelConfig` protocol
- [x] 2.3 Implement `provider` property in all ModelConfig implementations (AnthropicConfig, OpenAIConfig, GeminiConfig, OllamaConfig, DashScopeConfig)
- [x] 2.4 Modify `Agent.ask()` to inject `ModelConfig` into `context.dependencies` with internal key `"__ag2_model_config__"`

## 3. Core Metrics Infrastructure

- [x] 3.1 Add proper `prometheus_client` import with optional dependency handling
- [x] 3.2 Define all counter and histogram metrics in `MetricsMiddleware.__init__` with custom buckets per operation type
- [x] 3.3 Implement label extraction helpers (`_get_agent_name`, `_normalize_label`, etc.)
- [x] 3.4 Add error type extraction for outcome labeling

## 4. LLM Metrics Implementation

- [x] 4.1 Extend `on_llm_call` to emit `ag2_llm_calls_total` counter with all labels including `error_type`
- [x] 4.2 Add duration tracking for `ag2_llm_call_duration_seconds` histogram with custom buckets
- [x] 4.3 Implement `ag2_llm_tokens_total` counter with token type labels
- [x] 4.4 Handle all token types: prompt, completion, cache_read_input, cache_creation_input
- [x] 4.5 Implement missing value normalization to "unknown"
- [x] 4.6 Handle zero token values by omitting emission
- [x] 4.7 Support streaming LLM responses for duration tracking

## 5. Tool Metrics Implementation

- [x] 5.1 Implement `on_tool_execution` hook in `_MetricsMiddleware`
- [x] 5.2 Emit `ag2_tool_calls_total` counter with tool name, outcome, and error_type labels
- [x] 5.3 Add duration tracking for `ag2_tool_duration_seconds` histogram with custom buckets

## 6. Agent Turn Metrics Implementation

- [x] 6.1 Implement `on_turn` hook in `_MetricsMiddleware`
- [x] 6.2 Emit `ag2_agent_turns_total` counter with agent name, outcome, and error_type labels
- [x] 6.3 Add duration tracking for `ag2_agent_turn_duration_seconds` histogram with custom buckets

## 7. Human Input Metrics Implementation

- [x] 7.1 Implement `on_human_input` hook in `_MetricsMiddleware`
- [x] 7.2 Emit `ag2_human_input_requests_total` counter with agent name, outcome, and error_type labels
- [x] 7.3 Add duration tracking for `ag2_human_input_duration_seconds` histogram with custom buckets

## 8. Testing

- [x] 8.1 Add tests for LLM call counter with all label combinations
- [x] 8.2 Add tests for LLM duration histogram with custom buckets
- [x] 8.3 Add tests for token counter with all token types
- [x] 8.4 Add tests for missing value normalization
- [x] 8.5 Add tests for zero token value handling (omission)
- [x] 8.6 Add tests for tool execution metrics with error types
- [x] 8.7 Add tests for agent turn metrics with error types
- [x] 8.8 Add tests for human input metrics with error types
- [x] 8.9 Add tests for streaming LLM response duration tracking
- [x] 8.10 Add tests for retry counting behavior
- [x] 8.11 Add tests for agent injection in context dependencies
- [x] 8.12 Add test that creating multiple `MetricsMiddleware` instances for one `CollectorRegistry` fails clearly

## 9. Documentation

- [x] 9.1 Create `website/docs/beta/metrics.mdx` with frontmatter and overview
- [x] 9.2 Add Installation section with dependency instructions (`pip install "ag2[metrics]"`)
- [x] 9.3 Add Quick Start section with complete working example (Prometheus registry, agent setup)
- [x] 9.4 Add Metrics Reference table (all counters and histograms with labels, custom buckets)
- [x] 9.5 Add Label Reference section (outcome values, error_type values, token_type values, normalization rules)
- [x] 9.6 Add Configuration section (all MetricsMiddleware parameters)
- [x] 9.7 Add Prometheus Integration section (exposing /metrics endpoint)
- [x] 9.8 Add Grafana Dashboard section (example queries, dashboard setup, success rate calculations)
- [x] 9.9 Add Best Practices section (cardinality considerations, naming conventions, streaming considerations)
- [x] 9.10 Document `CollectorRegistry` lifecycle: create one `MetricsMiddleware` per registry and share it across agents
- [x] 9.11 Add page to navigation in `website/mint-json-template.json.jinja`
