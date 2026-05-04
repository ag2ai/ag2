## 1. Agent Context Enhancement

- [ ] 1.1 Modify `Agent.ask()` to inject agent instance into `context.dependencies` with key `"agent"`
- [ ] 1.2 Add tests to verify agent is properly injected into context dependencies

## 2. Core Metrics Infrastructure

- [ ] 2.1 Add proper `prometheus_client` import with optional dependency handling
- [ ] 2.2 Define all counter and histogram metrics in `MetricsMiddleware.__init__` with custom buckets per operation type
- [ ] 2.3 Implement label extraction helpers (`_get_agent_name`, `_normalize_label`, etc.)
- [ ] 2.4 Add error type extraction for outcome labeling

## 3. LLM Metrics Implementation

- [ ] 3.1 Extend `on_llm_call` to emit `ag2_llm_calls_total` counter with all labels including `error_type`
- [ ] 3.2 Add duration tracking for `ag2_llm_call_duration_seconds` histogram with custom buckets
- [ ] 3.3 Implement `ag2_llm_tokens_total` counter with token type labels
- [ ] 3.4 Handle all token types: prompt, completion, cache_read_input, cache_creation_input
- [ ] 3.5 Implement missing value normalization to "unknown"
- [ ] 3.6 Handle zero token values by omitting emission
- [ ] 3.7 Support streaming LLM responses for duration tracking

## 4. Tool Metrics Implementation

- [ ] 4.1 Implement `on_tool_execution` hook in `_MetricsMiddleware`
- [ ] 4.2 Emit `ag2_tool_calls_total` counter with tool name, outcome, and error_type labels
- [ ] 4.3 Add duration tracking for `ag2_tool_duration_seconds` histogram with custom buckets

## 5. Agent Turn Metrics Implementation

- [ ] 5.1 Implement `on_turn` hook in `_MetricsMiddleware`
- [ ] 5.2 Emit `ag2_agent_turns_total` counter with agent name, outcome, and error_type labels
- [ ] 5.3 Add duration tracking for `ag2_agent_turn_duration_seconds` histogram with custom buckets

## 6. Human Input Metrics Implementation

- [ ] 6.1 Implement `on_human_input` hook in `_MetricsMiddleware`
- [ ] 6.2 Emit `ag2_human_input_requests_total` counter with agent name, outcome, and error_type labels
- [ ] 6.3 Add duration tracking for `ag2_human_input_duration_seconds` histogram with custom buckets

## 7. Testing

- [ ] 7.1 Add tests for LLM call counter with all label combinations
- [ ] 7.2 Add tests for LLM duration histogram with custom buckets
- [ ] 7.3 Add tests for token counter with all token types
- [ ] 7.4 Add tests for missing value normalization
- [ ] 7.5 Add tests for zero token value handling (omission)
- [ ] 7.6 Add tests for tool execution metrics with error types
- [ ] 7.7 Add tests for agent turn metrics with error types
- [ ] 7.8 Add tests for human input metrics with error types
- [ ] 7.9 Add tests for streaming LLM response duration tracking
- [ ] 7.10 Add tests for retry counting behavior
- [ ] 7.11 Add tests for agent injection in context dependencies

## 8. Documentation

- [ ] 8.1 Create `website/docs/beta/metrics.mdx` with frontmatter and overview
- [ ] 8.2 Add Installation section with dependency instructions (`pip install "ag2[metrics]"`)
- [ ] 8.3 Add Quick Start section with complete working example (Prometheus registry, agent setup)
- [ ] 8.4 Add Metrics Reference table (all counters and histograms with labels, custom buckets)
- [ ] 8.5 Add Label Reference section (outcome values, error_type values, token_type values, normalization rules)
- [ ] 8.6 Add Configuration section (all MetricsMiddleware parameters)
- [ ] 8.7 Add Prometheus Integration section (exposing /metrics endpoint)
- [ ] 8.8 Add Grafana Dashboard section (example queries, dashboard setup, success rate calculations)
- [ ] 8.9 Add Best Practices section (cardinality considerations, naming conventions, streaming considerations)
- [ ] 8.10 Add page to navigation in `website/mint-json-template.json.jinja`
