# llm-metrics Specification

## Purpose
TBD - created by archiving change prometheus-metrics. Update Purpose after archive.
## Requirements
### Requirement: LLM call counter metric
The system SHALL emit a counter metric `ag2_llm_calls_total` for each LLM call with labels: `agent`, `provider`, `model`, `outcome`, `finish_reason`, `error_type`.

#### Scenario: Successful LLM call increments counter
- **WHEN** an LLM call completes successfully
- **THEN** the counter is incremented with `outcome="success"` and appropriate labels

#### Scenario: Failed LLM call increments counter with error outcome
- **WHEN** an LLM call fails with an exception
- **THEN** the counter is incremented with `outcome="error"` and `error_type` set to the exception type name

#### Scenario: Missing provider normalized to unknown
- **WHEN** an LLM call completes without provider information
- **THEN** the `provider` label is set to `"unknown"`

#### Scenario: Missing model normalized to unknown
- **WHEN** an LLM call completes without model information
- **THEN** the `model` label is set to `"unknown"`

#### Scenario: Missing finish_reason normalized to unknown
- **WHEN** an LLM call completes without finish reason
- **THEN** the `finish_reason` label is set to `"unknown"`

### Requirement: LLM call duration histogram metric
The system SHALL emit a histogram metric `ag2_llm_call_duration_seconds` for each LLM call with labels: `agent`, `provider`, `model`, `outcome`, `error_type` and custom buckets `[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, +Inf]`.

#### Scenario: Duration recorded on successful call
- **WHEN** an LLM call completes successfully
- **THEN** the histogram records the call duration in seconds

#### Scenario: Duration recorded on failed call
- **WHEN** an LLM call fails
- **THEN** the histogram records the call duration with `outcome="error"` and appropriate `error_type`

### Requirement: LLM token counter metric
The system SHALL emit a counter metric `ag2_llm_tokens_total` for token usage with labels: `agent`, `provider`, `model`, `token_type`.

#### Scenario: Prompt tokens emitted
- **WHEN** an LLM response contains `prompt_tokens` in usage
- **THEN** the counter is incremented with `token_type="input"` by the token count

#### Scenario: Completion tokens emitted
- **WHEN** an LLM response contains `completion_tokens` in usage
- **THEN** the counter is incremented with `token_type="output"` by the token count

#### Scenario: Cache read input tokens emitted
- **WHEN** an LLM response contains `cache_read_input_tokens` in usage
- **THEN** the counter is incremented with `token_type="cache_read_input"` by the token count

#### Scenario: Cache creation input tokens emitted
- **WHEN** an LLM response contains `cache_creation_input_tokens` in usage
- **THEN** the counter is incremented with `token_type="cache_creation_input"` by the token count

#### Scenario: Zero token values omitted
- **WHEN** an LLM response contains zero value for a token field
- **THEN** that token type MAY be omitted from emission

#### Scenario: Missing token values not synthesized
- **WHEN** an LLM response lacks a token field
- **THEN** that token type MUST NOT be emitted as zero

### Requirement: Retry counting
The system SHALL count retries as separate LLM calls if they produce separate provider requests/responses.

#### Scenario: Retry counted as separate call
- **WHEN** a retry middleware retries an LLM call
- **THEN** each retry attempt is counted as a separate call

### Requirement: Streaming LLM response duration tracking
The system SHALL track LLM call duration from the first `ModelMessageChunk` to the final `ModelResponse`.

#### Scenario: Streaming duration recorded
- **WHEN** an LLM call returns streaming responses
- **THEN** the duration is measured from the first chunk to the final response
