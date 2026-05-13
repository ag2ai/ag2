# human-input-metrics Specification

## Purpose
TBD - created by archiving change prometheus-metrics. Update Purpose after archive.
## Requirements
### Requirement: Human input request counter metric
The system SHALL emit a counter metric `ag2_human_input_requests_total` for each human input request with labels: `agent`, `outcome`, `error_type`.

#### Scenario: Successful human input increments counter
- **WHEN** a human input request completes successfully
- **THEN** the counter is incremented with `outcome="success"` and the agent name

#### Scenario: Failed human input increments counter with error outcome
- **WHEN** a human input request fails or is cancelled
- **THEN** the counter is incremented with `outcome="error"` and `error_type` set to the exception type name

#### Scenario: Timeout counted as error outcome
- **WHEN** a human input request times out
- **THEN** the counter is incremented with `outcome="error"` and `error_type="TimeoutError"`

### Requirement: Human input duration histogram metric
The system SHALL emit a histogram metric `ag2_human_input_duration_seconds` for each human input request with labels: `agent`, `outcome`, `error_type` and custom buckets `[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 1800.0, 3600.0, +Inf]`.

#### Scenario: Duration recorded on successful input
- **WHEN** a human input request completes successfully
- **THEN** the histogram records the request duration in seconds

#### Scenario: Duration recorded on failed input
- **WHEN** a human input request fails or times out
- **THEN** the histogram records the request duration with `outcome="error"` and appropriate `error_type`
