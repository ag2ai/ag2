# agent-metrics Specification

## Purpose
TBD - created by archiving change prometheus-metrics. Update Purpose after archive.
## Requirements
### Requirement: Agent turn counter metric
The system SHALL emit a counter metric `ag2_agent_turns_total` for each agent turn with labels: `agent`, `outcome`, `error_type`.

#### Scenario: Successful agent turn increments counter
- **WHEN** an agent turn completes successfully
- **THEN** the counter is incremented with `outcome="success"` and the agent name

#### Scenario: Failed agent turn increments counter with error outcome
- **WHEN** an agent turn fails with an exception
- **THEN** the counter is incremented with `outcome="error"` and `error_type` set to the exception type name

#### Scenario: Agent name extracted from context
- **WHEN** an agent turn executes
- **THEN** the `agent` label is set to the agent name from context

#### Scenario: Missing agent name normalized to unknown
- **WHEN** an agent turn executes without agent name in context
- **THEN** the `agent` label is set to `"unknown"`

### Requirement: Agent turn duration histogram metric
The system SHALL emit a histogram metric `ag2_agent_turn_duration_seconds` for each agent turn with labels: `agent`, `outcome`, `error_type` and custom buckets `[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0, +Inf]`.

#### Scenario: Duration recorded on successful turn
- **WHEN** an agent turn completes successfully
- **THEN** the histogram records the turn duration in seconds

#### Scenario: Duration recorded on failed turn
- **WHEN** an agent turn fails
- **THEN** the histogram records the turn duration with `outcome="error"` and appropriate `error_type`
