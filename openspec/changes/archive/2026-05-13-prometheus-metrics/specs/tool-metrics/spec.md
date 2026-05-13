## ADDED Requirements

### Requirement: Tool call counter metric
The system SHALL emit a counter metric `ag2_tool_calls_total` for each tool execution with labels: `agent`, `tool`, `outcome`, `error_type`.

#### Scenario: Successful tool execution increments counter
- **WHEN** a tool execution completes successfully
- **THEN** the counter is incremented with `outcome="success"` and the tool name

#### Scenario: Failed tool execution increments counter with error outcome
- **WHEN** a tool execution fails with an exception
- **THEN** the counter is incremented with `outcome="error"` and `error_type` set to the exception type name

#### Scenario: Tool name extracted from ToolCallEvent
- **WHEN** a tool is executed
- **THEN** the `tool` label is set to the tool name from `ToolCallEvent.name`

### Requirement: Tool execution duration histogram metric
The system SHALL emit a histogram metric `ag2_tool_duration_seconds` for each tool execution with labels: `agent`, `tool`, `outcome`, `error_type` and custom buckets `[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, +Inf]`.

#### Scenario: Duration recorded on successful execution
- **WHEN** a tool execution completes successfully
- **THEN** the histogram records the execution duration in seconds

#### Scenario: Duration recorded on failed execution
- **WHEN** a tool execution fails
- **THEN** the histogram records the execution duration with `outcome="error"` and appropriate `error_type`
