# ag2 test

> Evaluate and test agents from the command line.

## Problem

Testing agents is the #1 pain point for production teams. There's no `pytest`
equivalent for agents — no way to define test cases, run them reproducibly,
or track regressions. CrewAI has a basic `crewai test`.
AutoGenBench exists but is standalone and decoupled. Nobody does this well.

## Commands

### `ag2 test eval` — Run evaluation cases

```bash
# Run eval suite
ag2 test eval my_agent.py --eval tests/cases.yaml

# Test against golden transcripts
ag2 test eval my_team.py --eval tests/transcripts/

# Estimate cost without running
ag2 test eval my_agent.py --eval tests/ --dry-run

# Output results as JSON
ag2 test eval my_agent.py --eval tests/ --output json
```

**Options:**
- `--eval` / `-e` — evaluation cases file or directory (YAML)
- `--models` — comma-separated models to compare (coming soon)
- `--baseline` — previous results for regression comparison (coming soon)
- `--dry-run` — estimate cost without running
- `--output` / `-o` — output format: `json`

### `ag2 test bench` — Standardized benchmarks (coming soon)

```bash
ag2 test bench my_agent.py --suite gaia
ag2 test bench my_agent.py --suite humaneval
ag2 test bench my_agent.py --suite swe-bench-lite
ag2 test bench my_agent.py --suite ./my_benchmarks/
```

## Eval Case Format

```yaml
# tests/cases.yaml
name: "research-agent-evals"
description: "Evaluation suite for the research agent"

cases:
  - name: "basic_search"
    input: "What is the capital of France?"
    assertions:
      - type: contains
        value: "Paris"
      - type: max_turns
        value: 2
      - type: max_cost
        value: 0.05

  - name: "tool_usage"
    input: "Find the latest paper on RLHF and summarize the key findings"
    assertions:
      - type: tool_called
        value: "arxiv_search"
      - type: contains_any
        values: ["reinforcement learning", "RLHF", "human feedback"]
      - type: min_length
        value: 200
      - type: max_turns
        value: 5

  - name: "multi_step_reasoning"
    input: "Compare the populations of Tokyo and New York, then estimate which will be larger in 2050"
    assertions:
      - type: contains_all
        values: ["Tokyo", "New York"]
      - type: regex
        pattern: "\\d{1,3}(,\\d{3})*"  # Contains formatted numbers
      - type: llm_judge
        criteria: "Response provides specific population numbers and a reasoned projection"
        threshold: 0.8

  - name: "error_handling"
    input: "Search for information on xyznonexistent12345"
    assertions:
      - type: no_error
      - type: max_turns
        value: 3
      - type: contains_any
        values: ["could not find", "no results", "unable to"]
```

## Assertion Types

| Type | Description | Example |
|------|-------------|---------|
| `contains` | Output contains substring | `"Paris"` |
| `contains_all` | Output contains all substrings | `["Tokyo", "New York"]` |
| `contains_any` | Output contains at least one | `["RLHF", "human feedback"]` |
| `not_contains` | Output does not contain | `"I don't know"` |
| `regex` | Output matches regex | `"\\d{4}"` |
| `min_length` | Minimum character count | `200` |
| `max_length` | Maximum character count | `5000` |
| `max_turns` | Maximum conversation turns | `5` |
| `max_cost` | Maximum cost in USD | `0.10` |
| `max_tokens` | Maximum tokens used | `5000` |
| `max_time` | Maximum wall time (seconds) | `30` |
| `tool_called` | Specific tool was invoked | `"web_search"` |
| `tool_not_called` | Specific tool was NOT invoked | `"code_execution"` |
| `no_error` | No errors during execution | — |
| `exit_code` | Agent terminated with code | `0` |
| `llm_judge` | LLM evaluates against criteria | See below |

### LLM Judge

The `llm_judge` assertion uses a separate LLM call to evaluate output quality:

```yaml
- type: llm_judge
  criteria: "Response is factually accurate, well-structured, and cites sources"
  threshold: 0.8  # minimum score (0.0 to 1.0)
  model: gpt-4o   # optional, defaults to eval model
```

The judge prompt template:
```
Evaluate the following agent response against these criteria:
{criteria}

Agent input: {input}
Agent output: {output}

Score from 0.0 to 1.0, where 1.0 means fully meeting all criteria.
Respond with JSON: {"score": <float>, "reasoning": "<explanation>"}
```

## Terminal Output

```
╭─ AG2 Test ─ research-agent-evals ──────────────────╮
│ Agent: my_agent.py | Model: gpt-4o                  │
│ Cases: 4 | Assertions: 14                           │
╰─────────────────────────────────────────────────────╯

  ✓ basic_search               2/2 assertions   0.3s  $0.01
  ✓ tool_usage                 4/4 assertions   3.2s  $0.08
  ✗ multi_step_reasoning       2/3 assertions   5.1s  $0.12
    └─ FAIL: llm_judge (0.62 < 0.80 threshold)
       "Response lacks specific population projections for 2050"
  ✓ error_handling             3/3 assertions   1.1s  $0.03

╭─ Results ───────────────────────────────────────────╮
│ Passed: 3/4 (75%)                                   │
│ Assertions: 11/14 (79%)                             │
│ Total cost: $0.24                                   │
│ Total time: 9.7s                                    │
╰─────────────────────────────────────────────────────╯
```

## Implementation Notes

### Test Runner Architecture
```
ag2 test eval
  → Load agent from file (same discovery as ag2 run)
  → Parse eval cases from YAML
  → For each case:
      → Create fresh agent instance
      → Run initiate_chat() with the input
      → Capture: output, tool calls, turns, tokens, cost, time, errors
      → Evaluate assertions against captured data
  → Aggregate results
  → Format and display
```

### Determinism
Agent tests are inherently non-deterministic (LLM outputs vary). To handle this:
- `llm_judge` provides fuzzy evaluation for open-ended outputs
- `max_cost`/`max_turns`/`max_time` provide deterministic bounds

## Dependencies
- `ag2` — required for agent execution
- `pyyaml` — already in CLI deps
- `rich` — already in CLI deps
- LLM provider SDK — for llm_judge assertions
