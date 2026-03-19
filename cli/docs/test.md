# ag2 test

> Evaluate, benchmark, and regression-test agents from the command line.

## Problem

Testing agents is the #1 pain point for production teams. There's no `pytest`
equivalent for agents — no way to define test cases, run them reproducibly,
compare across models, or track regressions. CrewAI has a basic `crewai test`.
AutoGenBench exists but is standalone and decoupled. Nobody does this well.

## Commands

### `ag2 test eval` — Run evaluation cases

```bash
# Run eval suite
ag2 test eval my_agent.py --eval tests/cases.yaml

# Test against golden transcripts
ag2 test eval my_team.py --eval tests/transcripts/

# Compare across models
ag2 test eval my_agent.py --eval tests/ --models gpt-4o,claude-sonnet-4-6,gemini-2.0-flash

# Estimate cost without running
ag2 test eval my_agent.py --eval tests/ --dry-run

# Regression test against baseline
ag2 test eval my_agent.py --eval tests/ --baseline results/v1.json

# Output results in different formats
ag2 test eval my_agent.py --eval tests/ --output results.json
ag2 test eval my_agent.py --eval tests/ --output results.html
```

### `ag2 test bench` — Standardized benchmarks

```bash
# Run against standard benchmark suites
ag2 test bench my_agent.py --suite gaia
ag2 test bench my_agent.py --suite humaneval
ag2 test bench my_agent.py --suite swe-bench-lite

# Custom benchmark directory
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

### Multi-Model Comparison

```bash
ag2 test eval my_agent.py --eval tests/ --models gpt-4o,claude-sonnet-4-6,gemini-2.0-flash
```

```
╭─ Model Comparison ─────────────────────────────────────────────────╮
│                  gpt-4o    claude-sonnet    gemini-2.0-flash       │
│ Pass rate        75%       100%             50%                    │
│ Avg time         2.4s      3.1s             1.8s                  │
│ Avg cost/case    $0.06     $0.04            $0.02                 │
│ Total cost       $0.24     $0.16            $0.08                 │
╰────────────────────────────────────────────────────────────────────╯
```

### Regression Testing

```bash
# Save baseline
ag2 test eval my_agent.py --eval tests/ --output baseline.json

# Compare against baseline
ag2 test eval my_agent.py --eval tests/ --baseline baseline.json
```

```
╭─ Regression Report ────────────────────────────────╮
│ ✓ basic_search          PASS → PASS (stable)       │
│ ✓ tool_usage            PASS → PASS (stable)       │
│ ⚠ multi_step_reasoning  FAIL → PASS (improved!)    │
│ ✗ error_handling         PASS → FAIL (REGRESSION)   │
│                                                     │
│ Score: 75% → 75% (no change)                        │
│ Cost:  $0.24 → $0.20 (17% cheaper)                 │
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
- `--seed` flag to set LLM seed (when supported)
- `--runs N` to run each case N times and report pass rates
- `llm_judge` provides fuzzy evaluation for open-ended outputs
- `max_cost`/`max_turns`/`max_time` provide deterministic bounds

### Integration with pytest
Also provide a pytest plugin so users can run agent evals in their existing test suite:

```python
# test_agents.py
from ag2_cli.testing import ag2_eval

@ag2_eval("tests/cases.yaml")
def test_research_agent():
    from my_agents import researcher
    return researcher
```

```bash
pytest test_agents.py -v
```

## Dependencies
- `ag2` — required for agent execution
- `pyyaml` — already in CLI deps
- `rich` — already in CLI deps
- LLM provider SDK — for llm_judge assertions
