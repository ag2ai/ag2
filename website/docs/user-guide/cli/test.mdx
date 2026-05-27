---
sidebarTitle: Test
title: "ag2 test"
description: "Run evaluation suites and benchmarks against your agents."
---

The `ag2 test` command runs structured evaluations against your agents.

## ag2 test eval

Run YAML-defined test cases against an agent and check assertions on the output.

```bash
ag2 test eval <agent_file> --eval <eval_file> [--output <format>]
```

### Eval File Format

```yaml title="tests/eval.yaml"
cases:
  - name: greeting
    input: "Say hello"
    assertions:
      - type: contains
        value: "hello"

  - name: math
    input: "What is 2 + 2?"
    assertions:
      - type: contains
        value: "4"
      - type: max_turns
        value: 3
```

### Assertion Types

| Type | Description |
|------|-------------|
| `contains` | Output contains the given string |
| `contains_all` | Output contains all given strings |
| `contains_any` | Output contains at least one string |
| `not_contains` | Output does not contain the string |
| `regex` | Output matches a regex pattern |
| `min_length` / `max_length` | Output length bounds |
| `max_turns` | Conversation completed within N turns |
| `max_cost` | Total cost stayed under budget |
| `max_tokens` | Token usage stayed under limit |
| `max_time` | Execution completed within time limit |
| `tool_called` | A specific tool was invoked |
| `tool_not_called` | A specific tool was not invoked |
| `no_error` | No errors occurred during execution |
| `llm_judge` | LLM-based evaluation with criteria and threshold |

### LLM Judge Example

```yaml
cases:
  - name: quality_check
    input: "Write a haiku about Python"
    assertions:
      - type: llm_judge
        criteria: "Is this a valid haiku with 5-7-5 syllable structure?"
        threshold: 0.8
```

### Options

| Flag | Description |
|------|-------------|
| `--eval` | Path to YAML eval file |
| `--output` | Output format: `table` (default), `json`, `junit` |
| `--dry-run` | Show test cases without running them |
| `--runs` | Run each case N times for determinism testing |
| `--baseline` | Compare results against a baseline file for regression |

### Examples

```bash
# Run evals and show results table
ag2 test eval my_agent.py --eval tests/eval.yaml

# Output as JSON
ag2 test eval my_agent.py --eval tests/eval.yaml --output json

# Dry run to preview cases
ag2 test eval my_agent.py --eval tests/eval.yaml --dry-run
```
