# ag2 arena

> A/B test agent implementations — compare quality, cost, and speed.

## Problem

"Is my agent v2 actually better than v1?" Teams answer this by eyeballing
outputs. There's no systematic way to compare two agent implementations
across the same test cases, models, or scenarios.

## Commands

```bash
# Compare two implementations
ag2 arena agent_v1.py agent_v2.py --eval tests/cases.yaml

# Compare across models
ag2 arena my_agent.py --models gpt-4o,claude-sonnet-4-6 --eval tests/cases.yaml

# Tournament — multiple agents, multiple benchmarks
ag2 arena agents/ --eval tests/ --format table

# Interactive head-to-head
ag2 arena agent_v1.py agent_v2.py --interactive

# Export comparison report
ag2 arena agent_v1.py agent_v2.py --eval tests/ --output report.html
```

## Comparison Modes

### Eval-Based Comparison

```bash
ag2 arena agent_v1.py agent_v2.py --eval tests/cases.yaml
```

```
╭─ AG2 Arena ────────────────────────────────────────────────╮
│ Contenders: agent_v1.py vs agent_v2.py                     │
│ Eval cases: 10 | Model: gpt-4o                             │
╰────────────────────────────────────────────────────────────╯

                        agent_v1    agent_v2    winner
  basic_search          ✓           ✓           tie
  tool_usage            ✓           ✓           tie
  multi_step            ✗ (0.62)    ✓ (0.91)    agent_v2
  error_handling        ✓           ✗           agent_v1
  complex_reasoning     ✓ (0.78)    ✓ (0.95)    agent_v2
  code_generation       ✓           ✓           tie
  long_context          ✗           ✓           agent_v2
  structured_output     ✓           ✓           tie
  adversarial           ✗           ✗           tie
  latency_test          ✓ (2.1s)    ✓ (1.3s)    agent_v2

╭─ Summary ──────────────────────────────────────────────────╮
│                   agent_v1        agent_v2                  │
│ Pass rate         70%             80%          (+14%)       │
│ Avg quality       0.74            0.88         (+19%)       │
│ Avg time          3.2s            2.4s         (-25%)       │
│ Avg cost/case     $0.08           $0.06        (-25%)       │
│ Total cost        $0.80           $0.60                     │
│                                                             │
│ Winner: agent_v2 (better on 3 cases, worse on 1)           │
╰─────────────────────────────────────────────────────────────╯
```

### Model Comparison

```bash
ag2 arena my_agent.py --models gpt-4o,claude-sonnet-4-6,gemini-2.0-flash --eval tests/
```

Same agent, different backends. Helps you choose the best model for your use case.

### Interactive Mode

```bash
ag2 arena agent_v1.py agent_v2.py --interactive
```

```
╭─ AG2 Arena — Interactive ──────────────────────────╮
│ Send the same message to both agents.              │
│ Pick the winner for each round.                    │
╰────────────────────────────────────────────────────╯

You: Explain the CAP theorem with real-world examples

  ┌─ Agent A ────────────────────────────────────────┐
  │ The CAP theorem states that a distributed...     │
  │ (534 tokens, 2.1s, $0.02)                        │
  └──────────────────────────────────────────────────┘

  ┌─ Agent B ────────────────────────────────────────┐
  │ CAP theorem (Brewer's theorem) defines three...  │
  │ (412 tokens, 1.8s, $0.01)                        │
  └──────────────────────────────────────────────────┘

  Which is better? [A] Agent A  [B] Agent B  [T] Tie  [S] Skip
  > B

  Score: Agent A: 0  Agent B: 1  Ties: 0

You: █
```

Agent identities are hidden (A/B) to avoid bias. Revealed at the end.

### Tournament Mode

```bash
ag2 arena agents/ --eval tests/ --format table
```

Runs every agent file in `agents/` against every eval case:

```
╭─ Tournament Results ───────────────────────────────────────╮
│                case1  case2  case3  case4  case5  Score     │
│ researcher_v1  ✓      ✓      ✗      ✓      ✓     80%      │
│ researcher_v2  ✓      ✓      ✓      ✓      ✗     80%      │
│ researcher_v3  ✓      ✓      ✓      ✓      ✓     100% 🏆  │
│ baseline       ✓      ✗      ✗      ✓      ✗     40%      │
╰────────────────────────────────────────────────────────────╯
```

## ELO Rating System

For interactive mode, maintain an ELO rating across sessions:

```bash
ag2 arena --leaderboard
```

```
╭─ Agent Leaderboard ────────────────────────────────╮
│ Rank  Agent              ELO    W/L/T    Last      │
│ 1     researcher_v3      1523   12/2/1   Today     │
│ 2     researcher_v2      1487   8/4/3    Today     │
│ 3     researcher_v1      1445   6/6/3    Yesterday │
│ 4     baseline           1320   2/10/3   Yesterday │
╰─────────────────────────────────────────────────────╯
```

## Implementation Notes

### Parallel Execution
Run both agents concurrently using `asyncio.gather()` for speed.
Ensure they have isolated state (fresh agent instances per case).

### Statistical Significance
With `--runs N`, run each case N times and compute confidence intervals.
Report whether differences are statistically significant (p < 0.05).

### Cost Controls
- `--budget $5.00` — stop when total arena cost exceeds budget
- `--dry-run` — estimate cost before running
- Individual case cost limits from eval YAML

### Integration with ag2 test
Arena builds on the same eval case format as `ag2 test eval`.
The assertion system is shared — arena just runs two agents instead of one.
