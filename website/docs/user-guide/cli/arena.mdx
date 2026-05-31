---
sidebarTitle: Arena
title: "ag2 arena"
description: "A/B test and compare agent implementations with ELO ratings."
---

The `ag2 arena` command lets you systematically compare agent implementations, models, or strategies using evaluation suites.

## Basic Comparison

Compare two agent files against an eval suite:

```bash
ag2 arena agent_v1.py agent_v2.py --eval tests/eval.yaml
```

This runs both agents against every test case and reports:

- **Pass rate** per agent
- **Quality scores** (from `llm_judge` assertions)
- **Cost** comparison
- **Latency** comparison
- **Statistical significance** of differences

## Model Comparison

Compare the same agent across different LLM backends:

```bash
ag2 arena my_agent.py --models gpt-4o,claude-sonnet-4-20250514 --eval tests/eval.yaml
```

## Interactive Mode

Run head-to-head comparisons where you vote on the better output:

```bash
ag2 arena agent_a.py agent_b.py --interactive
```

## Tournament Mode

Run multiple agents against multiple benchmarks with a leaderboard:

```bash
ag2 arena agent_*.py --eval tests/ --tournament
```

## ELO Leaderboard

Arena maintains ELO ratings across sessions in `~/.ag2/arena/leaderboard.json`:

```bash
ag2 arena --leaderboard
```

## Options

| Flag | Description |
|------|-------------|
| `--eval` | Path to eval YAML file or directory |
| `--models` | Comma-separated list of models to compare |
| `--interactive` | Interactive voting mode |
| `--tournament` | Tournament mode with leaderboard |
| `--leaderboard` | Show current ELO leaderboard |
| `--budget` | Maximum cost budget for the run |
| `--dry-run` | Estimate cost without running |
