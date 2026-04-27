# `autogen.beta` Playground

Runnable examples showing how to build with `Agent` and the framework-core
harness primitives (observers, knowledge store, assembly policies, subtask
delegation, compaction, aggregation).

Every script hits a real LLM — **no mocks, no stubs**. They are designed to
be read top-to-bottom as tutorials and run as smoke tests.

## Prerequisites

```bash
# 1. Install beta extras (one-time)
uv sync --extra beta

# 2. Populate .env at the repo root
cat > .env <<'EOF'
GEMINI_API_KEY=...
# optional — only needed by examples that opt into another provider
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
EOF
```

## Running

```bash
# From the repo root
.venv-beta/bin/python playground/01_hello_actor.py
```

Every example is self-contained and prints what it's doing.

## Index

| # | File | Topic | Primitives exercised |
|---|------|-------|----------------------|
| 01 | `01_hello_actor.py` | Minimal Agent — one config, one ask | `Agent`, `GeminiConfig` |
| 02 | `02_recipe_builder.py` | Tools + Pydantic response schema | `Agent`, `tool`, `response_schema` |
| 03 | `03_travel_planner.py` | Multi-turn chat via chained `reply.ask()` | `AgentReply.ask`, context memory |
| 04 | `04_token_watchdog.py` | Observers: `TokenMonitor`, `LoopDetector`, custom `BaseObserver` | `BaseObserver`, `EventWatch`, `ObserverAlert` |
| 05 | `05_research_squad.py` | Parallel subtasks + sibling `as_tool` delegation | `run_subtasks`, `Agent.as_tool` |
| 06 | `06_journal_companion.py` | Knowledge store + working-memory aggregation | `KnowledgeConfig`, `WorkingMemoryAggregate`, `WorkingMemoryPolicy` |
| 07 | `07_long_doc_chat.py` | Assembly policies composed: conversation → sliding window → token budget | `assembly=[...]`, `SlidingWindowPolicy`, `TokenBudgetPolicy` |
| 08 | `08_safety_guard.py` | FATAL alert → `AlertPolicy` → `HaltEvent` → short-circuit | `ObserverAlert(FATAL)`, `AlertPolicy`, `HaltEvent` |

## Model choice

All examples default to `gemini-3-flash-preview` because it is cheap, fast, and
capable enough to drive tool-use, structured output, and the harness features
that ship with `autogen.beta`. Swap `_config.default_config()` for your own
`ModelConfig` if you want to run against another provider.
