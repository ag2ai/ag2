---
title: "Code Examples"
sidebarTitle: "Overview"
---

End-to-end runnable scripts demonstrating `autogen.beta`. Each example is self-contained, instantiates a `GeminiConfig` directly, and exercises one or two specific harness primitives so you can read it top-to-bottom and copy what you need.

## Examples

| # | Page | Topic | Primitives covered |
|---|------|-------|--------------------|
| 01 | [Hello Agent](01_hello_agent.md) | Minimal Agent — one config, one ask | `Agent`, `GeminiConfig` |
| 02 | [Recipe Builder](02_recipe_builder.md) | Tools + Pydantic structured output | `Agent`, `tools=`, `response_schema=` |
| 03 | [Travel Planner](03_travel_planner.md) | Multi-turn chat via chained `reply.ask()` | `agent.ask()`, `reply.ask()` |
| 04 | [Token Watchdog](04_token_watchdog.md) | Built-in + custom observers, `ObserverAlert` | `BaseObserver`, `TokenMonitor`, `LoopDetector`, `EventWatch` |
| 05 | [Research Squad](05_research_squad.md) | Parallel subtasks + sibling delegation | `run_subtasks(parallel=True)`, `Agent.as_tool()` |
| 06 | [Journal Companion](06_journal_companion.md) | Persistent memory across runs | `KnowledgeConfig`, `WorkingMemoryAggregate`, `WorkingMemoryPolicy` |
| 07 | [Long-Doc Chat](07_long_doc_chat.md) | Composing assembly policies + compaction | `assembly=[...]`, `SlidingWindowPolicy`, `TokenBudgetPolicy`, `TailWindowCompact` |
| 08 | [Safety Guard](08_safety_guard.md) | FATAL alert → `AlertPolicy` → `HaltEvent` | `ObserverAlert(FATAL)`, `AlertPolicy`, `HaltEvent` |

Set `GEMINI_API_KEY` (or swap in another provider's `ModelConfig` — `AnthropicConfig`, `OpenAIConfig`, etc.) before running.
