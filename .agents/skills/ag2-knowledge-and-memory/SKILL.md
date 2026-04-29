---
name: ag2-knowledge-and-memory
description: Persist agent state across runs, shape what the LLM sees per turn, and cap history to fit a context window. Covers `KnowledgeStore` (memory / sqlite / disk / redis), `KnowledgeConfig` (`store=`, `compact=`, `aggregate=`, `bootstrap=`), aggregation strategies (`WorkingMemoryAggregate`, `ConversationSummaryAggregate`), assembly policies (`WorkingMemoryPolicy`, `EpisodicMemoryPolicy`, `ConversationPolicy`, `SlidingWindowPolicy`, `TokenBudgetPolicy`, `AlertPolicy`), and compaction (`TailWindowCompact`, `SummarizeCompact`). Use when the user wants the agent to remember between conversations, manage long histories, or control prompt assembly.
license: Apache-2.0
---

# Knowledge, memory, and context assembly

This skill covers three related primitives that work together:

| Primitive | Lives in | Role |
|---|---|---|
| **`KnowledgeStore`** | `autogen.beta.knowledge` | Path-based persistent storage (memory / sqlite / disk / redis) |
| **Assembly policies** | `autogen.beta.policies` | Shape `(prompts, events)` per turn before the LLM call |
| **Aggregation / Compaction** | `autogen.beta.aggregate` / `.compact` | Write structured knowledge to the store / trim event history |

`KnowledgeConfig` wires all three onto an `Agent` via the `knowledge=` constructor parameter; assembly policies go via `assembly=`.

## When to use what

| User intent | Reach for |
|---|---|
| Remember user preferences / state between conversations | `WorkingMemoryAggregate` + `WorkingMemoryPolicy` (and a persistent store) |
| Summarise each session for next time | `ConversationSummaryAggregate` + `EpisodicMemoryPolicy` |
| Hard-cap event history sent to the LLM | `SlidingWindowPolicy(max_events=N)` |
| Cap by approximate token count | `TokenBudgetPolicy(max_tokens=N)` |
| Drop lifecycle / observer events from the LLM's view | `ConversationPolicy()` |
| Trim stream history (not just LLM view) | `TailWindowCompact` or `SummarizeCompact` |
| Route observer alerts to the LLM | `AlertPolicy()` |

## 60-second recipe — persistent working memory

```python
from autogen.beta import Agent, KnowledgeConfig
from autogen.beta.aggregate import AggregateTrigger, WorkingMemoryAggregate
from autogen.beta.config import OpenAIConfig
from autogen.beta.knowledge import DiskKnowledgeStore
from autogen.beta.policies import ConversationPolicy, WorkingMemoryPolicy

store = DiskKnowledgeStore("./journal-state")
config = OpenAIConfig(model="gpt-5")

agent = Agent(
    "journal",
    prompt="You are a daily journal companion.",
    config=config,
    knowledge=KnowledgeConfig(
        store=store,
        aggregate=WorkingMemoryAggregate(config=config),
        aggregate_trigger=AggregateTrigger(on_end=True),
    ),
    assembly=[
        WorkingMemoryPolicy(),  # injects /memory/working.md on every LLM call
        ConversationPolicy(),
    ],
)
```

After each conversation the aggregate writes `/memory/working.md`. The next time you build an `Agent` against the same store, `WorkingMemoryPolicy` reads that file in and injects it as prompt context. The agent "remembers" without replaying chat history. Full runnable example: `assets/journal_companion.py`.

## `KnowledgeStore` implementations

| Implementation | Use when |
|---|---|
| `MemoryKnowledgeStore()` | Tests, ephemeral sessions |
| `SqliteKnowledgeStore(path)` | Single-process durability — pragmatic default |
| `DiskKnowledgeStore(path)` | Files should be human-readable on disk |
| `RedisKnowledgeStore(url)` | Multi-process / cross-host sharing |
| `LockedKnowledgeStore(inner, lock=...)` | Wrap any store to serialize concurrent writers |

API (all async):

```python
await store.write("/artifacts/report.md", "# Q3...")
text = await store.read("/artifacts/report.md")
children = await store.list("/")            # immediate children, dirs end in '/'
await store.delete("/artifacts/old.md")
exists = await store.exists("/artifacts/report.md")

off = await store.append("/log/events.jsonl", '{"t":1}\n')   # WAL-style
new_slice = await store.read_range("/log/events.jsonl", off)  # only new bytes
sub = await store.on_change("/log/", on_change_callback)
```

## Assembly chain — what the LLM actually sees

Pass `AssemblyPolicy` instances via `assembly=[...]`. The Agent wires an internal `AssemblerMiddleware` at the outermost middleware position. Each policy transforms `(prompts, events)` and pipes into the next.

**Two kinds of policy — order matters: injection before reduction.**

| Kind | Purpose | Built-ins |
|---|---|---|
| **Injection** | Add to `prompts` | `WorkingMemoryPolicy`, `EpisodicMemoryPolicy`, `AlertPolicy` |
| **Reduction** | Trim `events` | `ConversationPolicy`, `SlidingWindowPolicy`, `TokenBudgetPolicy` |

Validate ordering manually:

```python
from autogen.beta.assembly import AssemblerMiddleware
warnings = AssemblerMiddleware.validate_order(policies)  # returns list of warnings on known bad orderings
```

(`AssemblerMiddleware` and the `AssemblyPolicy` protocol live in `autogen.beta.assembly` for advanced/manual harness wiring; you don't need to import them when just passing built-in policies via `assembly=[...]`.)

### Built-in policies

```python
from autogen.beta.policies import (
    AlertPolicy,
    ConversationPolicy,
    EpisodicMemoryPolicy,
    SlidingWindowPolicy,
    TokenBudgetPolicy,
    WorkingMemoryPolicy,
)

# Injection
WorkingMemoryPolicy()                                 # reads /memory/working.md
EpisodicMemoryPolicy(max_episodes=5, transparent=True) # reads recent /memory/conversations/
AlertPolicy()                                          # delivers ObserverAlerts to LLM, halts on FATAL

# Reduction
ConversationPolicy()                                  # drops non-conversation events
SlidingWindowPolicy(max_events=50, transparent=True)  # last N events
TokenBudgetPolicy(max_tokens=32_000, chars_per_token=4, transparent=True)
```

`transparent=True` appends a `[policy_name] Showing X of Y events.` note to the prompt — useful while tuning. Realistic chain:

```python
assembly=[
    WorkingMemoryPolicy(),
    EpisodicMemoryPolicy(max_episodes=3),
    AlertPolicy(),
    SlidingWindowPolicy(max_events=80),
]
```

## Aggregation — writing knowledge to the store

`AggregateStrategy.aggregate(events, ctx, store) → None` extracts and persists. Two built-ins, both take a `ModelConfig` for a summarisation call (use a cheaper model than the agent's main one):

| Strategy | Writes | Pairs with |
|---|---|---|
| `WorkingMemoryAggregate(config=...)` | `/memory/working.md` (single rolling file) | `WorkingMemoryPolicy` |
| `ConversationSummaryAggregate(config=...)` | `/memory/conversations/{ts}_{stream_id}.md` | `EpisodicMemoryPolicy` |

`AggregateTrigger` controls cadence — `every_n_turns`, `every_n_events`, `on_end`. `AggregateTrigger()` alone fires nothing; opt in to at least one. `on_end=True` defaults off because each fire is an LLM call.

## Compaction — trimming stream history

`CompactStrategy.compact(events, ctx, store) → list[BaseEvent]`. Replaces the stream's history. Two built-ins:

| Strategy | Behaviour | Cost |
|---|---|---|
| `TailWindowCompact(target=N)` | Keep last N events; drop the rest (optionally persist to `/log/`) | Zero LLM calls |
| `SummarizeCompact(target=N, config=...)` | Summarise dropped events into one `CompactionSummary`; insert at head | One LLM call per fire |

`CompactTrigger(max_events=N, max_tokens=M, chars_per_token=4)` — fires when any threshold is crossed.

```python
from autogen.beta.compact import CompactTrigger, TailWindowCompact, SummarizeCompact
```

`SummarizeCompact` inserts a `CompactionSummary` event at the head; `ConversationPolicy` allows it through so the LLM still gets that context.

## Wiring it all on the Agent

`KnowledgeConfig` is the bundle:

```python
from dataclasses import dataclass

@dataclass
class KnowledgeConfig:
    store: KnowledgeStore
    compact: CompactStrategy | None = None
    compact_trigger: CompactTrigger | None = None
    aggregate: AggregateStrategy | None = None
    aggregate_trigger: AggregateTrigger | None = None
    bootstrap: StoreBootstrap | None = None    # e.g. DefaultBootstrap()
```

Full shape:

```python
agent = Agent(
    "assistant",
    config=main_config,
    knowledge=KnowledgeConfig(
        store=DiskKnowledgeStore("./state"),
        compact=TailWindowCompact(target=100),
        compact_trigger=CompactTrigger(max_events=200),
        aggregate=ConversationSummaryAggregate(config=summarizer_config),
        aggregate_trigger=AggregateTrigger(every_n_turns=10, on_end=True),
        bootstrap=DefaultBootstrap(),  # seeds /SKILL.md, /artifacts/, /log/, /memory/
    ),
    assembly=[
        WorkingMemoryPolicy(),
        EpisodicMemoryPolicy(max_episodes=3),
        AlertPolicy(),
        SlidingWindowPolicy(max_events=80),
    ],
)
```

The harness wires internal middleware conditionally — `_AssemblerMiddleware`, `_HaltCheckMiddleware`, `_CompactionMiddleware`, `_AggregationMiddleware`. You only pay for what you turn on.

Lifecycle events emitted: `CompactionCompleted` (with `events_before` / `events_after` / `usage`), `AggregationCompleted` (with `strategy` / `usage`), `HaltEvent` (when `AlertPolicy` sees a FATAL alert). Subscribe via `ag2-observers-and-alerts`.

## Going deeper

- `assets/journal_companion.py` — runnable end-to-end working-memory demo (mirrors `code_examples/06`).
- `assets/long_doc_chat.py` — assembly + compaction stress test (mirrors `code_examples/07`).
- Source docs:
  - `website/docs/beta/advanced/knowledge_store.mdx` — store API, `EventLogWriter`, `LockedKnowledgeStore`.
  - `website/docs/beta/advanced/assembly.mdx` — full policy reference and ordering rules.
  - `website/docs/beta/advanced/aggregation.mdx` — aggregate strategies and custom strategies.
  - `website/docs/beta/advanced/compaction.mdx` — compact strategies and custom strategies.
  - `website/docs/beta/agent_harness.mdx` — `KnowledgeConfig` constructor reference, turn-lifecycle middleware order.

## Common pitfalls

- **Reduction before injection** — `SlidingWindowPolicy` before `WorkingMemoryPolicy` means the working memory injection isn't counted against the budget. Always: injections first, then `AlertPolicy`, then reductions.
- **Forgetting `KnowledgeStore` dependency for memory policies** — `WorkingMemoryPolicy` and `EpisodicMemoryPolicy` look up the store via `context.dependencies.get(KnowledgeStore)`. `KnowledgeConfig(store=...)` registers it for you; if you wire the policy manually, register the store in `dependencies` too.
- **Aggregation costs an LLM call per fire** — `on_end=True` on every conversation can add up. Pair `WorkingMemoryAggregate` and `ConversationSummaryAggregate` thoughtfully; consider `every_n_turns=N` for high-volume agents.
- **Mixing `HistoryLimiter` middleware with assembly reduction policies** — they both trim. Pick one mechanism. Assembly is more flexible (rich shaping, transparency notes); `HistoryLimiter` is simpler.
- **`read_range` operates on byte offsets, not character offsets** — multi-byte UTF-8 sequences need careful alignment.
- **Forgetting that `WorkingMemoryAggregate` is destructive** — it overwrites `/memory/working.md` each fire. That's intentional (rolling state, not log) but expect prior content to merge or disappear.
- **Expecting `AlertPolicy` to render alerts to the LLM without being in `assembly=`** — alerts sit on the stream as `ObserverAlert` events but only reach the LLM when `AlertPolicy` injects them.
