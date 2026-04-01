# V2 Iteration Review: Framework Layering & API Design

## Status

**IMPLEMENTED.** All changes below have been applied to the codebase. 499 tests passing.

Original goal: identify what to merge, remove, and restructure before release.

---

## Concept Audit

Before restructuring, we audited every abstraction pair in the framework to determine what's genuinely distinct vs what's redundant. This ensures we aren't just moving code around — we're shipping fewer concepts.

### Concepts to Eliminate or Merge

**Signal system is redundant with Event + AssemblyPolicy.**

`Signal` is a `BaseEvent` subclass with a parallel delivery channel (`SignalInjectionMiddleware` + `SignalPolicy`) that duplicates what assembly policies already do. The dedicated channel exists because Signals were designed before `AssemblyPolicy` replaced the old `ContextHarness`. The old harness couldn't inject into prompts, so Signals needed their own middleware. Now that assembly policies can modify prompts, the justification is gone.

| Signal system component | Replacement |
|------------------------|-------------|
| `Signal` event type | `ObserverAlert(BaseEvent)` — same fields (source, severity, message, data), clearer name |
| `SignalPolicy` protocol | An assembly policy (e.g., `AlertPolicy`) that reads ObserverAlerts from the stream and injects them into prompts |
| `SignalInjectionMiddleware` | Removed — alerts handled by AlertPolicy in the assembly chain |
| `InjectToPrompt` | Becomes the default AlertPolicy behavior |
| `EmitToStream` / `CallHandler` | Observers emit events on the stream directly (they already do — Signal IS a BaseEvent) |
| FATAL halt | Observer emits `HaltEvent` directly on the stream; a lightweight middleware catches it (or AlertPolicy does) |

What survives: the `Observer` abstraction (Watch + event production + error isolation), the severity concept (INFO/WARNING/CRITICAL/FATAL), and the FATAL halt mechanic. What dies: the dedicated delivery side-channel.

**Triple context window management collapses to two distinct concerns.**

Three systems currently truncate old events:

| System | Mechanism | Example |
|--------|-----------|---------|
| Core Middleware | `HistoryLimiter`, `TokenLimiter` | Per-call, middleware hook |
| Assembly Policies | `SlidingWindowPolicy`, `TokenBudgetPolicy` | Per-call, composable transform |
| Compaction | `TailWindowCompact`, `SummarizeCompact` | Post-turn, destructive, persistent |

`HistoryLimiter` ≈ `SlidingWindowPolicy` (count-based). `TokenLimiter` ≈ `TokenBudgetPolicy` (token-based). Assembly policies are strictly superior — they compose with other policies (episodic memory, working memory), support transparency notes, and participate in ordering validation.

Resolution: Assembly policies are the canonical per-call mechanism. Core middleware stays available for plain Agent users who don't use assembly, but positioned as simple fallbacks. Compaction is genuinely different (destructive, trigger-based, persists dropped events) and stays as a separate concern.

**StateStore and KnowledgeStore stay as separate protocols.**

Both are async key-value stores, but they differ in type signature and semantic level:

| | StateStore | KnowledgeStore |
|---|---|---|
| Value type | `Any` | `str` |
| Semantics | Operational state (ephemeral, TTL) | Domain knowledge (persistent, filesystem) |
| User | Hub, plugins | Actor, policies |
| Mental model | Redis-like cache | Virtual filesystem |

Merging would force either Hub to store operational state in a filesystem abstraction (semantically wrong) or KnowledgeStore to gain `Any` values and TTL (becomes a Swiss Army knife doing two things). The protocol separation is cheap (two small interfaces) and preserves intent at usage sites (`state_store: StateStore` vs `knowledge_store: KnowledgeStore`).

Implementations may converge — e.g., a single backing store class could implement both protocols — but the protocol distinction stays. Both move to framework core (neither is network-specific).

**Cache protocol is removed.** Nothing uses it. Ship when something does.

### Concepts That Look Redundant But Aren't

Each of these was investigated and confirmed to provide unique value:

**Watch vs Stream.subscribe** — Watch adds timer-based triggers (IntervalWatch, DelayWatch, CronWatch), batching (BatchWatch, WindowWatch), and composition (AllOf, AnyOf, Sequence). Stream.subscribe is low-level fire-and-forget. EventWatch is the simplest member of the Watch family — it's a thin wrapper over subscribe, but exists to share the Watch interface with timer/batch watches. Correct layering: subscribe is the primitive, Watch is the abstraction.

**Channel vs Stream** — Channel is the envelope delivery layer (addressing, backpressure, priority, remote transport). Stream is the event observation layer (history, filtering, dependency injection). Channel moves envelopes between agents. Stream records what happened. Hub uses both: Channel to route, Stream to observe. Not redundant.

**AssemblyPolicy vs Middleware** — Policies are pure `(prompts, events) -> (prompts, events)` transforms. Middleware is stateful with 4 lifecycle hooks. Policies live inside AssemblerMiddleware. One is a transform, the other is a lifecycle hook. Complementary.

**Observer vs Stream.subscribe** — Observer adds Watch composition, error isolation (exceptions logged, don't crash stream), explicit lifecycle (attach/detach), and typed event production. It's a monitoring pattern built on subscribe. Different abstraction level.

**Envelope vs BaseEvent** — Envelope wraps BaseEvent with network metadata (sender, recipient, trace_id, TTL, priority). Merging would couple core framework to network concepts and bloat every event with unused fields. Clean layer boundary.

**Topology vs Plugin** — Topology implements Plugin to enable nesting (Pipeline can contain Pipeline). Minor conceptual overlap but functionally correct. The implicit conformance could be documented more clearly.

**Hub.ask() vs registration-time tool injection** — Hub.ask() injects network tools at call time because tools need caller context and delegation depth. Registration-time injection would break this. Correct design.

### Consolidated Concept Map

After eliminating redundancies, the framework has **14 concepts** (10 framework + 4 network). Each does one thing that nothing else does:

**Framework core (10):**

| Concept | What | Unique Value |
|---------|------|-------------|
| Event | What happened | Application semantics, typed, serializable |
| Stream | Event bus + history | Observation, filtering, persistence |
| Watch | Reactive trigger | Timers, batching, composition (AllOf/AnyOf/Sequence) |
| Middleware | Lifecycle hooks | Stateful cross-cutting (retry, logging, telemetry) |
| AssemblyPolicy | Context transform | Composable `(prompts, events) -> (prompts, events)` |
| Observer | Watch + event production | Monitoring pattern with error isolation |
| KnowledgeStore | Persistent knowledge | Virtual filesystem for actor knowledge |
| StateStore | Operational state | Ephemeral key-value with TTL for Hub/plugin coordination |
| CompactStrategy | Destructive stream reduction | Long-running agent health |
| AggregateStrategy | Constructive knowledge creation | Long-term memory from short-term experience |

**Network (4):**

| Concept | What | Unique Value |
|---------|------|-------------|
| Envelope | Event + network metadata | Addressing, tracing, TTL |
| Channel | Envelope delivery | Backpressure, priority, remote transport |
| Hub | Network center | Registry, delegation routing, network tool injection |
| Topology/Plugin | Routing composition | Pipeline, Fanout, Conditional with RouteDecision |

**Removed:**

| Was | Becomes |
|-----|---------|
| Signal (event + delivery channel) | `ObserverAlert` event + `AlertPolicy` assembly policy |
| SignalPolicy | Absorbed into AlertPolicy |
| SignalInjectionMiddleware | Removed |
| ContextHarness / HarnessMiddleware | Already replaced by AssemblyPolicy |
| ConversationHarness / NetworkHarness | Already replaced by ConversationPolicy / NetworkPolicy |
| Cache protocol | Removed (unused) |

---

## Core Insight: Network vs Framework

The most important structural finding is a **layering mistake**. Many features currently in the network module are general-purpose agent capabilities that have nothing to do with networking:

| Feature | Currently In | Actually Belongs To |
|---------|-------------|-------------------|
| KnowledgeStore | `network.primitives` | Framework core — any agent benefits from persistent knowledge |
| AssemblyPolicy + Assembler | `network.assembler` + `network.policies` | Framework core — any agent needs context window management |
| CompactStrategy | `network.primitives` | Framework core — any long-running agent needs this |
| AggregateStrategy | `network.primitives` | Framework core — any agent with memory needs this |
| Observer + Watch | `network.observer` + `network.primitives` | Framework core — monitoring is not networking |
| Envelope + Channel | `network.primitives` | Network — transport is genuinely network |
| Hub + Registry | `network.hub` | Network — discovery and routing are genuinely network |
| Topology + Plugin | `network.topology` | Network — routing composition is genuinely network |
| Scheduler | `network.scheduler` | Framework core — reactive scheduling is not networking |
| Topics | `network` (inside Hub) | Network — pub/sub between agents is networking |
| RemoteAgent | `network.remote` | Network — cross-process communication is networking |

An agent that persists knowledge, manages its context window, compacts its history, and monitors its own token usage is not doing anything "network." It's doing competent single-agent operation. These capabilities should be available to `Agent` users without importing from `network`.

Conversely: Hub, delegation, topology, channels, envelopes, topics, remote agents — these are genuinely about agent-to-agent communication and belong in a network module.

---

## Proposed Structure

### Framework Core (`autogen.beta`)

What moves here from network:

```
autogen/beta/
  agent.py              # Agent (existing)
  stream.py             # Stream, MemoryStream (existing)
  context.py            # Context (existing)
  events/               # BaseEvent, model events, tool events (existing)
    types.py             # + ObserverAlert, HaltEvent (promoted from Signal)
  middleware/            # BaseMiddleware, chain (existing)
  tools/                # Tool, FunctionTool (existing)

  # --- Promoted from network ---

  knowledge.py           # KnowledgeStore protocol + MemoryKnowledgeStore
  state.py               # StateStore protocol + MemoryStateStore (Hub/plugin operational state)
  assembly.py            # AssemblyPolicy protocol + AssemblerMiddleware
  compact.py             # CompactStrategy protocol + TailWindowCompact, SummarizeCompact
  aggregate.py           # AggregateStrategy protocol + ConversationSummaryAggregate, WorkingMemoryAggregate
  observer.py            # Observer protocol + BaseObserver (output: BaseEvent, not Signal)
  watch.py               # Watch protocol + all watch types
  scheduler.py           # Scheduler (single class, optional Hub parameter for delegation mode)

  policies/              # Assembly policies that don't require network
    conversation.py      # ConversationPolicy (default)
    sliding_window.py    # SlidingWindowPolicy
    token_budget.py      # TokenBudgetPolicy
    episodic_memory.py   # EpisodicMemoryPolicy
    working_memory.py    # WorkingMemoryPolicy
    alert.py             # AlertPolicy (replaces SignalPolicy — reads ObserverAlerts, injects into prompt)

  observers/             # Built-in observers
    loop_detector.py
    token_monitor.py
```

No `signal.py`. No `SignalPolicy`. No `SignalInjectionMiddleware`. Observer alert delivery is handled by `AlertPolicy` in the assembly chain, using the same mechanism as every other policy. FATAL halt is handled by the observer emitting `HaltEvent` directly on the stream.

**AlertPolicy implementation notes:**

AlertPolicy replaces the dedicated `_SignalInjectionMiddleware` queue. Key behaviors to preserve:

- **Deduplication**: Track delivered alert IDs across calls (equivalent to the old `_delivered_ids` set). Each alert is injected once, then marked as delivered.
- **Cross-turn accumulation**: Read `ObserverAlert` events from stream history, not just current-turn events. Observers emit to the stream via `ctx.send()`, so all alerts land in stream history regardless of when they fire (during tool execution, during other middleware, etc.).
- **Ordering**: AlertPolicy should be an injection policy — place it after other injection policies (working memory, episodic memory) and before reduction policies (sliding window, token budget). The existing policy ordering validation already catches this.
- **FATAL handling**: `_HaltCheckMiddleware` sits after the assembler in the middleware chain. It subscribes to `HaltEvent` on the stream. When AlertPolicy emits a HaltEvent (on FATAL alert), the middleware catches it and returns a synthetic `ModelResponse(content="HALTED: ...")` without calling the LLM. This is simpler than the old approach where `_SignalInjectionMiddleware` split fatal/non-fatal signals and conditionally skipped `call_next`.

### Network Module (`autogen.beta.network`)

What stays here — genuinely about agent-to-agent communication:

```
autogen/beta/network/
  hub.py                 # Hub: registry + delegation routing (topics extracted)
  topology.py            # Pipeline, Fanout, Conditional, RouteDecision
  envelope.py            # Envelope (network metadata wrapper)
  channel.py             # Channel protocol + LocalChannel, BufferedChannel, PriorityChannel
  remote.py              # RemoteAgent (aligned with Channel)
  events.py              # DelegationRequest/Result, TopicMessage, FormattedEvent
  convenience.py         # Network class (Hub + Scheduler wiring)

  channels/
    http.py              # HttpChannel

  plugins/               # Hub plugins
    rate_limiter.py
    telemetry.py
    topic.py             # TopicPlugin (extracted from Hub)

  policies/              # Network-specific assembly policies
    network.py           # NetworkPolicy
    topic_inbox.py       # TopicInboxPolicy
```

### Actor: Agent + Everything

Actor becomes a convenience class that wires framework core features onto Agent. It does NOT live in `network` — it's the "batteries-included" Agent:

```python
# autogen/beta/actor.py (NOT network)
class Actor(Agent):
    """Agent with observers, knowledge, assembly, compaction, aggregation, and tasks."""
```

An Actor that isn't registered with a Hub is just a capable single agent. Adding it to a Hub makes it network-aware. This is the correct layering.

---

## What Gets Merged, Removed, or Deferred

Concept-level merges (Signal system, context window, StateStore/KnowledgeStore, Cache, ContextHarness) are covered in the Concept Audit above. This section covers structural and design-level changes.

### Extract: Topics from Hub

Topics (publish, subscribe, cursor tracking) are currently wired into Hub.py (~100 lines of topic state management). Hub's core job is discovery + delegation routing.

Move topics to a `TopicPlugin` that:
- Implements the Plugin protocol (install/uninstall)
- Manages its own topic state
- Exposes publish/subscribe via the network tool

Hub stays focused: registry + delegation + topology.

### Clarify: spawn_task vs Hub Delegation

| spawn_task | Hub delegation |
|-----------|---------------|
| Creates anonymous throwaway Agent | Routes to registered named Agent |
| Invisible to Hub (no observability) | Full topology, plugins, events |
| No network tools on subtask | Subtask gets network tools |
| Always available on Actor | Requires Hub registration |

These serve different purposes but the distinction isn't obvious to users.

**Recommendation:** Rename `spawn_task` to `run_subtask` (emphasizes "local compute," not delegation). Document clearly: "Use `run_subtask` for isolated compute work. Use Hub delegation for cross-agent collaboration."

If a Hub is available, consider having `run_subtask` optionally emit a TaskRequest/TaskResult on the Hub stream for observability, even though it doesn't route through topology.

### Align: RemoteAgent with Channel

RemoteAgent has its own HTTP client with retry/backoff. HttpChannel is the proper HTTP transport. These should be unified:

```python
# Current: RemoteAgent has its own HTTP logic
class RemoteAgent:
    async def ask(self, message):
        async with httpx.AsyncClient() as client:
            response = await client.post(self._url, ...)

# Proposed: RemoteAgent wraps Channel
class RemoteAgent:
    def __init__(self, url, *, channel: Channel | None = None):
        self._channel = channel or HttpChannel(url)
```

### Defer: Priority System

PriorityScheme, ConflictResolver, DefaultPriority, PriorityChannel — designed but unused in practice. No component auto-assigns priorities. No user scenario currently requires it.

**Keep internally.** Don't feature in public API or docs for v1. Ship when distributed deployment creates real need.

---

## API Considerations

### Import Paths

The key question: what does a user import and from where?

**Single-agent with memory (no network):**

```python
from autogen.beta import Actor
from autogen.beta import MemoryKnowledgeStore, KnowledgeConfig
from autogen.beta import TailWindowCompact, CompactTrigger
from autogen.beta import ConversationSummaryAggregate, AggregateTrigger
from autogen.beta import ConversationPolicy, SlidingWindowPolicy, AlertPolicy
from autogen.beta import TokenMonitor

actor = Actor(
    "researcher",
    config=config,
    knowledge=KnowledgeConfig(
        store=MemoryKnowledgeStore(),
        compact=TailWindowCompact(100),
        compact_trigger=CompactTrigger(max_events=200),
        aggregate=ConversationSummaryAggregate(config),
    ),
    assembly=[ConversationPolicy(), AlertPolicy(), SlidingWindowPolicy(50)],
    observers=[TokenMonitor(warn_threshold=50_000)],
)
reply = await actor.ask("Research AI trends")
```

No `network` import needed. Nothing here is network-related. Observer alerts are delivered through AlertPolicy in the assembly chain — same mechanism as every other policy.

**Multi-agent network:**

```python
from autogen.beta import Actor
from autogen.beta.network import Hub, Network, Pipeline, RateLimiter, TopicPlugin
from autogen.beta.network import NetworkPolicy, TopicInboxPolicy

hub = Hub(
    topology=Pipeline(RateLimiter(max_per_minute=10)),
    plugins=[TopicPlugin()],  # Topics are now a plugin, not built into Hub
)
await hub.register(researcher, capabilities=["research"])
await hub.register(writer, capabilities=["writing"])

reply = await hub.ask(researcher, "Research and write a report")
```

Network imports only when doing agent-to-agent communication. Note that `TopicInboxPolicy` now takes a `TopicPlugin` instance (not Hub) as its first argument.

### Actor.__init__ Simplification

Current: 20 individual parameters. Proposed: group related params and remove Signal-related params.

```python
# Current (too many params, includes dead Signal system)
Actor(
    name="agent",
    config=config,
    knowledge_store=store,
    bootstrap=bootstrap,
    assembly=[...],
    compact=strategy,
    compact_trigger=trigger,
    aggregate=strategy,
    aggregate_trigger=trigger,
    observers=[...],
    signal_policy=policy,      # ← REMOVED (Signal system eliminated)
    task_config=config,
    task_prompt="...",
    ...
)

# Proposed: grouped configuration, no signal_policy
Actor(
    name="agent",
    config=config,
    knowledge=KnowledgeConfig(        # Groups 5 params
        store=store,
        compact=TailWindowCompact(100),
        compact_trigger=CompactTrigger(max_events=200),
        aggregate=ConversationSummaryAggregate(config),
        aggregate_trigger=AggregateTrigger(on_end=True),
    ),
    assembly=[                         # Stays top-level (frequently customized)
        ConversationPolicy(),
        AlertPolicy(),                 # ← replaces signal_policy param
    ],
    observers=[TokenMonitor(50_000)],  # Stays top-level (frequently customized)
    tasks=TaskConfig(                  # Groups 2 params
        config=mini_config,
        prompt="You are a task agent.",
    ),
)
```

Signal delivery is now just another assembly policy in the `assembly` list. No dedicated parameter. This means alert formatting, ordering relative to other policies, and transparency are all controlled the same way as everything else.

`KnowledgeConfig` and `TaskConfig` are simple dataclasses defined in `actor.py`, not protocols.

Middleware chain after restructure::

    1. AssemblerMiddleware(policies)     -- outermost: assembles context (AlertPolicy runs here)
    2. _HaltCheckMiddleware              -- catches HaltEvent from AlertPolicy, short-circuits LLM
    3. CompactionMiddleware              -- triggers compaction after turns
    4. AggregationMiddleware             -- triggers aggregation after turns
    5. User-provided middleware           -- logging, retry, etc.
    6. LLM client call                    -- innermost

### Tool Surface

Actor auto-injects up to 3 tool groups. Reduce to clear, non-overlapping tools:

| Tool | Current Name | Proposed | Exposed When |
|------|-------------|----------|-------------|
| Spawn subtask | `spawn_task` | `run_subtask` | Always (Actor) |
| Spawn multiple | `spawn_tasks` | `run_subtasks` | Always (Actor) |
| Knowledge CRUD | `knowledge` | `knowledge` | KnowledgeStore configured |
| Manual compact/aggregate | `memory` | Remove or rename | Strategy configured |
| Network operations | `network` | `network` | Hub.ask() |

The `memory` tool (compact/summarize) is questionable. Compaction and aggregation are maintenance operations — they should be automatic (trigger-based), not LLM-initiated. An LLM deciding to "compact" its own memory is unreliable. If the triggers are configured correctly, manual invocation shouldn't be needed.

**Recommendation:** Remove the `memory` tool. Compaction and aggregation are infrastructure concerns, not agent decisions. If users want manual control, they call `actor.compact()` or `actor.aggregate()` from application code.

### Network Tool Design

The consolidated `network(action, target, topic, message)` tool works well for LLM ergonomics (one tool, multiple actions). But `query` and `query_list` (reading another actor's knowledge store) are unusual — they break actor encapsulation.

Consider whether cross-actor knowledge access should be:
- **Explicit opt-in** on the target actor (e.g., `Actor(expose_knowledge=True)`)
- **Mediated by delegation** instead (ask the other actor to read its own store)
- **Or removed** for v1 (simpler API, less surface)

### Hub.ask() vs Hub.delegate()

Two entry points:

```python
# Agent mode: actor gets network tools, LLM decides what to delegate
reply = await hub.ask(actor, "Research and write a report")

# Headless mode: direct routing, no LLM call on the initiator
result = await hub.delegate("api-gateway", "worker", "Process batch")
```

Both are needed. `ask()` is for interactive agents. `delegate()` is for infrastructure routing (API gateways, schedulers, external triggers). Keep both.

### Scheduler API

Scheduler has two modes, both served by a single class:

```python
# Standalone mode — no Hub needed, framework core
scheduler = Scheduler()
scheduler.add(IntervalWatch(300), callback=my_health_check)

# Hub mode — delegates through Hub when watch fires
scheduler = Scheduler(hub=hub)
scheduler.add(IntervalWatch(300), target="monitor", task="Check health")
```

Scheduler lives in framework core as one class. The optional `hub` parameter enables delegation mode. No split needed — the optional parameter IS the clean layering. Network convenience class (`Network`) can still wrap Hub + Scheduler for ergonomics.

---

## Summary: Before vs After

### Before (Current) — ~18 concepts, layering issues

```
autogen/beta/                    # Framework core (thin)
  agent.py, stream.py, context.py, events/, middleware/, tools/

autogen/beta/network/            # Everything else (massive, mixed concerns)
  primitives/
    watch.py, signal.py, priority.py, channel.py, envelope.py,
    harness.py (DEAD), infra.py (Cache UNUSED, StateStore OVERLAPS), knowledge.py,
    compact.py, aggregate.py
  policies/
    conversation.py, sliding_window.py, token_budget.py,
    episodic_memory.py, working_memory.py, network.py, topic_inbox.py
  observers/
    loop_detector.py, token_monitor.py
  plugins/
    rate_limiter.py, telemetry.py
  channels/
    http.py
  actor.py, hub.py, observer.py, scheduler.py, assembler.py,
  topology.py, events.py, convenience.py, remote.py
```

### After (Proposed) — 14 concepts, clean layering

```
autogen/beta/                    # Framework core (10 concepts)
  agent.py                       #   Agent (existing)
  actor.py                       #   Actor = Agent + observers + knowledge + assembly
  stream.py                      #   Stream, MemoryStream (existing)
  context.py                     #   Context (existing)
  events/                        #   BaseEvent + ObserverAlert + HaltEvent
  middleware/                    #   BaseMiddleware (existing)
  tools/                         #   Tool, FunctionTool (existing)
  knowledge.py                   #   KnowledgeStore + MemoryKnowledgeStore
  state.py                       #   StateStore + MemoryStateStore (Hub/plugin state)
  assembly.py                    #   AssemblyPolicy + AssemblerMiddleware
  compact.py                     #   CompactStrategy
  aggregate.py                   #   AggregateStrategy
  observer.py                    #   Observer + BaseObserver
  watch.py                       #   Watch (all types)
  scheduler.py                   #   Scheduler (single class, optional hub param)
  policies/                      #   Assembly policies
    conversation.py              #     ConversationPolicy (default)
    sliding_window.py            #     SlidingWindowPolicy
    token_budget.py              #     TokenBudgetPolicy
    episodic_memory.py           #     EpisodicMemoryPolicy
    working_memory.py            #     WorkingMemoryPolicy
    alert.py                     #     AlertPolicy (replaces Signal delivery)
  observers/
    loop_detector.py
    token_monitor.py

autogen/beta/network/            # Network only (4 concepts)
  hub.py                         #   Hub: registry + delegation
  topology.py                    #   Topology/Plugin: Pipeline, Fanout, Conditional
  envelope.py                    #   Envelope
  channel.py                     #   Channel + LocalChannel, BufferedChannel
  remote.py                      #   RemoteAgent (via Channel)
  events.py                      #   DelegationRequest/Result, TopicMessage, FormattedEvent
  convenience.py                 #   Network (Hub + Scheduler wiring)
  channels/
    http.py                      #   HttpChannel
  plugins/
    rate_limiter.py
    telemetry.py
    topic.py                     #   TopicPlugin (extracted from Hub)
  policies/
    network.py                   #   NetworkPolicy
    topic_inbox.py               #   TopicInboxPolicy
```

### What Changed

| Change | Impact |
|--------|--------|
| Signal system eliminated | -3 concepts (Signal, SignalPolicy, SignalInjectionMiddleware) → +1 assembly policy (AlertPolicy) + lightweight _HaltCheckMiddleware for FATAL halt |
| Framework core gains promoted features | Knowledge, state, assembly, compaction, aggregation, observers, watches, scheduler available to all Agent users |
| Network module shrinks | Only genuinely network concerns: Hub, delegation, routing, channels, remote |
| Actor is a framework class | Works standalone or with Hub — correct layering |
| Dead code removed | ContextHarness, HarnessMiddleware, ConversationHarness, NetworkHarness |
| Unused protocols removed | Cache protocol |
| Storage protocols kept separate | StateStore and KnowledgeStore stay as distinct protocols; both move to framework core |
| Priority system deferred | Internal only, not in public API for v1 |
| Topics extracted from Hub | Becomes TopicPlugin — Hub stays focused on registry + delegation. TopicInboxPolicy now takes TopicPlugin (not Hub). |
| Context window story simplified | Assembly policies are canonical; core middleware is fallback for plain Agent |
| Actor.__init__ simplified | signal_policy param removed, knowledge params grouped into KnowledgeConfig |
| memory tool removed | Compaction/aggregation are infrastructure, not agent decisions |
| spawn_task renamed | `run_subtask` — emphasizes local compute, not delegation |
| Scheduler stays unified | Single class in framework core with optional `hub` param — no split |

---

## Migration Notes

### Status: Complete

All phases executed. 499 tests passing. Restructure applied as a single coordinated change.

### What was done

1. **Move files and update imports** — 15 files promoted from `network/` to `beta/`. Both `__init__.py` files updated.
2. **Eliminate Signal system** — Created `ObserverAlert`, `Severity`, `HaltEvent` in `events/alert.py`. Created `AlertPolicy`. Added `_HaltCheckMiddleware` for FATAL halt. Deleted `signal.py`.
3. **Extract TopicPlugin from Hub** — Created `plugins/topic.py`. Updated `TopicInboxPolicy` to take `TopicPlugin` (not Hub). Removed ~50 lines from Hub.
4. **Remove dead code** — Deleted `harness.py`, removed `Cache`/`MemoryCache` from `infra.py`.
5. **Actor.__init__ restructure** — Added `KnowledgeConfig` and `TaskConfig` dataclasses in `actor.py`. Grouped 7 params into `knowledge`, 2 into `tasks`. Removed `signal_policy`.
6. **Rename spawn_task → run_subtask** — Tool names and internal method updated.
7. **Update test imports** — All 22 affected test files updated. No test logic changes needed beyond constructor/import updates.

### Remaining

- **playground/demo**: Imports still reference old paths. Update needed before demo is runnable.
