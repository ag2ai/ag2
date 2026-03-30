# Agent Harness

## Context

The AG2 Network Framework (see `ag2_network_framework.md`) provides the infrastructure for distributed autonomous agent networks. All layers are implemented: primitives (Watch, Signal, Channel, Envelope, Priority, Harness, Infra), building blocks (Actor, Hub, Observer, Scheduler), and composition (Topology, Plugins, Network).

The final missing piece is the **Agent Harness** — the system that enables genuine stateful operations at production grade. The current harness (`ContextHarness`) is a stateless filter that runs before each LLM call. It controls what events the LLM sees, but nothing more. A production agent needs:

1. **Persistent knowledge** across conversations and across actors
2. **Intelligent context assembly** from multiple sources (history, knowledge, network)
3. **Post-processing** to maintain healthy operation over time (compaction, aggregation)

These capabilities must work across three scenarios:

| Scenario | What | Example |
|----------|------|---------|
| Single actor, long context | Hot memory within one conversation | Research agent running for hours |
| Single actor, multiple streams | Episodic memory across conversations | Assistant that remembers past sessions |
| Multi-actor context sharing | Network-wide knowledge flow | Team of agents sharing findings |

---

## Decomposition

Any agentic operation decomposes into four components:

```
Persistence (knowledge) → Assembly (context) → Execution (LLM) → Post-Processing (maintenance)
```

1. **Persistence** — what the actor knows. Durable, cross-conversation, actor-owned.
2. **Assembly** — what the actor sees. Composed per LLM call from prompts, events, and knowledge.
3. **Execution** — the LLM call. Already exists. No changes.
4. **Post-processing** — maintenance operations. Compaction respects system constraints. Aggregation retains performance.

The Agent Harness covers components 1, 2, and 4. Component 3 is the existing execution pipeline.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│  AGENT HARNESS                                                     │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │ KnowledgeStore   │  │ CompactStrategy  │  │ AggregateStrategy│ │
│  │                  │  │                  │  │                  │ │
│  │ read / write /   │  │ events →         │  │ events →         │ │
│  │ list / delete /  │  │   reduced events │  │   store writes   │ │
│  │ exists           │  │                  │  │                  │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘ │
│  Layer 2: PRIMITIVES (independent, zero cross-dependencies)        │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Assembler                                                    │  │
│  │                                                              │  │
│  │ AssemblyPolicy₁ → AssemblyPolicy₂ → ... → AssemblerMiddleware│  │
│  │                                                              │  │
│  │ (prompts, events) → transform → transform → ... → LLM call  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  Layer 3: BUILDING BLOCKS (composes primitives)                    │
│                                                                     │
│  Network: Topics (pub/sub) on Hub, delivered via assembly policy   │
│                                                                     │
│  Actor ties everything together in _execute()                      │
└───────────────────────────────────────────────────────────────────┘
```

**Layer placement:**

| Component | Layer | Rationale |
|-----------|-------|-----------|
| `KnowledgeStore` | 2 (Primitive) | Standalone storage protocol. Zero dependencies. |
| `CompactStrategy` | 2 (Primitive) | Standalone transform: events → events. Zero dependencies. |
| `AggregateStrategy` | 2 (Primitive) | Standalone transform: events → store writes. Depends only on KnowledgeStore. |
| `AssemblyPolicy` | 3 (Building Block) | Composes KnowledgeStore reads, event filtering, prompt injection. Depends on Context, events, middleware. |
| `AssemblerMiddleware` | 3 (Building Block) | Bridges AssemblyPolicy into the middleware chain. |
| Topic system | 3 (Building Block) | Extension to Hub. |

Design invariants:

- **Additive.** No changes to Agent, Stream, Context, or any Layer 1 code.
- **Protocol-based.** Every new concept is a protocol with an in-memory default.
- **Independent.** Each primitive works alone. KnowledgeStore without Assembler. CompactStrategy without AggregateStrategy.
- **No backward compatibility.** `ContextHarness`, `HarnessMiddleware`, `ConversationHarness`, `NetworkHarness` are all replaced. The `harness` parameter on Actor is removed.

---

## Layer 2: New Primitives

### KnowledgeStore

The knowledge store is a virtual filesystem scoped to an actor. It stores everything the actor is associated with throughout its lifetime: operational logs, external artifacts, summaries, working memory, and any other data.

The interface is filesystem semantics because:
1. LLMs are trained on filesystem operations (read, write, list, delete with paths)
2. Hierarchical paths give free semantic grouping without schema design
3. Any storage backend (memory, disk, object store, database) can implement path-based key-value

**Protocol:**

```python
@runtime_checkable
class KnowledgeStore(Protocol):
    """Virtual path-based store for actor knowledge.

    Provides filesystem semantics over any storage backend.
    Paths use Unix conventions: /dir/subdir/file.txt
    Directories are implicit — writing /a/b/c.txt implies /a/ and /a/b/ exist.
    Listing returns immediate children. Directory entries end with '/'.
    """

    async def read(self, path: str) -> str | None:
        """Read content at path. Returns None if not found."""
        ...

    async def write(self, path: str, content: str) -> None:
        """Write content to path. Creates parent directories implicitly."""
        ...

    async def list(self, path: str = "/") -> list[str]:
        """List immediate children of a directory path.

        Returns relative names. Directories end with '/'.
        Example: list("/log/") might return ["stream-abc.jsonl", "stream-def.jsonl"]
        """
        ...

    async def delete(self, path: str) -> None:
        """Delete entry at path. No-op if not found."""
        ...

    async def exists(self, path: str) -> bool:
        """Check if path exists."""
        ...
```

5 methods. Minimal, LLM-friendly, backend-agnostic.

**Default implementation:**

```python
class MemoryKnowledgeStore:
    """In-memory KnowledgeStore. Development default.

    Backed by a flat dict. Paths are keys. Directories are inferred
    from stored paths via prefix matching.
    """

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def read(self, path: str) -> str | None:
        return self._data.get(_normalize(path))

    async def write(self, path: str, content: str) -> None:
        self._data[_normalize(path)] = content

    async def list(self, path: str = "/") -> list[str]:
        prefix = _normalize(path).rstrip("/") + "/"
        if prefix == "//":
            prefix = "/"

        children: set[str] = set()
        for key in self._data:
            if not key.startswith(prefix):
                continue
            remainder = key[len(prefix):]
            if "/" in remainder:
                # Subdirectory — return first segment with trailing /
                children.add(remainder.split("/")[0] + "/")
            else:
                children.add(remainder)
        return sorted(children)

    async def delete(self, path: str) -> None:
        normalized = _normalize(path)
        # Delete exact match
        self._data.pop(normalized, None)
        # Delete children (if deleting a directory)
        prefix = normalized.rstrip("/") + "/"
        to_delete = [k for k in self._data if k.startswith(prefix)]
        for k in to_delete:
            del self._data[k]

    async def exists(self, path: str) -> bool:
        normalized = _normalize(path)
        if normalized in self._data:
            return True
        # Check if it's a directory (any children exist)
        prefix = normalized.rstrip("/") + "/"
        return any(k.startswith(prefix) for k in self._data)


def _normalize(path: str) -> str:
    """Normalize path: ensure leading /, collapse //, strip trailing /."""
    if not path.startswith("/"):
        path = "/" + path
    while "//" in path:
        path = path.replace("//", "/")
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    return path
```

**Future backends:**

| Backend | Implementation | Use case |
|---------|---------------|----------|
| `MemoryKnowledgeStore` | `dict[str, str]` | Development, testing |
| `DiskKnowledgeStore` | Local filesystem | Single-machine persistent |
| `S3KnowledgeStore` | S3-compatible object store | Cloud persistent |
| `RedisKnowledgeStore` | Redis hash per actor | Low-latency persistent |

Same protocol, different backend. Same application code.

**Relationship to existing protocols:**

| Protocol | Purpose | Scope |
|----------|---------|-------|
| `Storage` (existing) | Stream event persistence | Per-conversation hot path |
| `StateStore` (existing) | Operational key-value state | Hub, plugins, infrastructure |
| `KnowledgeStore` (new) | Actor knowledge with FS semantics | Per-actor, cross-conversation |

These are distinct concerns. Storage is the working set (in-conversation events). StateStore is operational state (counters, flags). KnowledgeStore is actor knowledge (conversations, artifacts, summaries). They have different access patterns and different mental models.

**Knowledge completeness:**

The knowledge store represents everything the actor is associated with. This includes:

1. **Operational logs** — WAL entries from event streams. The framework provides `EventLogWriter` to persist stream events after a conversation ends:

    ```python
    class EventLogWriter:
        """Persists stream events to the knowledge store as WAL entries."""

        def __init__(self, store: KnowledgeStore) -> None:
            self._store = store

        async def persist(self, stream_id: StreamId, events: Iterable[BaseEvent]) -> None:
            path = f"/log/{stream_id}.jsonl"
            lines = [json.dumps(e.to_dict()) for e in events]
            await self._store.write(path, "\n".join(lines))

        async def load(self, stream_id: StreamId) -> list[dict]:
            path = f"/log/{stream_id}.jsonl"
            content = await self._store.read(path)
            if content is None:
                return []
            return [json.loads(line) for line in content.strip().split("\n") if line]
    ```

    `EventLogWriter` is a utility, not a protocol. It bridges the existing Storage system (per-conversation hot path) to the KnowledgeStore (persistent cross-conversation knowledge). The Actor uses it automatically when a knowledge store is configured.

2. **External artifacts** — user-uploaded files, web-downloaded data, connected data sources. The actor writes these to the store via knowledge tools. The framework doesn't prescribe where — but convention suggests `/artifacts/`.

3. **Summaries and working memory** — outputs of aggregation strategies. Convention: `/memory/`.

**SKILL.md convention:**

The framework does not create or enforce SKILL.md files. They are a convention for helping the actor understand its knowledge store. Aggregation strategies or application code can create them. The built-in knowledge tool checks for SKILL.md when listing a directory and includes its content in the listing, making the convention useful without baking it into the protocol.

Example knowledge store after a few conversations:

```
/
├── log/
│   ├── stream-a1b2c3.jsonl     # Events from conversation 1
│   └── stream-d4e5f6.jsonl     # Events from conversation 2
├── artifacts/
│   ├── SKILL.md                # "Reference materials uploaded by the user"
│   ├── report-draft.md         # User-uploaded file
│   └── dataset.csv             # Downloaded data
└── memory/
    ├── SKILL.md                # "Summaries and working memory"
    ├── working.md              # Current state (updated by aggregation)
    └── conversations/
        ├── a1b2c3.md           # Summary of conversation 1
        └── d4e5f6.md           # Summary of conversation 2
```

This is convention, not mechanism. The store doesn't know or care about this structure. Any organization works.

---

### CompactStrategy

Compaction reduces stream history to respect system constraints: context window limits, I/O degradation in large histories, storage budgets. It operates on the active event stream and returns a smaller event list.

**Protocol:**

```python
@runtime_checkable
class CompactStrategy(Protocol):
    """Reduces stream history to respect system constraints.

    Compaction protects runtime stability. It is the constraint-respecting
    operation — triggered when measurable limits are approached.

    Returns a reduced event list that replaces the current stream history.
    Must preserve causal ordering of retained events.
    """

    async def compact(
        self,
        events: list[BaseEvent],
        context: Context,
        store: KnowledgeStore | None,
    ) -> list[BaseEvent]:
        """Return compacted event list.

        Args:
            events: Current stream history.
            context: Execution context.
            store: Actor's knowledge store (for persisting dropped content). None if not configured.
        """
        ...
```

**Built-in implementations:**

```python
class TailWindowCompact:
    """Keep the last N events. Drop the rest.

    Zero LLM cost. Simplest strategy. Suitable when old context
    has diminishing value and recent events are most relevant.
    """

    def __init__(self, target: int) -> None:
        self._target = target

    async def compact(self, events, context, store):
        if len(events) <= self._target:
            return events
        # Optionally persist dropped events to store
        if store:
            dropped = events[: -self._target]
            await EventLogWriter(store).persist(context.stream.id, dropped)
        return events[-self._target :]
```

```python
class SummarizeCompact:
    """Summarize old events into a CompactionSummary event, keep recent.

    Uses an LLM call to create a summary of dropped events. The summary
    becomes a CompactionSummary event at the head of the history, preserving
    context without preserving all events.

    Costs one LLM call per compaction.
    """

    def __init__(self, target: int, config: ModelConfig) -> None:
        self._target = target
        self._config = config

    async def compact(self, events, context, store):
        if len(events) <= self._target:
            return events
        old = events[: -self._target]
        recent = events[-self._target :]

        # Persist full old events before dropping
        if store:
            await EventLogWriter(store).persist(context.stream.id, old)

        # Summarize via LLM
        summary_text = await self._summarize(old)
        summary_event = CompactionSummary(
            summary=summary_text,
            event_count=len(old),
        )
        return [summary_event] + recent

    async def _summarize(self, events: list[BaseEvent]) -> str:
        """Use LLM to summarize events."""
        client = self._config.create()
        prompt_event = ModelRequest(
            content="Summarize the following conversation history concisely, "
            "preserving key decisions, findings, and context:\n\n"
            + "\n".join(str(e) for e in events)
        )
        response = await client(
            [prompt_event],
            Context(MemoryStream()),
            tools=[],
            response_schema=None,
        )
        return response.content or ""
```

**CompactionSummary event:**

```python
class CompactionSummary(BaseEvent):
    """Synthetic event replacing a sequence of compacted events.

    Created by SummarizeCompact (and similar strategies) to preserve
    context when old events are dropped. The summary is formatted
    by assembly policies for LLM consumption.
    """

    summary: str
    event_count: int  # How many events were summarized
```

**Trigger configuration:**

Compaction triggers are deterministic and measurable:

```python
@dataclass(slots=True)
class CompactTrigger:
    """Deterministic conditions for triggering compaction.

    Compaction fires when ANY condition is exceeded.
    """

    max_events: int = 0  # Compact when event count exceeds this. 0 = disabled.
    max_tokens: int = 0  # Compact when estimated token count exceeds this. 0 = disabled.
    chars_per_token: int = 4  # For token estimation.
```

The `CompactionMiddleware` (internal to Actor) checks these conditions after each agent turn. When exceeded, compaction fires.

```python
class _CompactionMiddleware(BaseMiddleware):
    """Triggers compaction after agent turns when thresholds are exceeded."""

    def __init__(self, event, context, *, strategy, store, trigger):
        super().__init__(event, context)
        self._strategy = strategy
        self._store = store
        self._trigger = trigger

    async def on_turn(self, call_next, event, context):
        result = await call_next(event, context)

        events = list(await context.stream.history.get_events())

        should_compact = False
        if self._trigger.max_events > 0 and len(events) > self._trigger.max_events:
            should_compact = True
        if self._trigger.max_tokens > 0:
            estimated = sum(len(str(e)) for e in events) // self._trigger.chars_per_token
            if estimated > self._trigger.max_tokens:
                should_compact = True

        if should_compact:
            compacted = await self._strategy.compact(events, context, self._store)
            await context.stream.history.replace(compacted)

        return result
```

---

### AggregateStrategy

Aggregation organizes knowledge for sustained performance over time. Unlike compaction (which reduces the active context), aggregation creates new organized knowledge in the store. It is how an actor builds long-term memory from short-term experience.

**Protocol:**

```python
@runtime_checkable
class AggregateStrategy(Protocol):
    """Organizes knowledge for sustained performance.

    Aggregation extracts structured knowledge from raw events and writes
    it to the knowledge store. This is the knowledge-organizing operation —
    triggered at deterministic milestones to maintain actor effectiveness.

    Unlike compaction (which removes), aggregation creates.
    """

    async def aggregate(
        self,
        events: list[BaseEvent],
        context: Context,
        store: KnowledgeStore,
    ) -> None:
        """Extract and store knowledge.

        Args:
            events: Current stream history.
            context: Execution context.
            store: Actor's knowledge store to write into.
        """
        ...
```

**Built-in implementations:**

```python
class ConversationSummaryAggregate:
    """Summarize conversation and write to /memory/conversations/.

    Creates a per-conversation summary in the knowledge store.
    Costs one LLM call per aggregation.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    async def aggregate(self, events, context, store):
        if not events:
            return
        summary = await self._summarize(events)
        stream_id = str(context.stream.id)
        await store.write(f"/memory/conversations/{stream_id}.md", summary)

    async def _summarize(self, events: list[BaseEvent]) -> str:
        client = self._config.create()
        prompt_event = ModelRequest(
            content="Summarize this conversation. Include key decisions, "
            "findings, outcomes, and any unfinished work:\n\n"
            + "\n".join(str(e) for e in events)
        )
        response = await client(
            [prompt_event],
            Context(MemoryStream()),
            tools=[],
            response_schema=None,
        )
        return response.content or ""
```

```python
class WorkingMemoryAggregate:
    """Update /memory/working.md with latest context.

    Reads existing working memory, merges with new events, writes
    updated working memory. The actor starts each new conversation
    with this as context (via WorkingMemoryPolicy).

    Costs one LLM call per aggregation.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    async def aggregate(self, events, context, store):
        if not events:
            return
        existing = await store.read("/memory/working.md") or ""
        updated = await self._merge(existing, events)
        await store.write("/memory/working.md", updated)

    async def _merge(self, existing: str, events: list[BaseEvent]) -> str:
        client = self._config.create()
        prompt = (
            "You maintain an actor's working memory. Update it based on "
            "the new conversation below. Preserve important existing context. "
            "Remove outdated information. Keep it concise and actionable.\n\n"
            f"## Current Working Memory\n{existing or '(empty)'}\n\n"
            f"## New Conversation\n"
            + "\n".join(str(e) for e in events)
        )
        response = await client(
            [ModelRequest(content=prompt)],
            Context(MemoryStream()),
            tools=[],
            response_schema=None,
        )
        return response.content or existing
```

**Trigger configuration:**

```python
@dataclass(slots=True)
class AggregateTrigger:
    """Deterministic conditions for triggering aggregation.

    Multiple conditions can be set. Each fires independently.
    """

    every_n_turns: int = 0   # Aggregate every N LLM turns. 0 = disabled.
    every_n_events: int = 0  # Aggregate every N new events since last aggregation. 0 = disabled.
    on_end: bool = True      # Aggregate when conversation ends (in Actor._execute finally block).
```

The Actor tracks turn count and event count. When a threshold is hit, aggregation fires. `on_end` fires in the `finally` block of `_execute()`.

```python
class _AggregationMiddleware(BaseMiddleware):
    """Triggers aggregation after agent turns when thresholds are exceeded."""

    def __init__(self, event, context, *, strategy, store, trigger):
        super().__init__(event, context)
        self._strategy = strategy
        self._store = store
        self._trigger = trigger
        self._turn_count = 0
        self._last_aggregate_event_count = 0

    async def on_turn(self, call_next, event, context):
        result = await call_next(event, context)
        self._turn_count += 1

        events = list(await context.stream.history.get_events())

        should_aggregate = False
        if self._trigger.every_n_turns > 0 and self._turn_count % self._trigger.every_n_turns == 0:
            should_aggregate = True
        if self._trigger.every_n_events > 0:
            new_events = len(events) - self._last_aggregate_event_count
            if new_events >= self._trigger.every_n_events:
                should_aggregate = True

        if should_aggregate:
            await self._strategy.aggregate(events, context, self._store)
            self._last_aggregate_event_count = len(events)

        return result
```

---

### Topic Events

New event types for the pub/sub system on Hub.

```python
class TopicMessage(BaseEvent):
    """A message published to a network topic."""

    topic: str
    sender: str
    message: str
    data: dict = Field(default_factory=dict)


class TopicSubscription(BaseEvent):
    """Emitted when an actor subscribes to a topic."""

    actor: str
    topic: str


class TopicUnsubscription(BaseEvent):
    """Emitted when an actor unsubscribes from a topic."""

    actor: str
    topic: str
```

These are standard `BaseEvent` subclasses. They flow through the Hub's stream and are visible to system plugins and observers.

---

## Layer 3: Building Blocks

### Assembler

The assembler composes `AssemblyPolicy` instances into a middleware that transforms context before each LLM call. It replaces the old `ContextHarness` entirely.

**AssemblyPolicy protocol:**

```python
@runtime_checkable
class AssemblyPolicy(Protocol):
    """Transforms context before each LLM invocation.

    A policy receives (prompts, events) and returns modified (prompts, events).
    Policies compose: each sees the output of the previous.

    Policies are pure transforms with one exception: they may read from
    KnowledgeStore or Hub (via context.dependencies). They must not have
    side effects on the stream.

    Transparency: policies can optionally annotate what they did by
    appending to prompts. This helps the LLM understand its context.
    """

    name: str

    async def apply(
        self,
        prompts: list[str],
        events: list[BaseEvent],
        context: Context,
    ) -> tuple[list[str], list[BaseEvent]]:
        """Transform prompts and events. Return modified copies."""
        ...
```

**AssemblerMiddleware:**

```python
class AssemblerMiddleware(BaseMiddleware):
    """Runs assembly policies before each LLM call.

    Sits at the outermost position in the middleware chain. Runs all
    policies in order, transforming (prompts, events) before they
    reach the LLM client.

    Middleware ordering in Actor._execute():
        1. AssemblerMiddleware(policies)     — outermost: assembles context
        2. SignalInjectionMiddleware(queue)   — injects alerts
        3. CompactionMiddleware              — triggers compaction after turns
        4. AggregationMiddleware             — triggers aggregation after turns
        5. User-provided middleware           — logging, retry, etc.
        6. LLM client call                    — innermost: sends to model
    """

    def __init__(self, event, context, *, policies: list[AssemblyPolicy]):
        super().__init__(event, context)
        self._policies = policies

    async def on_llm_call(self, call_next, events, context):
        prompts = list(context.prompt)
        event_list = list(events)

        for policy in self._policies:
            prompts, event_list = await policy.apply(prompts, event_list, context)

        original_prompt = context.prompt
        context.prompt = prompts
        try:
            return await call_next(event_list, context)
        finally:
            context.prompt = original_prompt
```

The prompt swap is temporary — restored in `finally`. This composes correctly with `SignalInjectionMiddleware` (which also temporarily modifies prompts) because middleware nesting handles cleanup in reverse order.

**Policy ordering.** Policies compose left-to-right. A `SlidingWindowPolicy` before `EpisodicMemoryPolicy` means episodic memory operates on already-trimmed events. The assembler enforces ordering awareness via a `validate_order()` class method that warns at construction time when known-problematic orderings are detected:

```python
class AssemblerMiddleware(BaseMiddleware):
    # ...

    @staticmethod
    def validate_order(policies: list[AssemblyPolicy]) -> list[str]:
        """Check for known-problematic policy orderings. Returns warnings."""
        warnings = []
        names = [p.name for p in policies]

        # Reduction policies (sliding_window, token_budget) should come AFTER
        # injection policies (episodic_memory, working_memory, topic_inbox)
        # because injections add context that reduction should then trim.
        reduction = {"sliding_window", "token_budget"}
        injection = {"episodic_memory", "working_memory", "topic_inbox"}

        for i, name in enumerate(names):
            if name in reduction:
                for j in range(i + 1, len(names)):
                    if names[j] in injection:
                        warnings.append(
                            f"Policy '{name}' (index {i}) runs before '{names[j]}' (index {j}). "
                            f"Injection policies should generally run before reduction policies "
                            f"so injected context is included in the reduction budget."
                        )
        return warnings
```

Actor calls `validate_order()` in `__init__` and logs warnings. It does not raise — ordering is the user's choice, and there are valid reasons to override the default.

---

### Built-in Assembly Policies

#### ConversationPolicy

Replaces `ConversationHarness`. Filters events to only conversation and tool events.

```python
class ConversationPolicy:
    """Only conversation and tool events reach the LLM.

    This is the default policy. It preserves the current Agent behavior exactly.
    """

    name = "conversation"

    _TYPES = (
        ModelRequest,
        ModelResponse,
        ToolCallEvent,
        ToolCallsEvent,
        ToolResultEvent,
        ToolResultsEvent,
        ToolErrorEvent,
        CompactionSummary,
    )

    async def apply(self, prompts, events, context):
        filtered = [e for e in events if isinstance(e, self._TYPES)]
        return prompts, filtered
```

Note: `CompactionSummary` is included — compacted summaries must be visible to the LLM.

#### NetworkPolicy

Replaces `NetworkHarness`. Includes network events in LLM context with formatting.

```python
class NetworkPolicy:
    """Includes network events in the LLM context.

    The actor sees delegation results, signals, scheduler events,
    and topic messages alongside conversation — enabling
    network-aware reasoning.
    """

    name = "network"

    async def apply(self, prompts, events, context):
        network_types = ConversationPolicy._TYPES + (
            DelegationResult,
            Signal,
            SchedulerTriggerFired,
            TopicMessage,
        )
        filtered = [e for e in events if isinstance(e, network_types)]

        # Format network events
        formatted = []
        for event in filtered:
            fmt = self._format(event)
            if fmt is not None:
                formatted.append(FormattedEvent(content=fmt, original=event))
            else:
                formatted.append(event)
        return prompts, formatted

    def _format(self, event):
        if isinstance(event, Signal):
            level = event.severity.upper() if isinstance(event.severity, str) else str(event.severity)
            return f"[SIGNAL/{level}] ({event.source}): {event.message}"
        if isinstance(event, DelegationResult):
            return f"[DELEGATION RESULT] {event.source} → {event.target}: {event.result}"
        if isinstance(event, SchedulerTriggerFired):
            return f"[SCHEDULED] Trigger '{event.watch_id}' fired for {event.target}"
        if isinstance(event, TopicMessage):
            return f"[TOPIC/{event.topic}] {event.sender}: {event.message}"
        if isinstance(event, CompactionSummary):
            return f"[CONTEXT SUMMARY] ({event.event_count} earlier events)\n{event.summary}"
        return None
```

#### SlidingWindowPolicy

```python
class SlidingWindowPolicy:
    """Keep the last N events. Drop older events.

    Optional transparency: injects a note about how many events were omitted.
    """

    name = "sliding_window"

    def __init__(self, max_events: int, transparent: bool = False):
        self._max = max_events
        self._transparent = transparent

    async def apply(self, prompts, events, context):
        total = len(events)
        if total <= self._max:
            return prompts, events
        trimmed = events[-self._max :]
        if self._transparent:
            prompts = prompts + [
                f"[{self.name}] Showing last {len(trimmed)} of {total} events."
            ]
        return prompts, trimmed
```

#### TokenBudgetPolicy

```python
class TokenBudgetPolicy:
    """Keep events within a token budget.

    Estimates tokens by character count. Retains most recent events first.
    """

    name = "token_budget"

    def __init__(self, max_tokens: int, chars_per_token: int = 4, transparent: bool = False):
        self._max_chars = max_tokens * chars_per_token
        self._transparent = transparent

    async def apply(self, prompts, events, context):
        total_chars = sum(len(str(e)) for e in events)
        if total_chars <= self._max_chars:
            return prompts, events

        # Retain from the end, fitting within budget
        retained = []
        budget = self._max_chars
        for event in reversed(events):
            cost = len(str(event))
            if budget - cost < 0 and retained:
                break
            retained.append(event)
            budget -= cost
        retained.reverse()

        if self._transparent:
            prompts = prompts + [
                f"[{self.name}] Showing {len(retained)} of {len(events)} events (token budget)."
            ]
        return prompts, retained
```

#### EpisodicMemoryPolicy

```python
class EpisodicMemoryPolicy:
    """Injects past conversation summaries from the knowledge store.

    Reads /memory/conversations/ and injects the most recent summaries
    into the system prompt. This gives the actor context about past episodes.
    """

    name = "episodic_memory"

    def __init__(self, max_episodes: int = 5, transparent: bool = True):
        self._max = max_episodes
        self._transparent = transparent

    async def apply(self, prompts, events, context):
        store = context.dependencies.get(KnowledgeStore)
        if not store:
            return prompts, events

        entries = await store.list("/memory/conversations/")
        if not entries:
            return prompts, events

        # Read most recent summaries
        recent = entries[-self._max :]
        summaries = []
        for entry in recent:
            content = await store.read(f"/memory/conversations/{entry}")
            if content:
                summaries.append(content)

        if summaries:
            block = "## Past Conversations\n\n" + "\n\n---\n\n".join(summaries)
            prompts = prompts + [block]
            if self._transparent:
                prompts = prompts + [
                    f"[{self.name}] Injected {len(summaries)} past conversation summaries."
                ]

        return prompts, events
```

#### WorkingMemoryPolicy

```python
class WorkingMemoryPolicy:
    """Injects /memory/working.md from the knowledge store.

    Working memory is the actor's persistent state — updated by
    AggregateStrategy between conversations. This policy injects
    it as system prompt context so the actor has continuity.
    """

    name = "working_memory"

    async def apply(self, prompts, events, context):
        store = context.dependencies.get(KnowledgeStore)
        if not store:
            return prompts, events

        content = await store.read("/memory/working.md")
        if content:
            prompts = prompts + [f"## Working Memory\n\n{content}"]

        return prompts, events
```

#### TopicInboxPolicy

```python
class TopicInboxPolicy:
    """Injects unread topic messages into the actor's context.

    Reads new messages from all subscribed topics via the Hub.
    Injects them as system prompt context. Advances the read cursor
    so messages are not re-injected.
    """

    name = "topic_inbox"

    def __init__(self, hub: Hub, actor_name: str, max_messages: int = 50):
        self._hub = hub
        self._actor = actor_name
        self._max = max_messages

    async def apply(self, prompts, events, context):
        # Collect new messages from all subscribed topics
        all_messages: list[TopicMessage] = []
        subscriptions = self._hub.subscriptions_for(self._actor)

        for topic in subscriptions:
            messages = await self._hub.read_topic(self._actor, topic)
            all_messages.extend(messages)

        if not all_messages:
            return prompts, events

        # Apply limit (newest first)
        if len(all_messages) > self._max:
            all_messages = all_messages[-self._max :]

        # Format as prompt block
        lines = ["## Network Messages\n"]
        for msg in all_messages:
            lines.append(f"- **[{msg.topic}]** {msg.sender}: {msg.message}")
        prompts = prompts + ["\n".join(lines)]

        return prompts, events
```

---

### Hub Extensions

The Hub gains topic management for pub/sub communication.

```python
class Hub:
    def __init__(self, ...):
        # Existing fields...

        # NEW: topic management
        self._topics: dict[str, list[TopicMessage]] = {}      # topic → messages
        self._subscriptions: dict[str, set[str]] = {}          # topic → actor names
        self._cursors: dict[tuple[str, str], int] = {}         # (actor, topic) → last-read index

    # ------------------------------------------------------------------
    # Topic management
    # ------------------------------------------------------------------

    async def publish(self, sender: str, topic: str, message: str, data: dict | None = None) -> None:
        """Publish a message to a topic."""
        msg = TopicMessage(topic=topic, sender=sender, message=message, data=data or {})
        self._topics.setdefault(topic, []).append(msg)
        await self._emit(msg)

    async def subscribe_topic(self, actor_name: str, topic: str) -> None:
        """Subscribe an actor to a topic. Cursor starts at current end (no replay)."""
        self._subscriptions.setdefault(topic, set()).add(actor_name)
        self._cursors[(actor_name, topic)] = len(self._topics.get(topic, []))
        await self._emit(TopicSubscription(actor=actor_name, topic=topic))

    async def unsubscribe_topic(self, actor_name: str, topic: str) -> None:
        """Unsubscribe an actor from a topic."""
        self._subscriptions.get(topic, set()).discard(actor_name)
        self._cursors.pop((actor_name, topic), None)
        await self._emit(TopicUnsubscription(actor=actor_name, topic=topic))

    async def read_topic(self, actor_name: str, topic: str) -> list[TopicMessage]:
        """Read new messages from a topic since last read. Advances cursor."""
        cursor = self._cursors.get((actor_name, topic), 0)
        messages = self._topics.get(topic, [])
        new_messages = messages[cursor:]
        self._cursors[(actor_name, topic)] = len(messages)
        return new_messages

    async def list_topics(self) -> list[str]:
        """List all active topics."""
        return list(self._topics.keys())

    def subscriptions_for(self, actor_name: str) -> list[str]:
        """List topics an actor is subscribed to."""
        return [
            topic
            for topic, actors in self._subscriptions.items()
            if actor_name in actors
        ]
```

For production (distributed Hub), topic storage moves from in-memory dicts to `StateStore` (or a dedicated `TopicStore` protocol). The interface stays the same.

**Network tools extension:**

Hub's `_build_network_tools()` gains new actions in the grouped tool:

```python
def _build_network_tools(self, caller: str = "") -> list[Tool]:
    hub = self

    @tool
    async def network(
        action: str,
        target: str = "",
        topic: str = "",
        message: str = "",
    ) -> str:
        """Communicate over the agent network.

        Actions:
            discover   - Find agents. target=capability filter (optional).
            request    - Delegate task to agent. target=agent name, message=task description.
            publish    - Publish to topic. topic=topic name, message=content.
            subscribe  - Subscribe to a topic. topic=topic name.
            topics     - List all active topics.
            query      - Read from another actor's knowledge. target=agent name, message=path.
            query_list - List another actor's knowledge entries. target=agent name, message=path (default /).
        """
        if action == "discover":
            agents = await hub.discover(target)
            infos = [a for a in agents if a.name != caller]
            if not infos:
                return "No other agents found."
            lines = []
            for a in infos:
                caps = ", ".join(a.capabilities) if a.capabilities else "general"
                desc = f" - {a.description}" if a.description else ""
                lines.append(f"- {a.name} [{caps}]{desc}")
            return "\n".join(lines)

        elif action == "request":
            if not target:
                return "Error: target is required for request action."
            if target == caller:
                return "Error: cannot delegate to yourself."
            if not message:
                return "Error: message is required for request action."
            return await hub._delegate(target, message, source=caller)

        elif action == "publish":
            if not topic:
                return "Error: topic is required for publish action."
            if not message:
                return "Error: message is required for publish action."
            await hub.publish(caller, topic, message)
            return f"Published to topic '{topic}'."

        elif action == "subscribe":
            if not topic:
                return "Error: topic is required for subscribe action."
            await hub.subscribe_topic(caller, topic)
            return f"Subscribed to topic '{topic}'."

        elif action == "topics":
            topics = await hub.list_topics()
            if not topics:
                return "No active topics."
            return "\n".join(f"- {t}" for t in topics)

        elif action == "query":
            if not target or not message:
                return "Error: target (agent name) and message (path) required for query."
            content = await hub.query_knowledge(caller, target, message)
            if content is None:
                return f"Not accessible: {target}:{message} (not found or not exposed)."
            return content

        elif action == "query_list":
            if not target:
                return "Error: target (agent name) required for query_list."
            path = message or "/"
            entries = await hub.list_knowledge(caller, target, path)
            if entries is None:
                return f"Not accessible: {target}:{path} (not found or not exposed)."
            if not entries:
                return f"Empty: {target}:{path}"
            return "\n".join(entries)

        else:
            return f"Unknown action: {action}. Available: discover, request, publish, subscribe, topics, query, query_list."

    return [network]
```

---

### Actor Extensions

Actor ties the harness together. Changes are additive to `_execute()`.

**Constructor:**

```python
class Actor(Agent):
    def __init__(
        self,
        name: str,
        prompt: PromptType | Iterable[PromptType] = (),
        *,
        config: ModelConfig | None = None,

        # Existing
        observers: Iterable[Observer] = (),
        signal_policy: SignalPolicy | None = None,
        hitl_hook: HumanHook | None = None,
        task_config: ModelConfig | None = None,
        task_prompt: str = "You are a task agent. Complete the assigned task thoroughly and concisely. Return only the result.",
        tools: Iterable[Callable[..., Any] | Tool] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,

        # NEW: Agent Harness
        knowledge_store: KnowledgeStore | None = None,
        bootstrap: StoreBootstrap | None = None,
        assembly: Iterable[AssemblyPolicy] = (),
        compact: CompactStrategy | None = None,
        compact_trigger: CompactTrigger | None = None,
        aggregate: AggregateStrategy | None = None,
        aggregate_trigger: AggregateTrigger | None = None,
    ) -> None:
        super().__init__(...)
        # Existing
        self._observers = list(observers)
        self._signal_policy = signal_policy or InjectToPrompt()
        self._task_config = task_config or config
        self._task_prompt = task_prompt

        # NEW
        self._knowledge_store = knowledge_store
        self._bootstrap = bootstrap
        self._policies = list(assembly) if assembly else [ConversationPolicy()]
        self._compact_strategy = compact
        self._compact_trigger = compact_trigger or CompactTrigger()
        self._aggregate_strategy = aggregate
        self._aggregate_trigger = aggregate_trigger or AggregateTrigger()

        # Validate policy ordering
        warnings = AssemblerMiddleware.validate_order(self._policies)
        if warnings:
            import logging
            logger = logging.getLogger(__name__)
            for w in warnings:
                logger.warning("Assembly policy ordering: %s", w)
```

No `harness` parameter.

**Tools:**

```python
def _build_knowledge_tool(self) -> list[Tool]:
    """Build the knowledge tool (typed action group)."""
    store = self._knowledge_store

    @tool
    async def knowledge(action: str, path: str = "/", content: str = "") -> str:
        """Manage your knowledge store.

        Actions:
            read   - Read file at path.
            write  - Write content to path.
            list   - List entries at path.
            delete - Delete file at path.
        """
        if action == "read":
            result = await store.read(path)
            return result if result is not None else f"Not found: {path}"

        elif action == "write":
            if not content:
                return "Error: content is required for write action."
            await store.write(path, content)
            return f"Written to {path}"

        elif action == "list":
            entries = await store.list(path)
            if not entries:
                return f"Empty: {path}"
            # Check for SKILL.md
            skill = await store.read(f"{path.rstrip('/')}/SKILL.md")
            listing = "\n".join(entries)
            if skill:
                listing = f"{skill}\n---\n{listing}"
            return listing

        elif action == "delete":
            await store.delete(path)
            return f"Deleted: {path}"

        else:
            return f"Unknown action: {action}. Available: read, write, list, delete."

    return [knowledge]


def _build_memory_tool(self) -> list[Tool]:
    """Build the memory management tool (typed action group)."""
    actor = self

    @tool
    async def memory(action: str, ctx: Context) -> str:
        """Manage your memory and context.

        Actions:
            compact   - Compact conversation history to free context space.
            summarize - Aggregate current context into knowledge store.
        """
        if action == "compact":
            if not actor._compact_strategy:
                return "Compaction not configured."
            events = list(await ctx.stream.history.get_events())
            compacted = await actor._compact_strategy.compact(
                events, ctx, actor._knowledge_store
            )
            await ctx.stream.history.replace(compacted)
            return f"Compacted: {len(events)} events → {len(compacted)} events."

        elif action == "summarize":
            if not actor._aggregate_strategy or not actor._knowledge_store:
                return "Aggregation not configured."
            events = list(await ctx.stream.history.get_events())
            await actor._aggregate_strategy.aggregate(
                events, ctx, actor._knowledge_store
            )
            return "Knowledge store updated."

        else:
            return f"Unknown action: {action}. Available: compact, summarize."

    return [memory]
```

**_execute() flow:**

```python
async def _execute(self, event, *, context, client, additional_tools=(), additional_middleware=(), response_schema=omit):
    spawn_tools = self._build_spawn_tools()

    # NEW: bootstrap knowledge store on first use
    if self._knowledge_store:
        if not await self._knowledge_store.exists("/.initialized"):
            bootstrap = self._bootstrap or DefaultBootstrap()
            await bootstrap.bootstrap(self._knowledge_store, self.name)

    # NEW: harness tools
    knowledge_tools = self._build_knowledge_tool() if self._knowledge_store else []
    memory_tools = self._build_memory_tool() if (self._compact_strategy or self._aggregate_strategy) else []

    # NEW: make knowledge store available via DI
    if self._knowledge_store:
        context.dependencies[KnowledgeStore] = self._knowledge_store

    # Signal collection (existing, unchanged)
    signal_queue: list[Signal] = []
    _delivered_ids: set[int] = set()
    async def _collect_signal(signal: Signal) -> None:
        if id(signal) not in _delivered_ids:
            signal_queue.append(signal)
    signal_sub = context.stream.where(Signal).subscribe(_collect_signal)

    # Attach observers (existing, unchanged)
    for obs in self._observers:
        obs.attach(context.stream, context)
        await context.send(ObserverStarted(name=obs.name))

    # Build middleware chain
    assembler_mw = _AssemblerMiddlewareFactory(self._policies)      # CHANGED: replaces harness
    signal_mw = _SignalInjectionFactory(signal_queue, self._signal_policy, _delivered_ids)

    harness_middleware: list[MiddlewareFactory] = [assembler_mw, signal_mw]

    # NEW: compaction middleware
    if self._compact_strategy:
        harness_middleware.append(
            _CompactionMiddlewareFactory(
                self._compact_strategy,
                self._knowledge_store,
                self._compact_trigger,
            )
        )

    # NEW: aggregation middleware (for every_n_turns / every_n_events triggers)
    if self._aggregate_strategy and self._knowledge_store:
        trigger = self._aggregate_trigger
        if trigger.every_n_turns > 0 or trigger.every_n_events > 0:
            harness_middleware.append(
                _AggregationMiddlewareFactory(
                    self._aggregate_strategy,
                    self._knowledge_store,
                    trigger,
                )
            )

    try:
        return await super()._execute(
            event,
            context=context,
            client=client,
            additional_tools=(
                list(additional_tools) + spawn_tools + knowledge_tools + memory_tools
            ),
            additional_middleware=harness_middleware + list(additional_middleware),
            response_schema=response_schema,
        )
    finally:
        # Detach observers (existing, unchanged)
        for obs in self._observers:
            try:
                obs.detach()
            except Exception:
                logger.exception("Failed to detach observer %s", obs.name)
            finally:
                with suppress(Exception):
                    await context.send(ObserverCompleted(name=obs.name))
        context.stream.unsubscribe(signal_sub)

        # NEW: on_end aggregation
        if (
            self._aggregate_strategy
            and self._knowledge_store
            and self._aggregate_trigger.on_end
        ):
            try:
                events = list(await context.stream.history.get_events())
                await self._aggregate_strategy.aggregate(
                    events, context, self._knowledge_store
                )
            except Exception:
                logger.exception("Aggregation failed for %s", self.name)

        # NEW: persist event log
        if self._knowledge_store:
            try:
                events = list(await context.stream.history.get_events())
                await EventLogWriter(self._knowledge_store).persist(
                    context.stream.id, events
                )
            except Exception:
                logger.exception("Event log persistence failed for %s", self.name)
```

---

## Built-in Tools Summary

3 tools, each a typed action group:

| Tool | Actions | Available When |
|------|---------|----------------|
| `knowledge` | `read`, `write`, `list`, `delete` | Actor has `knowledge_store` |
| `network` | `discover`, `request`, `publish`, `subscribe`, `topics`, `query`, `query_list` | Actor invoked via `hub.ask()` |
| `memory` | `compact`, `summarize` | Actor has compact or aggregate strategy |

Plus existing tools (unchanged):

| Tool | Actions | Available When |
|------|---------|----------------|
| `spawn_task` | (single function) | Always |
| `spawn_tasks` | (single function) | Always |

Total: 5 tools maximum. 3 of them are typed action groups.

---

## The Three Scenarios

### Scenario 1: Single Actor, Long Context Window (Hot Memory)

```python
actor = Actor(
    "researcher",
    prompt="You are a research agent. Use your knowledge store to save findings.",
    config=config,
    knowledge_store=MemoryKnowledgeStore(),
    assembly=[
        ConversationPolicy(),
        WorkingMemoryPolicy(),
    ],
    compact=TailWindowCompact(target=100),
    compact_trigger=CompactTrigger(max_events=150),
)

reply = await actor.ask("Research quantum computing trends for the next hour")
```

**What happens:**
1. Actor starts. Gets `knowledge` and `memory` tools. Assembly policies: conversation filter + working memory injection.
2. Each LLM call: `ConversationPolicy` filters to conversation events. `WorkingMemoryPolicy` injects `/memory/working.md` (empty on first run).
3. Actor uses `knowledge(action="write", path="/findings/qc-trends.md", content="...")` to save findings.
4. After 150 events: `CompactionMiddleware` fires `TailWindowCompact` → keeps last 100 events.
5. Actor uses `memory(action="compact")` explicitly if it notices context getting long before threshold.
6. Actor reads past findings via `knowledge(action="read", path="/findings/qc-trends.md")`.

### Scenario 2: Single Actor, Multiple Streams (Episodic Memory)

```python
store = MemoryKnowledgeStore()  # Shared across conversations

actor = Actor(
    "assistant",
    prompt="You are a helpful assistant with memory across conversations.",
    config=config,
    knowledge_store=store,
    assembly=[
        ConversationPolicy(),
        WorkingMemoryPolicy(),
        EpisodicMemoryPolicy(max_episodes=3),
    ],
    aggregate=ConversationSummaryAggregate(config=config),
    aggregate_trigger=AggregateTrigger(on_end=True),
)

# Conversation 1
reply1 = await actor.ask("My name is Alice and I'm working on project X")
# on_end: ConversationSummaryAggregate writes /memory/conversations/{id1}.md
# EventLogWriter persists /log/{id1}.jsonl

# Conversation 2
reply2 = await actor.ask("What was I working on?")
# EpisodicMemoryPolicy reads /memory/conversations/ → injects summary
# LLM sees the summary → "You mentioned working on project X"
```

**What happens:**
1. Conversation 1: normal execution. On end, aggregation creates conversation summary. Event log persisted.
2. Conversation 2: fresh stream, same store. `EpisodicMemoryPolicy` reads past summaries, injects into prompts. LLM has context about past episodes.
3. Working memory accumulates across conversations (if `WorkingMemoryAggregate` is also configured).

### Scenario 3: Multi-Actor Context Sharing

```python
hub = Hub()

researcher = Actor(
    "researcher",
    config=config,
    knowledge_store=MemoryKnowledgeStore(),
    assembly=[ConversationPolicy(), NetworkPolicy()],
)
analyst = Actor(
    "analyst",
    config=config,
    knowledge_store=MemoryKnowledgeStore(),
    assembly=[ConversationPolicy(), NetworkPolicy(), TopicInboxPolicy(hub, "analyst")],
)

await hub.register(researcher, capabilities=["research"], exposed_paths=["/memory/", "/artifacts/"])
await hub.register(analyst, capabilities=["analysis"])

# Subscribe analyst to findings topic
await hub.subscribe_topic("analyst", "findings")

# Researcher runs and publishes
reply = await hub.ask(researcher, "Research AI safety and publish key findings")
# During execution, researcher calls:
#   network(action="publish", topic="findings", message="Key finding: ...")
# Messages accumulate in Hub's topic store

# Later, analyst runs
reply = await hub.ask(analyst, "Analyze the latest research findings")
# TopicInboxPolicy reads from "findings" topic → injects messages into prompts
# Analyst sees all published findings and analyzes them

# Analyst can also directly query researcher's knowledge store:
#   network(action="query", target="researcher", message="/memory/working.md")
#   network(action="query_list", target="researcher", message="/artifacts/")
# Works because researcher registered with exposed_paths=["/memory/", "/artifacts/"]
```

**Distributed variant:** same code, but:
- Knowledge stores backed by S3/Redis instead of memory
- Hub topic storage backed by StateStore (Redis)
- Cross-process: `Hub.serve()` exposes topic endpoints
- Remote actors use `RemoteAgent`

---

## File Structure

```
autogen/beta/network/
├── primitives/
│   ├── __init__.py
│   ├── watch.py               # Existing
│   ├── signal.py              # Existing
│   ├── priority.py            # Existing
│   ├── envelope.py            # Existing
│   ├── channel.py             # Existing
│   ├── infra.py               # Existing
│   ├── harness.py             # REMOVED (replaced by assembler)
│   ├── knowledge.py           # NEW: KnowledgeStore, MemoryKnowledgeStore,
│   │                          #      StoreBootstrap, DefaultBootstrap,
│   │                          #      EventLogWriter, LockedKnowledgeStore
│   ├── compact.py             # NEW: CompactStrategy, CompactTrigger, CompactionSummary,
│   │                          #      TailWindowCompact, SummarizeCompact
│   └── aggregate.py           # NEW: AggregateStrategy, AggregateTrigger,
│                              #      ConversationSummaryAggregate, WorkingMemoryAggregate
│
├── assembler.py               # NEW (Layer 3): AssemblyPolicy, AssemblerMiddleware
│
├── policies/                  # NEW: Built-in assembly policies
│   ├── __init__.py
│   ├── conversation.py        # ConversationPolicy
│   ├── network.py             # NetworkPolicy
│   ├── sliding_window.py      # SlidingWindowPolicy
│   ├── token_budget.py        # TokenBudgetPolicy
│   ├── episodic_memory.py     # EpisodicMemoryPolicy
│   ├── working_memory.py      # WorkingMemoryPolicy
│   └── topic_inbox.py         # TopicInboxPolicy, TopicOverflow
│
├── actor.py                   # MODIFIED: harness integration, knowledge/memory tools
├── hub.py                     # MODIFIED: topic pub/sub, knowledge queries, exposed_paths
├── events.py                  # MODIFIED: + TopicMessage, TopicSubscription, TopicUnsubscription,
│                              #            CompactionSummary, CompactionCompleted,
│                              #            AggregationCompleted, UnknownEvent
├── observer.py                # Existing, unchanged
├── scheduler.py               # Existing, unchanged
├── topology.py                # Existing, unchanged
├── convenience.py             # Existing, unchanged
├── remote.py                  # Existing, unchanged
│
├── channels/                  # Existing, unchanged
├── observers/                 # Existing, unchanged
└── plugins/                   # Existing, unchanged
```

**Removed:** `primitives/harness.py`

**New files:** `primitives/knowledge.py`, `primitives/compact.py`, `primitives/aggregate.py`, `assembler.py`, `policies/*.py`

**Modified files:** `actor.py`, `hub.py`, `events.py`, `__init__.py`

### Changes to BaseEvent

The existing `BaseEvent` gains `from_dict()` and improved `to_dict()` for round-trip WAL serialization (see Cross-Cutting Concern #7). These are additive — no existing behavior changes.

---

## Public API Additions

```python
from autogen.beta.network import (
    # Existing exports (unchanged)...

    # NEW: Agent Harness — Primitives
    KnowledgeStore, MemoryKnowledgeStore, LockedKnowledgeStore,
    StoreBootstrap, DefaultBootstrap, EventLogWriter,
    CompactStrategy, CompactTrigger, CompactionSummary,
    TailWindowCompact, SummarizeCompact,
    AggregateStrategy, AggregateTrigger,
    ConversationSummaryAggregate, WorkingMemoryAggregate,

    # NEW: Agent Harness — Assembler
    AssemblyPolicy, AssemblerMiddleware,

    # NEW: Agent Harness — Policies
    ConversationPolicy, NetworkPolicy,
    SlidingWindowPolicy, TokenBudgetPolicy,
    EpisodicMemoryPolicy, WorkingMemoryPolicy,
    TopicInboxPolicy, TopicOverflow,

    # NEW: Events
    TopicMessage, TopicSubscription, TopicUnsubscription,
    CompactionCompleted, AggregationCompleted, UnknownEvent,
)
```

**Removed exports:** `ContextHarness`, `ConversationHarness`, `NetworkHarness`, `HarnessMiddleware`, `FormattedEvent`.

Note: `FormattedEvent` is still used internally by `NetworkPolicy` for formatting network events. It moves from a public export to an internal utility in the policies module.

---

## Cross-Cutting Concerns

Seven concerns that span the harness components. Each is designed as a concrete mechanism, not deferred.

---

### 1. Assembly Policy Ordering Validation

Covered in the Assembler section above. The `AssemblerMiddleware.validate_order()` static method checks for known-problematic orderings at Actor construction time and logs warnings. It does not raise — ordering is the user's choice.

---

### 2. Aggregation Cost Visibility

LLM-based strategies (SummarizeCompact, ConversationSummaryAggregate, WorkingMemoryAggregate) make LLM calls that cost money. The framework must make this visible.

**Events:**

```python
class CompactionCompleted(BaseEvent):
    """Emitted on the actor's stream when compaction finishes."""
    actor: str
    strategy: str
    events_before: int
    events_after: int
    llm_calls: int           # 0 for TailWindowCompact, 1 for SummarizeCompact
    usage: dict = Field(default_factory=dict)  # Token usage from LLM calls


class AggregationCompleted(BaseEvent):
    """Emitted on the actor's stream when aggregation finishes."""
    actor: str
    strategy: str
    event_count: int          # Events that were aggregated
    llm_calls: int
    usage: dict = Field(default_factory=dict)
```

These events flow on the actor's stream (visible to observers and the Hub stream if the actor is registered). System plugins can subscribe to track costs across the network.

**Strategy contract:** Every strategy that makes LLM calls must return usage metadata. The protocol methods themselves don't change — the Actor wraps the calls and emits the events:

```python
# In _CompactionMiddleware.on_turn, after compaction:
compacted = await self._strategy.compact(events, context, self._store)
await context.send(CompactionCompleted(
    actor=...,
    strategy=type(self._strategy).__name__,
    events_before=len(events),
    events_after=len(compacted),
    llm_calls=1 if hasattr(self._strategy, '_config') else 0,
))
```

The same pattern applies to aggregation. The Actor emits `AggregationCompleted` after each aggregation call.

**Built-in strategies report usage** by storing it as an instance attribute after each call:

```python
class SummarizeCompact:
    def __init__(self, target: int, config: ModelConfig):
        self._target = target
        self._config = config
        self.last_usage: dict = {}  # Set after each compact()

    async def compact(self, events, context, store):
        # ... summarize ...
        self.last_usage = response.usage if response.usage else {}
        return [summary_event] + recent
```

The middleware reads `strategy.last_usage` when emitting the completion event.

---

### 3. Knowledge Store Bootstrapping

First time an actor runs with a store, the store is empty. The actor has no knowledge of what's available, how the store is organized, or what tools relate to what paths. Bootstrapping solves this.

**Protocol:**

```python
@runtime_checkable
class StoreBootstrap(Protocol):
    """Initializes a knowledge store with a starting structure.

    Called once when an actor first runs with a store. Subsequent
    runs skip bootstrapping (detected via a sentinel file).
    """

    async def bootstrap(self, store: KnowledgeStore, actor_name: str) -> None:
        """Create initial store structure."""
        ...
```

**Default implementation:**

```python
class DefaultBootstrap:
    """Creates the standard knowledge store layout with SKILL.md files."""

    async def bootstrap(self, store: KnowledgeStore, actor_name: str) -> None:
        await store.write("/.initialized", actor_name)

        await store.write("/SKILL.md", (
            f"# {actor_name} Knowledge Store\n\n"
            "This is your persistent knowledge store. Use the `knowledge` tool to manage it.\n\n"
            "## Directories\n"
            "- `/log/` — Conversation history (auto-managed)\n"
            "- `/artifacts/` — External files and data\n"
            "- `/memory/` — Working memory and summaries (auto-managed)\n"
        ))

        await store.write("/log/SKILL.md",
            "Conversation logs. Each file is a JSONL record of one conversation's events. "
            "Auto-populated by the framework after each conversation."
        )

        await store.write("/artifacts/SKILL.md",
            "External data: uploaded files, downloaded content, reference materials. "
            "Write here to store data you want to reference later."
        )

        await store.write("/memory/SKILL.md",
            "Working memory and conversation summaries. "
            "`working.md` contains your current persistent state. "
            "`conversations/` contains per-conversation summaries. "
            "Both are auto-updated by aggregation strategies."
        )
```

**Integration in Actor._execute():**

```python
# At the start of _execute(), before anything else:
if self._knowledge_store:
    if not await self._knowledge_store.exists("/.initialized"):
        bootstrap = self._bootstrap or DefaultBootstrap()
        await bootstrap.bootstrap(self._knowledge_store, self.name)
```

The `/.initialized` sentinel prevents re-bootstrapping. The `bootstrap` parameter on Actor is optional — defaults to `DefaultBootstrap`.

**Actor constructor addition:**

```python
class Actor(Agent):
    def __init__(
        self,
        ...
        knowledge_store: KnowledgeStore | None = None,
        bootstrap: StoreBootstrap | None = None,        # NEW
        ...
    ):
```

Users can pass a custom bootstrap to create domain-specific store layouts. Or `None` for the default.

---

### 4. Topic Back-Pressure

When a topic accumulates many messages and an actor hasn't read them in a while, the `TopicInboxPolicy` must handle the backlog without blowing up the context window.

**Design:**

`TopicInboxPolicy` has a configurable overflow strategy:

```python
class TopicOverflow(str, Enum):
    """How to handle topic message backlogs."""
    NEWEST = "newest"       # Keep the N newest messages. Default.
    OLDEST = "oldest"       # Keep the N oldest messages.
    SUMMARY = "summary"     # Summarize the backlog into one message.


class TopicInboxPolicy:
    name = "topic_inbox"

    def __init__(
        self,
        hub: Hub,
        actor_name: str,
        max_messages: int = 50,
        overflow: TopicOverflow = TopicOverflow.NEWEST,
        summary_config: ModelConfig | None = None,  # Required if overflow=SUMMARY
    ):
        self._hub = hub
        self._actor = actor_name
        self._max = max_messages
        self._overflow = overflow
        self._summary_config = summary_config

    async def apply(self, prompts, events, context):
        subscriptions = self._hub.subscriptions_for(self._actor)
        all_messages: list[TopicMessage] = []

        for topic in subscriptions:
            messages = await self._hub.read_topic(self._actor, topic)
            all_messages.extend(messages)

        if not all_messages:
            return prompts, events

        # Apply overflow strategy
        if len(all_messages) > self._max:
            if self._overflow == TopicOverflow.NEWEST:
                dropped = len(all_messages) - self._max
                all_messages = all_messages[-self._max :]
                overflow_note = f"({dropped} older messages omitted)"
            elif self._overflow == TopicOverflow.OLDEST:
                dropped = len(all_messages) - self._max
                all_messages = all_messages[: self._max]
                overflow_note = f"({dropped} newer messages queued)"
            elif self._overflow == TopicOverflow.SUMMARY:
                summary = await self._summarize(all_messages)
                prompts = prompts + [f"## Network Messages (summarized)\n\n{summary}"]
                return prompts, events
        else:
            overflow_note = ""

        lines = ["## Network Messages\n"]
        if overflow_note:
            lines.append(f"*{overflow_note}*\n")
        for msg in all_messages:
            lines.append(f"- **[{msg.topic}]** {msg.sender}: {msg.message}")
        prompts = prompts + ["\n".join(lines)]

        return prompts, events

    async def _summarize(self, messages: list[TopicMessage]) -> str:
        """Summarize a backlog of topic messages via LLM."""
        client = self._summary_config.create()
        formatted = "\n".join(
            f"[{m.topic}] {m.sender}: {m.message}" for m in messages
        )
        response = await client(
            [ModelRequest(content=f"Summarize these {len(messages)} network messages concisely:\n\n{formatted}")],
            Context(MemoryStream()),
            tools=[],
            response_schema=None,
        )
        return response.content or ""
```

The `SUMMARY` overflow strategy costs one LLM call but preserves all information semantically. `NEWEST` (default) is zero-cost and works for most cases.

---

### 5. Cross-Actor Knowledge Queries

An actor should be able to read from another actor's knowledge store without a full delegation (which runs an LLM call). This is a lightweight read — like a database query vs. an RPC call.

**Design:**

The Hub mediates knowledge queries. Actors must explicitly expose parts of their store.

**Exposure model:**

When an actor registers with the Hub, it can declare which store paths are readable by other actors:

```python
await hub.register(
    researcher,
    capabilities=["research"],
    exposed_paths=["/memory/", "/artifacts/"],  # Other actors can read these
)
```

Paths not in `exposed_paths` are private. If `exposed_paths` is empty or None, nothing is exposed (default — private by default, public by opt-in).

**Hub additions:**

```python
class Hub:
    def __init__(self, ...):
        # ...
        self._exposed_paths: dict[str, list[str]] = {}  # actor → exposed path prefixes

    async def register(self, agent, capabilities=None, description="", exposed_paths=None):
        # ... existing registration ...
        if exposed_paths:
            self._exposed_paths[agent.name] = list(exposed_paths)
        return handle

    async def query_knowledge(
        self, requester: str, target: str, path: str,
    ) -> str | None:
        """Read from another actor's knowledge store.

        Returns content if the path is exposed. Returns None if the
        target has no store, the path is not exposed, or the path
        doesn't exist.
        """
        agent = self._agents.get(target)
        if not agent or not isinstance(agent, Actor):
            return None

        store = agent._knowledge_store
        if not store:
            return None

        # Check exposure
        exposed = self._exposed_paths.get(target, [])
        if not any(path.startswith(prefix) for prefix in exposed):
            return None

        return await store.read(path)

    async def list_knowledge(
        self, requester: str, target: str, path: str = "/",
    ) -> list[str] | None:
        """List entries in another actor's knowledge store.

        Same exposure rules as query_knowledge.
        """
        agent = self._agents.get(target)
        if not agent or not isinstance(agent, Actor):
            return None

        store = agent._knowledge_store
        if not store:
            return None

        exposed = self._exposed_paths.get(target, [])
        if not any(path.startswith(prefix) for prefix in exposed):
            return None

        return await store.list(path)
```

**Network tool extension:**

The `network` tool gains a `query` action:

```python
elif action == "query":
    if not target or not message:
        return "Error: target (agent name) and message (path) required for query."
    content = await hub.query_knowledge(caller, target, message)
    if content is None:
        return f"Not accessible: {target}:{message} (not found or not exposed)."
    return content

elif action == "query_list":
    if not target:
        return "Error: target (agent name) required for query_list."
    path = message or "/"
    entries = await hub.list_knowledge(caller, target, path)
    if entries is None:
        return f"Not accessible: {target}:{path} (not found or not exposed)."
    if not entries:
        return f"Empty: {target}:{path}"
    return "\n".join(entries)
```

Updated tool docstring:

```python
"""Communicate over the agent network.

Actions:
    discover   - Find agents. target=capability filter (optional).
    request    - Delegate task to agent. target=agent name, message=task description.
    publish    - Publish to topic. topic=topic name, message=content.
    subscribe  - Subscribe to a topic. topic=topic name.
    topics     - List all active topics.
    query      - Read from another actor's knowledge store. target=agent name, message=path.
    query_list - List entries in another actor's store. target=agent name, message=path (default /).
"""
```

**Privacy model:** Private by default. Only paths in `exposed_paths` are readable. Actors cannot write to each other's stores through the Hub — only read. To push data, use topics (pub/sub). This gives clear ownership semantics.

---

### 6. Concurrency Safety

Compaction and aggregation both operate on the event list and the knowledge store. Within a single conversation and across conversations sharing a store, operations must not corrupt state.

**Within a single conversation:**

Compaction runs in `_CompactionMiddleware.on_turn()`. Aggregation runs either in `_AggregationMiddleware.on_turn()` or in `Actor._execute()`'s finally block. These are sequential within the middleware chain — `on_turn` calls are not concurrent. The finally-block aggregation runs after the middleware chain completes. No concurrent access within one conversation.

However, compaction modifies stream history (`history.replace()`), and a concurrent `on_llm_call` in the same turn could read stale history. This doesn't happen because `on_turn` wraps the entire turn — the LLM call is inside `call_next(event, context)`, and compaction happens after `call_next` returns.

**Across conversations sharing a store:**

If two conversations with the same actor run concurrently (e.g., the actor is registered with a Hub and receives two delegations simultaneously), both write to the same KnowledgeStore.

The framework provides a `StoreLock` wrapper that serializes write operations:

```python
class LockedKnowledgeStore:
    """Wraps a KnowledgeStore with a Lock for concurrent access safety.

    Reads are not locked (safe for concurrent access on all backends).
    Writes and deletes acquire the lock.
    """

    def __init__(self, store: KnowledgeStore, lock: Lock) -> None:
        self._store = store
        self._lock = lock

    async def read(self, path: str) -> str | None:
        return await self._store.read(path)

    async def write(self, path: str, content: str) -> None:
        acquired = await self._lock.acquire(f"store:write:{path}", ttl=30.0)
        if not acquired:
            raise RuntimeError(f"Failed to acquire write lock for {path}")
        try:
            await self._store.write(path, content)
        finally:
            await self._lock.release(f"store:write:{path}")

    async def list(self, path: str = "/") -> list[str]:
        return await self._store.list(path)

    async def delete(self, path: str) -> None:
        acquired = await self._lock.acquire(f"store:write:{path}", ttl=30.0)
        if not acquired:
            raise RuntimeError(f"Failed to acquire delete lock for {path}")
        try:
            await self._store.delete(path)
        finally:
            await self._lock.release(f"store:write:{path}")

    async def exists(self, path: str) -> bool:
        return await self._store.exists(path)
```

This uses the existing `Lock` primitive (Layer 2). In-memory: `LocalLock`. Distributed: `RedisLock`. Users wrap their store when concurrency is expected:

```python
store = MemoryKnowledgeStore()
lock = LocalLock()
safe_store = LockedKnowledgeStore(store, lock)

actor = Actor("researcher", knowledge_store=safe_store, ...)
```

The framework doesn't force locking — it's opt-in. Single-actor-single-conversation use doesn't need it. Multi-conversation or multi-process use wraps with `LockedKnowledgeStore`.

**Stream history concurrency:** Stream history mutations (compaction's `history.replace()`) only happen within `on_turn`, which is scoped to one conversation's middleware chain. Two concurrent conversations have separate streams, so no conflict on history.

---

### 7. Event Serialization for WAL

WAL persistence requires reliable round-trip serialization for all event types. The `EventLogWriter` serializes events to JSONL. Deserialization must reconstruct the original event types.

**Serialization contract:**

Every `BaseEvent` subclass used in the framework must support:

```python
event.to_dict() -> dict          # Serialize to JSON-compatible dict
EventType.from_dict(d) -> event  # Reconstruct from dict (class method)
```

The existing `BaseEvent` already has `to_dict()` for the Envelope wire format. We need to ensure `from_dict()` exists and handles all event types.

**EventLogWriter with type-tagged serialization:**

```python
class EventLogWriter:
    """Persists stream events to the knowledge store as WAL entries.

    Each event is serialized as a JSON line with a type tag for
    deserialization. Uses the same EventRegistry as Envelope for
    type resolution.
    """

    def __init__(self, store: KnowledgeStore) -> None:
        self._store = store

    async def persist(self, stream_id: StreamId, events: Iterable[BaseEvent]) -> None:
        """Write events as type-tagged JSONL."""
        path = f"/log/{stream_id}.jsonl"
        lines = []
        for event in events:
            record = {
                "type": qualified_name(type(event)),
                "data": event.to_dict(),
            }
            lines.append(json.dumps(record, default=str))
        await self._store.write(path, "\n".join(lines))

    async def load(
        self,
        stream_id: StreamId,
        registry: EventRegistry | None = None,
    ) -> list[BaseEvent]:
        """Load events from a WAL file. Returns typed BaseEvent instances."""
        path = f"/log/{stream_id}.jsonl"
        content = await self._store.read(path)
        if not content:
            return []

        events: list[BaseEvent] = []
        for line in content.strip().split("\n"):
            if not line:
                continue
            record = json.loads(line)
            event_type = record["type"]
            event_data = record["data"]

            # Resolve type using the same mechanism as Envelope.from_dict()
            cls = _resolve_event_type(event_type, registry)
            if cls is not None:
                events.append(cls.from_dict(event_data))
            else:
                # Unknown type — preserve as raw dict wrapped in a generic event
                events.append(UnknownEvent(type_name=event_type, data=event_data))

        return events
```

**UnknownEvent fallback:**

```python
class UnknownEvent(BaseEvent):
    """Placeholder for events whose type cannot be resolved during deserialization.

    Preserves the raw data so nothing is lost. Assembly policies can
    filter these out or format them generically.
    """
    type_name: str
    data: dict = Field(default_factory=dict)
```

**`qualified_name` and `_resolve_event_type`** reuse the same functions from `envelope.py`. The type resolution logic is shared — import from there.

**from_dict() on BaseEvent:**

The existing `BaseEvent` metaclass auto-generates `__init__` from field descriptors. We add a `from_dict` class method that reconstructs from a dict:

```python
class BaseEvent:
    # ... existing metaclass-generated code ...

    @classmethod
    def from_dict(cls, data: dict) -> "BaseEvent":
        """Reconstruct event from serialized dict."""
        # Filter to only fields this class knows about
        fields = {f.name for f in cls._fields}  # _fields from metaclass
        filtered = {k: v for k, v in data.items() if k in fields}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        result = {}
        for f in self._fields:
            value = getattr(self, f.name)
            if isinstance(value, BaseEvent):
                result[f.name] = {"_type": qualified_name(type(value)), **value.to_dict()}
            elif isinstance(value, dict):
                result[f.name] = value
            elif isinstance(value, list):
                result[f.name] = [
                    ({"_type": qualified_name(type(v)), **v.to_dict()} if isinstance(v, BaseEvent) else v)
                    for v in value
                ]
            else:
                result[f.name] = value
        return result
```

**Validation:** The test suite validates round-trip serialization for every event type:

```python
@pytest.mark.parametrize("event", [
    ModelRequest(content="hello"),
    ModelResponse(message=ModelMessage(content="world")),
    ToolCallEvent(call_id="1", name="fn", arguments={"x": 1}),
    DelegationRequest(source="a", target="b", task="t"),
    Signal(source="obs", severity="info", message="m"),
    TopicMessage(topic="t", sender="s", message="m"),
    CompactionSummary(summary="s", event_count=10),
    # ... all event types
])
def test_event_round_trip(event):
    data = event.to_dict()
    reconstructed = type(event).from_dict(data)
    assert reconstructed.to_dict() == data
```

This test is part of the harness test suite and runs for every event type in the framework.
