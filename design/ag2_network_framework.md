# AG2 Network Framework

## Vision

AG2 V2 is a paradigm shift: from research-oriented multi-agent group chat to **production-ready network of distributed autonomous agents**.

The core problem is **agent fragmentation**. Agents today are islands. A local Claude Code agent cannot see GPT agents you built, cannot talk to a service agent you use, and cannot gain context outside of predefined tools. We can train more powerful models, but this will not solve the fragmentation crisis. This is like computers and the internet — a powerful CPU cannot replace the importance of networking.

Existing solutions fall into two camps, both insufficient:

1. **Orchestration frameworks** (LangGraph, CrewAI, AutoGen V1) over-emphasize central control. The internet never orchestrates computers. Complexity in orchestration grows exponentially with agent count. It does not naturally scale.

2. **Communication protocols** (A2A, MCP) simulate how humans interact over networks — fire-and-forget, stateless by default. They lack visibility (silent failures, context drift), have no universal stateful engine standard (distributed state consistency is unguaranteed), and provide no traffic control (every infra feature is built ad-hoc behind the server).

**What's missing is the network infrastructure layer for agents.** Not more orchestration. Not bare wire protocols. The layer between the protocol and the application — the nervous system that enables autonomous agents to discover, communicate, observe, and coordinate at scale.

AG2 V2 builds this layer.

---

## Positioning

**The infrastructure gap:** Orchestration frameworks (LangGraph, CrewAI, OpenAI Agents SDK) treat multi-agent as single-deployment, in-process coordination. Communication protocols (A2A, MCP) define wire formats but leave coordination, state, delivery guarantees, and observability to implementers. Infrastructure projects (AGNTCY) provide discovery and messaging but not agent construction, scheduling, or developer experience.

**AG2 V2 fills the gap** — the runtime layer between wire protocols and application code. Transport-agnostic communication, composable routing, reactive scheduling, and pluggable observability as framework primitives.

AG2's controlled choreography pattern — Hub sets constraints, Actors operate autonomously within them — scales naturally without the exponential complexity growth of rigid orchestration.

**Integration, not competition:**
- **A2A** plugs in as a Channel backend. AG2's Hub routes and monitors; A2A carries the traffic cross-organization.
- **MCP** remains the vertical (agent-to-tool) protocol. AG2 provides horizontal (agent-to-agent) coordination on top.
- **AGNTCY** provides infrastructure middleware (DNS, TLS, identity for agents). AG2 provides the framework layer above it — the two are complementary, not overlapping.
- **Orchestration frameworks** solve single-deployment coordination. AG2 extends this to distributed, cross-framework, and cross-model agent networks.

**AI-first API design:** Coding agents will be primary framework consumers. AG2's API is designed for machine readability: complete type hints, consistent protocol surfaces, self-documenting errors, introspectable state, and convention-based defaults.

---

## Architecture

Five layers, each building on the previous. Each layer can be used independently.

```
┌──────────────────────────────────────────────────────────┐
│  Layer 5: APPLICATION                                     │
│  User code: custom actors, observers, plugins, topologies │
├──────────────────────────────────────────────────────────┤
│  Layer 4: COMPOSITION                                     │
│  Pipeline · Fanout · Conditional · RouteDecision · Network│
├──────────────────────────────────────────────────────────┤
│  Layer 3: BUILDING BLOCKS                                 │
│  Actor · Hub · Observer · Scheduler                       │
├──────────────────────────────────────────────────────────┤
│  Layer 2: PRIMITIVES                                      │
│  Watch · Signal · Priority · Channel · Envelope · Harness │
├──────────────────────────────────────────────────────────┤
│  Layer 1: FRAMEWORK CORE (AG2 Beta)                       │
│  Agent · Stream · Events · Tools · Middleware             │
└──────────────────────────────────────────────────────────┘
```

**Design principle: additive, not invasive.** No changes to Layer 1. Every higher layer composes lower layers via additional tools, middleware, and subscribers.

---

## Layer 1: Framework Core (Existing)

Solid foundation. No changes needed.

| Component | Role |
|-----------|------|
| `Agent` | Execution loop: prompt → LLM → tool calls → loop until done |
| `Stream` / `MemoryStream` | In-memory pub-sub event bus with condition-based filtering |
| `BaseEvent` | Metaclass-driven event base with field-level conditions |
| `Tool` / `FunctionTool` | Protocol-based tool system with dependency injection |
| `BaseMiddleware` | Chainable middleware: on_turn, on_llm_call, on_tool_execution |
| `Context` | Execution context: stream, prompts, dependencies, variables |
| `ModelConfig` / `LLMClient` | Pluggable LLM backends (OpenAI, Anthropic, Gemini, etc.) |

---

## Layer 2: Primitives

Primitives are the fundamental building blocks of the network framework. Each is **independently useful** with **zero dependencies on other primitives**. These are the "nn" layer — low-level, composable, unopinionated.

### Watch

A Watch encapsulates a condition, internal state, check logic, and firing semantics. It is the unified abstraction for all reactive behavior — event monitoring, time-based scheduling, and composite conditions.

**Protocol:**

```python
@runtime_checkable
class Watch(Protocol):
    """A condition that can be armed on a stream and fires a callback when met."""

    id: str

    def arm(self, stream: Stream, callback: WatchCallback) -> None:
        """Start watching. Subscribe to relevant events or start timers."""
        ...

    def disarm(self) -> None:
        """Stop watching. Clean up subscriptions and timers."""
        ...

    @property
    def is_armed(self) -> bool: ...
```

`WatchCallback = Callable[[list[BaseEvent], Context], Awaitable[None]]`

**Implementations:**

| Watch | Behavior | Replaces |
|-------|----------|----------|
| `EventWatch(condition)` | Fire immediately on matching event | `OnEvent` trigger |
| `BatchWatch(n, condition)` | Buffer N matching events, fire with batch | `EveryNEvents` trigger |
| `WindowWatch(seconds, condition)` | Collect events in time window, fire with batch | New |
| `IntervalWatch(seconds)` | Fire periodically | Scheduler `_CronEntry` |
| `DelayWatch(seconds)` | Fire once after delay | Scheduler `_OnceEntry` |
| `CronWatch(expression)` | Fire on cron schedule (e.g. `"0 9 * * MON"`) | New |

**Composite watches:**

| Composite | Behavior |
|-----------|----------|
| `AllOf(w1, w2, ...)` | Fire when ALL sub-watches have fired at least once |
| `AnyOf(w1, w2, ...)` | Fire when ANY sub-watch fires |
| `Sequence(w1, w2, ...)` | Fire when sub-watches fire in order |

**Example:**

```python
from autogen.beta.network import EventWatch, IntervalWatch, AllOf

# Fire on a specific event
watch = EventWatch(DelegationResult)

# Fire every 5 minutes
watch = IntervalWatch(300)

# Fire when both conditions are met
watch = AllOf(
    EventWatch(DelegationResult.target == "monitor"),
    IntervalWatch(60),  # At most once per minute
)
```

### Signal

A Signal is a structured notification produced by an Observer and consumed by an Actor. It carries severity, source identification, and a human/LLM-readable message.

```python
class Severity(str, Enum):
    """Severity levels for signals. Extensible via string values."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class Signal(BaseEvent):
    """Structured notification from an observer."""
    source: str          # Observer name that produced this signal
    severity: str        # Severity level (uses Severity enum values, but accepts any string)
    message: str         # Human/LLM-readable description
    data: dict = Field(default_factory=dict)  # Optional structured payload
```

**Signal delivery policy** — how Signals reach the Actor — is configurable:

```python
class SignalPolicy(Protocol):
    """Defines how signals are delivered to the actor."""
    async def deliver(self, signals: list[Signal], context: Context) -> None: ...

# Default: inject into LLM system prompt (temporary, removed after call)
class InjectToPrompt(SignalPolicy): ...

# Alternative: emit as event on stream (observer pattern)
class EmitToStream(SignalPolicy): ...

# Alternative: call a handler function
class CallHandler(SignalPolicy):
    def __init__(self, handler: Callable[[list[Signal]], Awaitable[None]]): ...
```

**FATAL severity has mechanical consequences.** INFO/WARNING/CRITICAL are advisory — the LLM sees the alert and decides how to respond. FATAL means "stop execution." `InjectToPrompt` enforces this: it emits a `HaltEvent` on the stream. The `SignalInjectionMiddleware` in Actor intercepts FATAL signals before the LLM call and returns a synthetic `ModelResponse` to halt the agent's execution loop — no LLM call is made. Any prompt entries appended by the policy (e.g. non-fatal alerts in the same batch) are cleaned up before returning.

```python
class InjectToPrompt(SignalPolicy):
    """Default signal policy. Injects signals into LLM system prompt.

    - INFO/WARNING/CRITICAL: appended as alert text, removed after LLM call.
    - FATAL: emits HaltEvent on the stream. The SignalInjectionMiddleware
      intercepts FATAL signals before the LLM call and returns a synthetic
      ModelResponse to halt the agent's execution loop — no LLM call is made.

    Non-fatal signals in the same batch as a FATAL are still delivered first
    (appended to prompt) so diagnostic context is not lost, but prompt entries
    are cleaned up before the halt response is returned.
    """
    async def deliver(self, signals: list[Signal], context: Context) -> None:
        fatal = [s for s in signals if s.severity == Severity.FATAL]
        non_fatal = [s for s in signals if s.severity != Severity.FATAL]

        # Always deliver non-fatal signals first so diagnostic context
        # (e.g. warnings explaining why a FATAL occurred) is not lost.
        alert_text = self._format_alerts(non_fatal)
        if alert_text:
            context.prompt.append(alert_text)

        if fatal:
            # Emit HaltEvent — SignalInjectionMiddleware handles the actual halt
            await context.send(HaltEvent(
                reason=f"FATAL: {fatal[0].message}",
                source=fatal[0].source,
                signals=fatal,
            ))
            return
        # (non-fatal prompt entries removed after LLM call by the signal injection middleware)
```

`HaltOnFatal` is also available as a standalone policy that can wrap any other policy — it delivers non-fatal signals first via the inner policy, then halts on FATAL:

```python
class HaltOnFatal(SignalPolicy):
    """Wraps another policy. Delivers non-fatal via inner, then halts on FATAL."""
    def __init__(self, inner: SignalPolicy = InjectToPrompt()): ...
```

### Priority

Priority provides a **mechanism** for ordered delivery and conflict resolution. The framework defines the protocol; developers define the policy.

**Protocol:**

```python
class PriorityScheme(Protocol):
    """Defines how priorities are compared. Developer provides the policy."""

    def compare(self, a: Any, b: Any) -> int:
        """Negative if a < b, zero if equal, positive if a > b."""
        ...

class ConflictResolver(Protocol):
    """Defines how conflicts between competing events are resolved."""

    async def resolve(self, existing: Envelope, incoming: Envelope) -> Envelope:
        """Given two conflicting envelopes, return the winner."""
        ...
```

**Default implementation:**

```python
class DefaultPriority(IntEnum):
    """Three-tier priority levels. Sensible default, override for custom needs."""
    BACKGROUND = 0
    NORMAL = 1
    URGENT = 2

class DefaultPriorityScheme:
    """Default PriorityScheme using the three-tier DefaultPriority levels."""
    BACKGROUND = DefaultPriority.BACKGROUND
    NORMAL = DefaultPriority.NORMAL
    URGENT = DefaultPriority.URGENT

    def compare(self, a, b):
        return int(a) - int(b)

class HighestPriorityWins(ConflictResolver):
    async def resolve(self, existing, incoming):
        if incoming.priority is not None and existing.priority is not None:
            if int(incoming.priority) > int(existing.priority):
                return incoming
        return existing
```

**Custom priority:**

```python
class TicketPriority(PriorityScheme):
    P0_OUTAGE = 100
    P1_CRITICAL = 75
    P2_HIGH = 50
    P3_MEDIUM = 25
    P4_LOW = 0

    def compare(self, a, b):
        return a - b
```

Priority is consumed by Channel (for delivery ordering) and Hub (for routing decisions). It is never on BaseEvent — it lives on the Envelope, at the network level.

### Envelope

An Envelope wraps a BaseEvent with network metadata: addressing, tracing, priority, and delivery requirements. It is what flows through Channels.

```python
@dataclass(slots=True)
class Envelope:
    """Event wrapper with network metadata."""
    event: BaseEvent

    # Addressing
    sender: str
    recipient: str | None = None        # None = broadcast

    # Tracing
    trace_id: str = Field(default_factory=uuid4_hex)        # Groups entire workflow
    correlation_id: str = Field(default_factory=uuid4_hex)   # Groups request-response
    causation_id: str | None = None     # Points to parent envelope

    # Priority & delivery
    priority: Any = None                # Interpreted by PriorityScheme
    timestamp: float = Field(default_factory=time.time)      # Wall clock (for TTL across processes)
    ttl: float | None = None            # Time-to-live in seconds
    requires_ack: bool = False          # Whether delivery must be acknowledged
```

**Trace lineage example:**

```
trace_id: abc123 (entire workflow)
│
├── sender:dispatch → recipient:ems
│   correlation: 001, causation: None
│
├── sender:ems → recipient:hospital
│   correlation: 002, causation: 001
│
└── sender:dispatch → recipient:police
    correlation: 003, causation: None
```

**Design decision:** Envelope wraps BaseEvent rather than replacing it. Core agents that don't need network features continue using `Stream.send(event)`. Network-aware code uses `Channel.send(envelope)`. This preserves backward compatibility.

**Wire format (defined in Phase 1, used by all Channels):**

The Envelope serialization format is a de facto API contract — once HttpChannel ships in Phase 2, changing it breaks interop. Define it now.

```python
class Envelope:
    def to_dict(self) -> dict:
        """Canonical JSON-serializable representation."""
        return {
            "v": 1,                          # Schema version — enables evolution
            "event": {
                "type": qualified_name(self.event),  # e.g. "autogen.beta.events.ModelResponse"
                "data": self.event.to_dict(),        # Event's own serialization
            },
            "sender": self.sender,
            "recipient": self.recipient,
            "trace_id": self.trace_id,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "requires_ack": self.requires_ack,
        }

    @classmethod
    def from_dict(cls, data: dict, event_registry: EventRegistry | None = None) -> "Envelope":
        """Reconstruct from wire format. Uses event registry for type resolution."""
        ...
```

**Key decisions:**
- `"v": 1` — version field enables backward-compatible schema evolution
- Event type is a qualified Python name — enables deserialization without a separate type registry for standard events. The resolver handles nested qualnames (e.g., `Outer.Inner`) by walking the attribute chain after importing the module.
- `EventRegistry` allows custom event types to register their deserialization — plugins can add their own events
- `to_dict()` / `from_dict()` — not JSON strings, so transports choose their own encoding (JSON for HTTP, MessagePack for high-throughput, Protobuf for gRPC). Malformed payloads raise `ValueError` with a clear message including payload keys.
- LocalChannel skips serialization entirely — it passes Envelope objects in-memory
- **Wire format stability note:** Event type names are Python-qualified names tied to module paths. For the public-facing version, consider a stable string registry (e.g., `"ag2.DelegationRequest"`) to decouple wire format from internal module structure.

### Channel

A Channel is a typed event transport with delivery semantics. It abstracts over how events move between actors.

**Protocol:**

```python
@runtime_checkable
class Channel(Protocol):
    """Transport layer for envelopes between actors."""

    async def send(self, envelope: Envelope) -> None:
        """Send an envelope. Delivery semantics depend on implementation."""
        ...

    def subscribe(
        self,
        callback: Callable[[Envelope, Context], Awaitable[None]],
        *,
        condition: Condition | None = None,
    ) -> SubId:
        """Subscribe to incoming envelopes, optionally filtered."""
        ...

    def unsubscribe(self, sub_id: SubId) -> None: ...

    async def close(self) -> None:
        """Graceful shutdown. Flush pending, stop accepting."""
        ...
```

**Implementations:**

| Channel | Delivery | Use Case |
|---------|----------|----------|
| `LocalChannel` | In-process, ordered, at-most-once | Default. Wraps MemoryStream behavior. |
| `BufferedChannel` | In-process with bounded buffer + backpressure | When observers can't keep up. |
| `PriorityChannel` | In-process with priority-ordered delivery via PriorityScheme | When priority matters. |
| `HttpChannel` | HTTP, at-least-once with retry | Cross-process. Multi-server deployments. |
| Future: `RedisChannel` | Redis Streams, at-least-once, persistent | AG2 Cloud managed transport. |
| Future: `NatsChannel` | NATS JetStream, at-least-once | High-throughput distributed. |

**Delivery guarantees:**

```python
class LocalChannel:
    """In-process channel. Ordered delivery. At-most-once semantics."""
    # Wraps MemoryStream. Envelope.event is extracted and sent on the stream.
    # Envelope metadata is available to subscribers.

class BufferedChannel:
    """Bounded buffer with configurable backpressure policy.

    Uses collections.deque for O(1) popleft. The "block" policy uses
    asyncio.Condition for deadlock-free waiting.
    """
    def __init__(
        self,
        max_buffer: int = 1000,
        overflow_policy: Literal["drop_oldest", "drop_newest", "block"] = "drop_oldest",
    ): ...
```

**Design decision:** `LocalChannel` is a thin wrapper over the existing `MemoryStream`. It adds envelope metadata handling but delegates event broadcasting to the proven Stream implementation. This means the Channel primitive introduces zero risk to existing core behavior.

### Infrastructure Protocols

Infrastructure protocols define the backend contracts that make the framework production-ready. Each protocol has an in-memory default implementation. AG2 Cloud (or any deployment) swaps in persistent, distributed backends — same application code, different infrastructure.

```python
@runtime_checkable
class StateStore(Protocol):
    """Persistent key-value state for actors and plugins.

    Enables crash recovery, checkpointing, and distributed state.
    """
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl: float | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...


@runtime_checkable
class Cache(Protocol):
    """Result caching for reducing LLM calls and tool invocations."""
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl: float | None = None) -> None: ...
    async def invalidate(self, pattern: str) -> None: ...


@runtime_checkable
class Lock(Protocol):
    """Distributed coordination for exclusive access."""
    async def acquire(self, key: str, ttl: float = 30.0) -> bool: ...
    async def release(self, key: str) -> None: ...
    # Note: held() is a convenience on LocalLock, not part of the protocol.
    # Custom Lock implementations only need acquire() and release().


@runtime_checkable
class Registry(Protocol):
    """Service registry for actor discovery. Decoupled from Hub for swappability."""
    async def register(self, name: str, info: ActorInfo) -> None: ...
    async def unregister(self, name: str) -> None: ...
    async def discover(self, capability: str = "") -> list[ActorInfo]: ...
    async def heartbeat(self, name: str) -> None: ...
```

**Default implementations (in-memory):**

| Protocol | Default | Cloud Backend |
|----------|---------|--------------|
| `StateStore` | `MemoryStateStore` (dict) | `RedisStateStore`, `PostgresStateStore` |
| `Cache` | `MemoryCache` (dict + TTL) | `RedisCache` |
| `Lock` | `LocalLock` (asyncio.Lock) | `RedisLock`, `EtcdLock` |
| `Registry` | `LocalRegistry` (dict, used by Hub) | `EtcdRegistry`, `ConsulRegistry` |
| `Storage` | `MemoryStorage` (existing, for event history) | `RedisStorage`, `PostgresStorage` |
| `Channel` | `LocalChannel` (existing) | `HttpChannel`, `RedisChannel`, `NatsChannel` |

**Usage — protocols are injected, not hardcoded:**

```python
# Development: zero config, in-memory defaults
hub = Hub()

# Production: swap backends
hub = Hub(
    registry=EtcdRegistry("etcd://cluster:2379"),
    channel=RedisChannel("redis://cluster:6379"),
    state_store=RedisStateStore("redis://cluster:6379"),
)
# Same application code, different infrastructure
```

The `Storage` protocol already exists in AG2 Beta core (for event history). The new protocols extend this pattern to cover all infrastructure concerns needed for production deployment.

### Context Harness

The Context Harness defines **what events the LLM sees and how they're formatted**. It is the bridge between the event stream (which contains ALL events — conversation, tool, network, system, domain) and the LLM context window.

This is a critical primitive because it makes the Actor truly general-purpose. Without a customizable harness, Actors are limited to conversation events. With it, the same Actor infrastructure supports chat agents, network coordinators, system monitors, and domain-specific workflows.

**Protocol:**

```python
@runtime_checkable
class ContextHarness(Protocol):
    """Defines how stream events are assembled into LLM context.

    The harness controls what the LLM sees. Different actors need different
    context — a conversation agent sees chat history, a coordinator sees
    delegation results, a monitor sees system events.
    """

    def select(self, events: list[BaseEvent]) -> list[BaseEvent]:
        """Select which events from the stream history are relevant for the LLM.

        Called before each LLM invocation. Receives the full stream history.
        Returns the subset of events that should form the LLM context.
        """
        ...

    def format(self, event: BaseEvent) -> str | None:
        """Format an event into an LLM-readable message.

        Return None to skip events this harness doesn't know how to format.
        Unknown events fall through to the LLM client's default formatting.
        """
        ...
```

**Implementations:**

```python
class ConversationHarness(ContextHarness):
    """Standard chat harness. Only conversation and tool events reach the LLM.

    This is the default — preserves current Agent behavior exactly.
    """

    def select(self, events):
        return [e for e in events if isinstance(e,
            (ModelRequest, ModelResponse, ToolCallEvent, ToolCallsEvent,
             ToolResultEvent, ToolResultsEvent, ToolErrorEvent))]

    def format(self, event):
        return None  # Use LLM client's default formatting


class NetworkHarness(ContextHarness):
    """Includes network events in the LLM context.

    The actor sees delegation results, signals, scheduler events, and
    formatted events alongside normal conversation — enabling
    network-aware reasoning.
    """

    def select(self, events):
        return [e for e in events if isinstance(e,
            (ModelRequest, ModelResponse, ToolCallEvent, ToolResultEvent,
             ToolErrorEvent, DelegationResult, Signal, SchedulerTriggerFired,
             FormattedEvent))]

    def format(self, event):
        if isinstance(event, DelegationResult):
            return f"[DELEGATION RESULT] {event.source} → {event.target}: {event.result}"
        if isinstance(event, Signal):
            return f"[SIGNAL/{event.severity.upper()}] ({event.source}): {event.message}"
        if isinstance(event, SchedulerTriggerFired):
            return f"[SCHEDULED] Trigger '{event.watch_id}' fired for {event.target}"
        if isinstance(event, FormattedEvent):
            return event.content
        return None  # Fall through to default
```

**Custom domain harness — your events, your rules:**

```python
class MedicalTriageHarness(ContextHarness):
    """Includes patient vitals in the LLM context."""

    def select(self, events):
        return [e for e in events if isinstance(e,
            (ModelRequest, ModelResponse, ToolCallEvent, ToolResultEvent,
             VitalsUpdate, LabResult, NurseNote))]  # Custom domain events

    def format(self, event):
        if isinstance(event, VitalsUpdate):
            return f"[VITALS] HR:{event.heart_rate} BP:{event.blood_pressure} SpO2:{event.spo2}"
        if isinstance(event, LabResult):
            return f"[LAB] {event.test}: {event.value} ({'ABNORMAL' if event.abnormal else 'normal'})"
        return None
```

**The harness is the key to making Actors general-purpose.** The same stream infrastructure, same observer system, same middleware chain — but a different harness transforms a chat agent into a network coordinator or a domain-specific processor. The Actor is not a "conversation agent with extras" — it's an event-processing entity whose behavior is shaped by its harness.

### Harness-Middleware Integration

The Harness is implemented as a **middleware** — specifically, a middleware that wraps `on_llm_call`. This is the critical design decision that connects the primitive layer to the core framework.

```python
class HarnessMiddleware(BaseMiddleware):
    """Bridges ContextHarness into the middleware chain.

    Sits in the middleware chain like any other middleware. Intercepts
    on_llm_call to filter and format events before they reach the LLM client.
    This means the harness composes naturally with all other middleware —
    logging, token limiting, retry, signal injection — without special hooks.
    """

    def __init__(self, harness: ContextHarness):
        self._harness = harness

    async def on_llm_call(self, call_next, events, context):
        # 1. Select: filter stream history to relevant events
        selected = self._harness.select(events)

        # 2. Format: transform events the harness knows about
        formatted = []
        for event in selected:
            custom_format = self._harness.format(event)
            if custom_format is not None:
                # Wrap formatted string as a synthetic event
                formatted.append(FormattedEvent(content=custom_format, original=event))
            else:
                # Pass through — LLM client's default formatting
                formatted.append(event)

        # 3. Delegate to next middleware / LLM client with filtered events
        return await call_next(formatted, context)
```

**Why middleware, not a separate hook?**

1. **Composability.** The harness participates in the same chain as signal injection, token limiting, and logging. No special integration code needed.
2. **Ordering control.** The Actor controls where in the middleware chain the harness sits. Default: harness runs first (outermost), so signal injection and other middleware see the already-filtered event list.
3. **Core stays untouched.** Agent._execute() already supports additional_middleware. The harness plugs in via this existing extension point — zero changes to Layer 1.
4. **Testable in isolation.** A ContextHarness can be unit-tested with a list of events. The middleware wrapper is a thin bridge.

**Middleware ordering in Actor._execute():**

```
Actor._execute() builds middleware chain:
  1. HarnessMiddleware(harness)         ← outermost: filters what events reach the LLM
  2. SignalInjectionMiddleware(queue)    ← injects alerts into the (already filtered) prompt
  3. User-provided middleware            ← logging, token limiting, retry, etc.
  4. LLM client call                     ← innermost: sends to model
```

The ConversationHarness (default) passes through only conversation and tool events — reproducing current Agent behavior exactly. No behavioral change for users who don't use a harness. NetworkHarness and custom domain harnesses expand what the LLM perceives.

---

## Stream Ownership and Conversation Persistence

### Ownership Model

Two stream scopes exist in the framework:

```
PERSISTENT (lives across conversations):
├── Actor instance              — config, observers, harness
├── Observer instances          — carry state across conversations
├── Hub + Hub.stream            — cross-actor events, delegation traffic
├── Channel                     — transport endpoint (HTTP, Redis, etc.)
├── Scheduler + its watches     — trigger definitions persist
└── Registry                    — who's in the network

PER-CONVERSATION (created per .ask(), accumulated via reply.ask()):
├── MemoryStream                — event bus for this conversation
├── Context                     — prompts, dependencies, variables
├── Observer subscriptions      — attach on start, detach on end
└── Middleware instances         — created per _execute()
```

**The stream belongs to the conversation, not the Actor.** Each `.ask()` creates a new stream. Conversation continues via `reply.ask()` (same stream, events accumulate). The LLM sees full history within a conversation. Observers attach per conversation but carry state across conversations (e.g., TokenMonitor accumulates totals).

**The Hub's stream is persistent.** It lives for the Hub's lifetime. System plugins subscribe here for cross-actor monitoring. This is where delegation events, scheduling events, and system-wide signals flow.

### Conversation Persistence

The `Storage` protocol (already in AG2 Beta core) enables persisting conversation sessions. Swap `MemoryStorage` for a persistent backend and every conversation is automatically durable.

```python
# Development — in-memory, ephemeral
actor = Actor("researcher", config=config)
reply = await actor.ask("Research AI trends")
# Events in memory. Gone when process dies.

# Production — persistent storage
storage = RedisStorage("redis://localhost:6379")
actor = Actor("researcher", config=config, storage=storage)
reply = await actor.ask("Research AI trends")
# Events persisted. Survive restarts.

# Resume a previous conversation
reply = await actor.ask(
    "Continue where we left off",
    stream_id=previous_stream_id,  # Load history from storage
)
# LLM sees full prior conversation. Seamless continuation.
```

**Two persistence layers serve different purposes:**

| Protocol | What it persists | Use case |
|----------|-----------------|----------|
| `Storage` | Event streams (conversation history) | Resume conversations, audit trails, replay |
| `StateStore` | Key-value state (actor/observer state) | Crash recovery, distributed state sync |

Together they provide full durability: both the conversation history and the operational state.

> **Note — Future work:** Conversation persistence via `Storage` and `stream_id` resumption is a planned feature, not yet implemented in the network layer. Phase 1 focuses on the primitives and building blocks above. Storage-backed resumption will be added in a subsequent phase.

---

## Layer 3: Building Blocks

Building blocks compose primitives into reusable, independently useful components. Each building block **works standalone** and **composes optionally** with others.

### Actor

An Actor is an Agent enhanced with observer management, signal injection, and task spawning. It is the primary agent type for network-aware applications.

**Relationship to Agent:** Actor extends Agent. It delegates to `super()._execute()` rather than reimplementing the event loop. Actor concerns (observers, signals, tasks) are injected via additional tools and middleware.

```python
class Actor(Agent):
    """An autonomous agent with observers, signals, and task spawning.

    Actor extends Agent with three capabilities:
    1. Observer management — attach/detach observers that monitor the event stream
    2. Signal injection — observer signals are injected into the LLM prompt
    3. Task spawning — spawn_task/spawn_tasks tools for delegating subtasks

    Works standalone (no Hub required). Optionally registers with a Hub
    for cross-actor discovery and delegation.

    Example::

        actor = Actor(
            "researcher",
            prompt="You are a research agent. Use spawn_task for subtasks.",
            config=OpenAIConfig(model="gpt-4o"),
            observers=[TokenMonitor(warn_threshold=50_000)],
            task_config=OpenAIConfig(model="gpt-4o-mini"),
        )
        reply = await actor.ask("Research the latest AI trends")
    """

    def __init__(
        self,
        name: str,
        prompt: str | PromptHook | Iterable[PromptHook] = (),
        *,
        config: ModelConfig | None = None,
        observers: Iterable[Observer] = (),
        harness: ContextHarness | None = None,     # Default: ConversationHarness
        task_config: ModelConfig | None = None,
        task_prompt: str = "You are a task agent...",
        signal_policy: SignalPolicy | None = None,  # Default: InjectToPrompt
        hitl_hook: HumanHook | None = None,
        tools: Iterable[Callable | Tool] = (),
        middleware: Iterable[MiddlewareFactory] = (),
        dependencies: dict[Any, Any] | None = None,
        variables: dict[Any, Any] | None = None,
    ): ...
```

**Execution flow:**

```
Actor._execute()
│
├── 1. Create signal queue
├── 2. Subscribe to Signal events → signal queue
├── 3. Attach observers to stream (observer.attach)
├── 4. Build spawn_task / spawn_tasks tools
├── 5. Create signal injection middleware
├── 6. Call super()._execute() with:
│      additional_tools = [spawn_task, spawn_tasks]
│      additional_middleware = [signal_injection]
│
│   ┌── Agent._execute() runs normally ────────────┐
│   │   Full middleware chain preserved              │
│   │   Signal injection intercepts on_llm_call:    │
│   │     - Drains signal queue                     │
│   │     - Delivers via SignalPolicy               │
│   │     - (default: append to prompt, call LLM,   │
│   │       remove from prompt)                     │
│   │   Tools execute (including spawn_task)         │
│   └───────────────────────────────────────────────┘
│
├── 7. Detach observers (individually try/except — one failure doesn't block others)
└── 8. Return AgentReply
```

**Task spawning:**

`spawn_task(task)` creates an isolated sub-agent:
1. Create isolated `MemoryStream`
2. Bridge `ModelMessageChunk` → `TaskProgress` events to parent stream
3. Create Agent with `task_config`
4. `await agent.ask(task, stream=isolated_stream)`
5. Emit `TaskResult` with usage metrics
6. Return result text

`spawn_tasks(tasks, parallel=True)` runs multiple tasks, optionally in parallel via `asyncio.gather(return_exceptions=True)`. Partial failures return error strings for failed tasks while preserving successful results.

### Observer

An Observer attaches to a stream, uses a Watch to monitor for conditions, and produces Signals when those conditions are met.

**Protocol:**

```python
@runtime_checkable
class Observer(Protocol):
    """Monitors an event stream and produces signals."""

    name: str

    def attach(self, stream: Stream, ctx: Context) -> None:
        """Begin observing. Arm watches, start monitoring."""
        ...

    def detach(self) -> None:
        """Stop observing. Disarm watches, clean up."""
        ...
```

**Convenience base class:**

```python
class BaseObserver(ABC):
    """Trigger-driven observer. Subclasses implement process().

    Exceptions in process() are caught and logged — a failing observer
    does not crash the Actor or kill the stream subscription.
    """

    def __init__(self, name: str, watch: Watch) -> None: ...

    @abstractmethod
    async def process(self, events: list[BaseEvent], ctx: Context) -> Signal | None:
        """Analyze events and optionally return a signal."""
        ...
```

**Built-in observers:**

| Observer | Watches | Signals | Cost |
|----------|---------|---------|------|
| `TokenMonitor` | ModelResponse, TaskResult | WARNING at warn_threshold, CRITICAL at alert_threshold | Zero (rule-based) |
| `LoopDetector` | ToolCallEvent | WARNING on N consecutive identical tool calls | Zero (rule-based) |

**Custom observer:**

```python
from autogen.beta.network import BaseObserver, EventWatch, Signal, Severity

class SafetyMonitor(BaseObserver):
    """Flags responses containing sensitive terms."""

    def __init__(self):
        super().__init__("safety-monitor", watch=EventWatch(ModelResponse))
        self._terms = {"confidential", "password", "secret"}

    async def process(self, events, ctx):
        for event in events:
            if isinstance(event, ModelResponse) and event.content:
                for term in self._terms:
                    if term in event.content.lower():
                        return Signal(
                            source=self.name,
                            severity=Severity.CRITICAL,
                            message=f"Response contains sensitive term: '{term}'",
                        )
        return None
```

### Hub

The Hub is the network center — a discovery registry, priority router, and plugin pipeline. It enables cross-actor communication with full observability.

**What makes Hub powerful is the plugin topology.** The Hub doesn't just route — it transforms. Every delegation passes through a composable plugin pipeline, enabling authentication, rate limiting, telemetry, approval gates, and any custom logic.

```python
class Hub:
    """Discovery, routing, and plugin pipeline for a network of actors.

    The Hub provides:
    - Registry: actors register with capabilities for discovery
    - Router: all inter-actor delegation flows through the Hub
    - Topology: a composable plugin pipeline processes every delegation
    - Stream: the Hub's own event stream for cross-actor observation

    Works with plain Agents or Actors. Does not depend on Scheduler.

    Supports two modes:
    - **Agent mode**: hub.ask(actor, task) — invokes an actor with network tools
    - **Headless mode**: hub.delegate(source, target, task) — pure routing without
      an initiating LLM call. Use for infrastructure Hubs that only route traffic,
      run plugins, and manage registry — zero LLM cost.

    Example::

        hub = Hub()
        hub.register(researcher, capabilities=["research", "analysis"])
        hub.register(writer, capabilities=["writing", "editing"])

        reply = await hub.ask(researcher, "Research and write a report")
        # researcher can discover writer and delegate via network tools
    """

    def __init__(
        self,
        *,
        topology: Topology | None = None,         # Routing pipeline for delegations
        plugins: list[Plugin] | None = None,       # System plugins (observe, manage)
        priority_scheme: PriorityScheme | None = None,
        conflict_resolver: ConflictResolver | None = None,
        channel: Channel | None = None,            # Default: LocalChannel
        registry: Registry | None = None,          # Default: LocalRegistry
        state_store: StateStore | None = None,     # Default: MemoryStateStore
        max_delegation_depth: int = 5,
    ): ...
```

**Registry:**

```python
handle = await hub.register(
    actor_or_agent,
    capabilities: list[str] = [],
    description: str = "",
) -> RegistrationHandle

await handle.unregister()  # convenience method on the handle

await hub.unregister(name: str) -> None

await hub.discover(capability: str = "") -> list[ActorInfo]
```

**Headless mode** — pure infrastructure routing without LLM:

```python
# Headless: Hub as a routing service. No initiating LLM call.
hub = Hub()
await hub.register(worker_a, capabilities=["processing"])
await hub.register(worker_b, capabilities=["processing"])

# Direct delegation — Hub routes, runs plugins, emits events. Zero LLM cost.
result = await hub.delegate("external-api", "worker_a", "Process this batch")

# Hub as long-running service (e.g., AG2 Cloud deployment)
async with hub.serve():
    # Hub accepts delegations via Channel (HttpChannel for cross-process)
    # Runs topology pipeline, manages registry, fires scheduler
    # No agent owns the Hub — it IS the infrastructure
    await asyncio.Event().wait()
```

Headless mode matters for AG2 Cloud where the Hub may be a long-running service separate from any agent — managing registry, running plugins, routing traffic, firing scheduler triggers. It needs zero LLM calls for its own operation.

**Network tools** — automatically injected when an actor is invoked through `hub.ask()`:

- `discover_agents(capability="")` — find other actors, optionally filtered by capability
- `delegate_to(agent_name, task)` — delegate a task to another actor, routed through Hub

**Delegation flow through topology:**

```
Actor calls delegate_to("writer", "Write a report")
│
├── 1. Hub wraps as Envelope (sender, recipient, trace_id, etc.)
├── 2. Envelope passes through topology pipeline:
│      ┌─────────────────────────────────┐
│      │ AuthPlugin    → check permission │
│      │ RateLimiter   → check rate limit │
│      │ TelemetryPlugin → record metrics │
│      │ ApprovalGate  → await approval   │
│      └─────────────────────────────────┘
│      (any plugin can return None to reject)
├── 3. Emit DelegationRequest on Hub stream
├── 4. Call target actor.ask(task) with network tools
├── 5. Emit DelegationResult on Hub stream
└── 6. Return result to calling actor
```

**Self-delegation is rejected. Max delegation depth prevents infinite loops.**

### Scheduler

The Scheduler manages Watch lifecycles — registering, arming, disarming, and canceling watches. It is a **watch lifecycle manager**, not a rigid scheduling engine.

```python
class Scheduler:
    """Manages watch lifecycles. Fires callbacks or delegates tasks when watches trigger.

    Works standalone (manages local watches) or with a Hub (delegates to actors).
    Does not depend on Hub or Actor.

    Example (standalone)::

        scheduler = Scheduler()
        scheduler.add(IntervalWatch(300), callback=my_health_check)
        await scheduler.start()

    Example (with Hub)::

        scheduler = Scheduler(hub=hub)
        scheduler.add(IntervalWatch(300), target="monitor", task="Check health")
        scheduler.add(
            EventWatch(DelegationResult),
            target="auditor",
            task_factory=lambda e: f"Audit: {e.result}",
        )
        await scheduler.start()
    """

    def __init__(self, hub: Hub | None = None): ...
```

**API:**

```python
# Register a watch with a target actor (requires hub)
scheduler.add(
    watch: Watch,
    *,
    target: str = "",                  # Actor name (if hub-connected)
    task: str = "",                    # Static task string
    task_factory: Callable | None = None,  # Dynamic task from event
    callback: Callable | None = None,  # Direct callback (standalone mode)
    priority: Any = None,
) -> str  # Returns watch ID

# Lifecycle management
await scheduler.start()         # Arm all watches
await scheduler.stop()          # Disarm all watches
scheduler.pause(watch_id)       # Temporarily disarm one watch
scheduler.resume(watch_id)      # Re-arm one watch
scheduler.cancel(watch_id)      # Remove watch entirely

# Introspection
scheduler.watches               # List of (watch_id, watch, status) tuples
```

**Independence:** When used standalone (no Hub), Scheduler simply calls callbacks. When used with a Hub, it delegates tasks through the Hub's topology pipeline, getting the full benefits of routing, plugins, and observability.

---

## Layer 4: Composition

Composition patterns combine building blocks into reusable topologies and convenience wrappers.

### Plugin

Plugins are the universal extension mechanism for the Hub. A Plugin has a lifecycle (`install`/`uninstall`) and optionally processes envelopes in the delegation path (`process`).

This distinction is critical: **not all plugins are routing middleware.** Some plugins operate at the system level — monitoring traffic, scaling actors, managing circuit breakers — without sitting in the delegation path. Think of how network infrastructure works: a load balancer routes traffic (routing plugin), but an autoscaler monitors metrics and adjusts capacity independently (system plugin). Both are plugins.

**Protocol:**

```python
@runtime_checkable
class Plugin(Protocol):
    """Extension point for the Hub.

    Plugins can observe Hub state, intercept delegations, manage resources,
    and extend Hub capabilities. They are the primary customization mechanism.

    Two modes of operation:
    1. System plugins — implement install/uninstall only. They subscribe to
       the Hub's stream, react to events, and manage resources independently.
    2. Routing plugins — also implement process(). They sit in the delegation
       path and can transform, reject, or reroute envelopes.
    """

    def install(self, hub: Hub) -> None:
        """Called when plugin is added to Hub.

        System plugins subscribe to hub.stream here to monitor traffic,
        track metrics, or manage resources. Has full access to hub.register(),
        hub.unregister(), hub.discover(), hub.stream.
        """
        ...

    def uninstall(self) -> None:
        """Called when plugin is removed. Clean up subscriptions and resources."""
        ...

    async def process(self, envelope: Envelope, ctx: HubContext) -> Envelope | RouteDecision | None:
        """Optional: process an envelope in the delegation path.

        Return values:
        - Envelope — forward (possibly modified/rerouted)
        - RouteDecision — forward primary + trigger additional delegations
        - None — reject

        Default: pass through unchanged.
        """
        return envelope
```

### RouteDecision

RouteDecision is the routing primitive that enables multicast, co-routing, broadcast, and reject-with-notification patterns. It separates the primary delegation from additional delegations triggered as side-effects.

```python
@dataclass
class RouteDecision:
    """Structured routing outcome from a plugin.

    The primary envelope (if not None) flows through the remaining topology
    pipeline and becomes the delegation that returns a result to the caller.

    Additional envelopes are dispatched by the Hub after the topology pipeline
    completes. Each goes through the full delegation path (depth tracking,
    topology, events) as an independent delegation.
    """
    primary: Envelope | None = None
    additional: list[Envelope] = field(default_factory=list)
```

**Why RouteDecision exists:** Without it, `process()` is a single-in, single-out filter — it can forward, transform, or reject, but it cannot replicate. This is a packet filter, not a router. A router can send a packet to multiple destinations. RouteDecision makes the topology a full routing layer.

**Why a structured type, not `list[Envelope]`:** `list[Envelope]` cannot express "reject the primary but still trigger additional delegations." A circuit breaker that rejects (agent unhealthy) but notifies an alerting agent requires `RouteDecision(primary=None, additional=[alert_envelope])`.

**Composition semantics — additional envelopes propagate upward through topology composition. Only the primary flows through the pipeline chain:**

- **Pipeline:** Additional envelopes accumulate across plugins. Only the primary passes to the next plugin. If any plugin rejects the primary, accumulated additional envelopes are still returned (reject-with-side-effects).
- **Fanout:** Additional envelopes from all plugins are collected. The primary is the original envelope (unchanged, as before). If any plugin rejects or raises, the primary is rejected but accumulated additional envelopes from plugins that already completed are preserved (consistent with Pipeline).
- **Conditional:** Transparently passes through whatever the selected branch returns.

**Hub execution:** After the topology pipeline completes, the Hub dispatches additional envelopes as independent fire-and-forget delegations via `asyncio.create_task`. Each goes through the full `Hub._delegate()` path — depth tracking, topology processing, event emission. The delegation depth guard prevents recursive amplification.

**Example — multicast (replicate to additional targets):**

```python
class CoRouter(BasePlugin):
    """When delegating to any agent, also delegate to audit agents."""

    async def process(self, envelope, ctx):
        auditors = await ctx.hub.discover("audit")
        if auditors:
            return RouteDecision(
                primary=envelope,
                additional=[
                    envelope.child(envelope.event, recipient=a.name)
                    for a in auditors
                ],
            )
        return envelope
```

**Example — reject with notification:**

```python
class ComplianceGuard(BasePlugin):
    """Rejects policy violations but notifies the compliance team."""

    async def process(self, envelope, ctx):
        if self._is_violation(envelope):
            alert = DelegationRequest(
                source="compliance", target="compliance-agent",
                task=f"Policy violation: {envelope.sender} -> {envelope.recipient}",
            )
            return RouteDecision(
                primary=None,
                additional=[envelope.child(alert, recipient="compliance-agent")],
            )
        return envelope
```

**Example — capability-based fan-out:**

```python
class CapabilityFanOut(BasePlugin):
    """Replicate delegation to all agents with a given capability."""

    def __init__(self, capability: str):
        self._capability = capability

    async def process(self, envelope, ctx):
        agents = await ctx.hub.discover(self._capability)
        others = [a for a in agents if a.name != envelope.recipient]
        if others:
            return RouteDecision(
                primary=envelope,
                additional=[
                    envelope.child(envelope.event, recipient=a.name)
                    for a in others
                ],
            )
        return envelope
```

**System plugins** — observe and manage, not in delegation path:

```python
class AutoScaler(Plugin):
    """Monitors delegation traffic and scales actors horizontally."""

    def __init__(self, max_queue_depth: int = 10, actor_factory: Callable = None):
        self._factory = actor_factory
        self._max_depth = max_queue_depth

    def install(self, hub: Hub) -> None:
        self._hub = hub
        self._counts: dict[str, int] = {}
        # Subscribe to hub stream — observe, don't intercept
        hub.stream.subscribe(self._on_delegation, condition=DelegationRequest)
        hub.stream.subscribe(self._on_complete, condition=DelegationResult)

    async def _on_delegation(self, event, ctx):
        self._counts[event.target] = self._counts.get(event.target, 0) + 1
        if self._counts[event.target] > self._max_depth:
            replica = self._factory(event.target)
            self._hub.register(replica, capabilities=["overflow"])

    # process() not overridden — uses default pass-through
    # This plugin is NOT in the delegation path

class CircuitBreaker(Plugin):
    """Monitors failure rates and temporarily disables failing actors."""

    def install(self, hub: Hub) -> None:
        self._hub = hub
        self._failures: dict[str, int] = {}
        hub.stream.subscribe(self._on_error, condition=DelegationError)

    async def _on_error(self, event, ctx):
        self._failures[event.target] = self._failures.get(event.target, 0) + 1
        if self._failures[event.target] > self.threshold:
            self._hub.unregister(event.target)  # Trip the breaker
```

**Routing plugins** — in the delegation path, transform/reject envelopes:

```python
class LoadBalancer(Plugin):
    """Routes to least-loaded actor replica."""

    def install(self, hub: Hub) -> None:
        self._hub = hub
        self._load: dict[str, int] = {}
        hub.stream.subscribe(self._track_load, condition=DelegationRequest | DelegationResult)

    async def process(self, envelope, ctx):
        # Find all actors with the target's capabilities
        replicas = self._hub.discover(capability=envelope.recipient)
        if replicas:
            lightest = min(replicas, key=lambda r: self._load.get(r.name, 0))
            envelope.recipient = lightest.name  # Reroute
        return envelope

class RateLimiter(Plugin):
    """Rejects delegations that exceed rate limits."""

    def install(self, hub: Hub) -> None: ...

    async def process(self, envelope, ctx):
        if self._is_over_limit(envelope.sender):
            return None  # Reject
        self._record(envelope.sender)
        return envelope
```

**Developers build custom plugins by implementing the protocol.** Any Python class with `install()` and `uninstall()` is a valid plugin. Adding `process()` puts it in the delegation path.

### Topology

A Topology defines how routing plugins process envelopes. Topologies are composable — a Topology is itself usable wherever a routing plugin is expected. All topology types handle `RouteDecision` naturally: additional envelopes propagate upward, only the primary flows through the chain.

**Topology types:**

```python
class Pipeline(Topology):
    """Sequential processing. Each routing plugin sees the output of the previous.

    Like PyTorch nn.Sequential — transforms flow through in order.
    Additional envelopes from RouteDecision accumulate through the chain.
    Only the primary envelope is passed to the next plugin.

    Example::

        topology = Pipeline(
            AuthPlugin(),
            RateLimiter(max_per_minute=10),
            TelemetryPlugin(),
        )
    """
    def __init__(self, *plugins: Plugin): ...


class Fanout(Topology):
    """Parallel processing. All plugins see an isolated deep copy concurrently.

    Useful for side-effects (logging, metrics) that don't modify the envelope.
    Returns the original envelope unchanged (side-effect only).
    Additional envelopes from any plugin are collected into the result.
    If any plugin rejects or raises, the primary envelope is rejected but
    accumulated additional envelopes from plugins that already completed
    are preserved (reject-with-side-effects, consistent with Pipeline).

    Example::

        topology = Fanout(
            AuditLogger(),
            MetricsCollector(),
        )
    """
    def __init__(self, *plugins: Plugin): ...


class Conditional(Topology):
    """Branching. Route to different topologies based on a predicate.

    Transparently passes through whatever the selected branch returns —
    Envelope, RouteDecision, or None.

    Example::

        topology = Conditional(
            predicate=lambda env: (env.priority or 0) >= DefaultPriority.URGENT,
            if_true=Pipeline(AlertPlugin(), FastRouter()),
            if_false=Pipeline(QueuePlugin(), BatchRouter()),
        )
    """
    def __init__(
        self,
        predicate: Callable[[Envelope], bool],
        if_true: Topology,
        if_false: Topology | None = None,
    ): ...
```

**Composition — topologies nest freely, mixing routing, multicast, and system plugins:**

```python
hub = Hub(
    # Routing topology — processes every delegation
    topology=Pipeline(
        AuthPlugin(),                     # 1. Check permissions
        LoadBalancer(),                   # 2. Route to least-loaded replica
        CoRouter({"agent_a": ["audit"]}), # 3. Multicast: also notify audit
        Fanout(                           # 4. Side effects (parallel)
            AuditLogger(),
            MetricsCollector(),
        ),
        Conditional(                      # 5. Priority routing
            predicate=lambda e: (e.priority or 0) >= DefaultPriority.URGENT,
            if_true=FastRouter(),
            if_false=BatchRouter(),
        ),
    ),
    # System plugins — observe and manage, not in delegation path
    plugins=[
        AutoScaler(max_queue_depth=10, actor_factory=create_worker),
        CircuitBreaker(failure_threshold=5, recovery_seconds=60),
        TelemetryExporter(endpoint="https://otel.example.com"),
    ],
)
```

### Network

A convenience class that wires Hub + Scheduler with sensible defaults. For users who want the full stack without manual composition.

```python
class Network:
    """Convenience: Hub + Scheduler wired together with sensible defaults.

    Supports async context manager for automatic start/stop + cleanup::

        async with Network() as network:
            await network.register(researcher, capabilities=["research"])
            reply = await network.ask(researcher, "Write a report")
        # __aexit__ calls stop() then hub.close() — full cleanup

    Or manual lifecycle (note: stop() only stops the scheduler;
    call hub.close() separately for full resource cleanup)::

        network = Network()
        await network.register(researcher, capabilities=["research"])
        await network.start()
        reply = await network.ask(researcher, "Write a report")
        await network.stop()
        await network.hub.close()  # Clean up plugins, channel, etc.
    """

    def __init__(
        self,
        *,
        topology: Topology | None = None,
        plugins: Iterable[Plugin] = (),
        channel: Channel | None = None,
        registry: Registry | None = None,
        state_store: StateStore | None = None,
        priority_scheme: PriorityScheme | None = None,
        conflict_resolver: ConflictResolver | None = None,
        max_delegation_depth: int = 5,
    ):
        self.hub = Hub(
            topology=topology, plugins=plugins, channel=channel,
            registry=registry, state_store=state_store,
            priority_scheme=priority_scheme, conflict_resolver=conflict_resolver,
            max_delegation_depth=max_delegation_depth,
        )
        self.scheduler = Scheduler(hub=self.hub)
```

---

## Event Lifecycle

The framework provides clear answers to: **what** is sent **where**, **when**, by **whom**, using **which transport**, and **how** it is delivered.

### Within an Actor (local events)

Events flow through the Actor's stream (Layer 1 Stream primitive). No Envelope overhead.

```
Actor LLM call → ModelRequest → Stream → [subscribers]
                                           ├── LLM client (produces ModelResponse)
                                           ├── Tool executor (on ToolCallsEvent)
                                           └── Observers (via Watch)
                                                └── Signal → signal queue → prompt injection
```

**What:** BaseEvent subclasses (ModelRequest, ModelResponse, ToolCallEvent, etc.)
**Where:** Actor's local MemoryStream
**When:** Immediately on send
**By whom:** Agent execution loop, tools, observers
**Transport:** In-memory (MemoryStream)
**Delivery:** Broadcast to all matching subscribers. Ordered. Synchronous within a turn.

### Between Actors (network events)

Events are wrapped in Envelopes and flow through the Hub's Channel and topology.

```
Actor A calls delegate_to("B", task)
│
├── Hub creates Envelope:
│     event: DelegationRequest(source="A", target="B", task=task)
│     sender: "A"
│     recipient: "B"
│     trace_id: (inherited from current workflow)
│     correlation_id: (new for this delegation)
│     causation_id: (ID of envelope that triggered this delegation)
│     priority: (from caller or default)
│
├── Envelope enters topology Pipeline:
│     Plugin 1 (auth)      → Envelope, RouteDecision, or None
│     Plugin 2 (rate limit) → Envelope, RouteDecision, or None
│     Plugin 3 (multicast)  → RouteDecision(primary, additional=[...])
│     ...
│     Result: primary envelope + accumulated additional envelopes
│
├── If primary survives pipeline:
│     Hub dispatches additional envelopes (fire-and-forget via asyncio.create_task)
│     Hub emits DelegationRequest on Hub stream (for system plugins)
│     Hub sends envelope through Channel (for cross-process transport)
│     Hub calls target_actor.ask(task)
│     Hub emits DelegationResult on Hub stream
│     Hub sends result envelope through Channel (child of request envelope)
│     Result returned to Actor A
│
├── If primary is rejected but additional exist:
│     Hub dispatches additional envelopes (reject-with-side-effects)
│     Hub emits DelegationRejected on its stream
│     Error returned to Actor A
│
└── If primary is rejected with no additional:
    Hub emits DelegationRejected on its stream
    Error returned to Actor A
```

**What:** Envelope containing DelegationRequest/DelegationResult
**Where:** Hub's Channel → Hub's stream → target Actor's stream
**When:** Priority-ordered (via PriorityChannel) or immediate (via LocalChannel)
**By whom:** Hub router, triggered by Actor's delegate_to tool
**Transport:** Channel implementation (LocalChannel default, pluggable)
**Delivery:** At-most-once (LocalChannel), at-least-once (future distributed channels)
**Ack:** Optional via `Envelope.requires_ack`
**Retry:** Channel-implementation-specific (LocalChannel: none; future: configurable)
**TTL:** Envelope-level, enforced by Channel

### Scheduler-initiated events

```
Watch fires (IntervalWatch timer, EventWatch match, etc.)
│
├── Scheduler emits SchedulerTriggerFired on Hub stream (or locally)
├── If hub-connected:
│     Scheduler calls hub._delegate(target, task)
│     → enters Hub topology pipeline (same as actor delegation)
├── If standalone:
│     Scheduler calls registered callback directly
```

---

## Design Principles

### 1. Additive, Not Invasive

No changes to core Agent, Stream, or Event classes. Everything is layered on top via additional tools, middleware, and subscribers. An Actor IS an Agent with extras. A Channel wraps a Stream.

### 2. Same Patterns at Every Level

Observers observe streams. This pattern works for Actors (local observers), Hub (plugins as observers), and future extensions. Watch is the universal condition primitive used by Observers and Schedulers alike.

### 3. Protocol-Based Extensibility

Watch, Observer, Plugin, Channel, PriorityScheme, ConflictResolver, SignalPolicy — all are protocols. Custom implementations plug in without subclassing. `@runtime_checkable` enables structural typing.

### 4. Independence and Composability

Every building block works alone:
- Actor works without Hub (standalone agent with observers)
- Hub works without Actor (routes between plain Agents)
- Scheduler works without Hub (standalone watch manager)
- Any combination works (Actor + Hub, Hub + Scheduler, all three)

### 5. Mechanism, Not Policy

The framework provides delivery mechanisms (Channel, Priority, ConflictResolver). Business logic (what priority levels mean, how conflicts resolve, what approval requires) is defined by developers. Sensible defaults are provided but never mandated.

### 6. In-Process and Distributed From Day One

LocalChannel and HttpChannel ship together. In-process for development and testing, HTTP for multi-process deployments. RedisChannel/NatsChannel for high-throughput production. Same application code, different Channel backend. The Envelope carries all metadata needed for any transport.

### 7. AI-First API Design

The framework is designed for both human developers and coding agents:

- **Complete type hints** — AI agents infer correct usage from types
- **Consistent protocol surfaces** — learn one building block, predict all others
- **Self-documenting errors** — suggest the fix, not just the problem
- **Introspectable state** — every object exposes its current state as properties
- **Convention-based defaults** — zero-config works; everything overridable
- **Runnable examples in docstrings** — AI agents read docstrings more than docs sites

---

## File Structure (Current)

```
autogen/beta/
├── (core: agent.py, stream.py, context.py, events/, middleware/, tools/, config/)
│
└── network/                    # AG2 Network Framework package
    ├── __init__.py             # Public API — all symbols, one import
    │
    ├── primitives/             # Layer 2: Primitives
    │   ├── watch.py            # Watch protocol + EventWatch, BatchWatch, WindowWatch,
    │   │                       #   IntervalWatch, DelayWatch, CronWatch, AllOf, AnyOf, Sequence
    │   ├── signal.py           # Signal + Severity + InjectToPrompt, EmitToStream, HaltOnFatal
    │   ├── priority.py         # PriorityScheme + ConflictResolver + DefaultPriority
    │   ├── envelope.py         # Envelope dataclass + wire format (to_dict/from_dict)
    │   ├── channel.py          # Channel protocol + LocalChannel, BufferedChannel, PriorityChannel
    │   ├── harness.py          # ContextHarness + HarnessMiddleware + ConversationHarness, NetworkHarness
    │   └── infra.py            # StateStore, Cache, Lock, Registry protocols + in-memory defaults
    │
    ├── actor.py                # Layer 3: Actor (extends Agent) + signal injection + task spawning
    ├── observer.py             # Observer protocol + BaseObserver
    ├── hub.py                  # Hub + Registry + Router + topology pipeline + headless mode
    ├── scheduler.py            # Scheduler (watch lifecycle manager)
    ├── convenience.py          # Network convenience class (Hub + Scheduler)
    ├── events.py               # All network events: observer, task, delegation, scheduler
    ├── topology.py             # Layer 4: Plugin protocol, RouteDecision + Pipeline, Fanout, Conditional
    │
    ├── remote.py               # RemoteAgent + RemoteAgentReply — cross-server agent proxy
    │
    ├── channels/               # Transport implementations
    │   └── http.py             # HttpChannel — cross-process HTTP transport, at-least-once with retry
    │
    ├── observers/              # Built-in observers
    │   ├── token_monitor.py    # TokenMonitor — tracks cumulative token usage
    │   └── loop_detector.py    # LoopDetector — detects repetitive tool-call patterns
    │
    └── plugins/                # Built-in plugins
        ├── rate_limiter.py     # RateLimiter — per-sender rate limiting (routing plugin)
        └── telemetry.py        # TelemetryPlugin — delegation metrics (system plugin)
```

### Public API

```python
from autogen.beta.network import (
    # Layer 2: Primitives
    Watch, EventWatch, BatchWatch, WindowWatch, IntervalWatch, DelayWatch, CronWatch,
    AllOf, AnyOf, Sequence,
    Signal, Severity, SignalPolicy, InjectToPrompt, EmitToStream, HaltOnFatal, HaltEvent,
    PriorityScheme, ConflictResolver, DefaultPriority, DefaultPriorityScheme, HighestPriorityWins,
    Envelope, EventRegistry,
    Channel, LocalChannel, BufferedChannel, PriorityChannel, HttpChannel,
    ContextHarness, ConversationHarness, NetworkHarness, HarnessMiddleware, FormattedEvent,
    StateStore, MemoryStateStore, Cache, MemoryCache, Lock, LocalLock, Registry, LocalRegistry, ActorInfo,

    # Events
    ObserverStarted, ObserverCompleted, TaskRequest, TaskProgress, TaskResult,
    DelegationRequest, DelegationResult, DelegationRejected, DelegationError, SchedulerTriggerFired,

    # Layer 3: Building Blocks
    Actor, Observer, BaseObserver, Hub, RegistrationHandle, Scheduler, WatchStatus, Network,

    # Layer 4: Composition
    Plugin, BasePlugin, Topology, Pipeline, Fanout, Conditional, HubContext,

    # Distributed
    RemoteAgent, RemoteAgentReply,

    # Built-in Observers & Plugins
    TokenMonitor, LoopDetector, RateLimiter, TelemetryPlugin,
)
```

---

## Usage Examples

### Standalone Actor with Observers

```python
from autogen.beta.network import Actor, TokenMonitor, LoopDetector
from autogen.beta.config.openai import OpenAIConfig

actor = Actor(
    "researcher",
    prompt="You are a research agent. Use spawn_task for subtasks.",
    config=OpenAIConfig(model="gpt-4o"),
    observers=[
        TokenMonitor(warn_threshold=50_000),
        LoopDetector(repeat_threshold=3),
    ],
    task_config=OpenAIConfig(model="gpt-4o-mini"),
)

reply = await actor.ask("Research the latest AI trends and compile a report")
print(reply.content)
```

### Multi-Actor Network with Hub

```python
from autogen.beta.network import Actor, Hub, Pipeline
from autogen.beta.config.openai import OpenAIConfig

config = OpenAIConfig(model="gpt-4o")

researcher = Actor("researcher", prompt="You research topics thoroughly.", config=config)
writer = Actor("writer", prompt="You write clear, engaging reports.", config=config)
reviewer = Actor("reviewer", prompt="You review for accuracy and clarity.", config=config)

hub = Hub()
await hub.register(researcher, capabilities=["research", "analysis"])
await hub.register(writer, capabilities=["writing", "editing"])
await hub.register(reviewer, capabilities=["review", "quality"])

# researcher can discover writer and reviewer, and delegate to them
reply = await hub.ask(researcher, "Research AI trends, write a report, and have it reviewed.")
```

### Hub with Routing Topology + System Plugins

```python
from autogen.beta.network import Hub, Pipeline, Fanout

hub = Hub(
    # Routing topology — processes every delegation in order
    topology=Pipeline(
        AuthPlugin(allowed_delegations={"researcher": ["writer", "reviewer"]}),
        LoadBalancer(strategy="least-loaded"),
        Fanout(
            AuditLogger(log_path="delegations.jsonl"),
            MetricsCollector(statsd_host="localhost"),
        ),
        RateLimiter(max_per_minute=20),
    ),
    # System plugins — observe and manage independently
    plugins=[
        AutoScaler(max_queue_depth=10, actor_factory=create_worker),
        CircuitBreaker(failure_threshold=5, recovery_seconds=60),
    ],
    max_delegation_depth=3,
)
```

### Scheduler with Watches

```python
from autogen.beta.network import Hub, Scheduler, IntervalWatch, EventWatch, CronWatch

hub = Hub()
await hub.register(monitor, capabilities=["monitoring"])
await hub.register(reporter, capabilities=["reporting"])

scheduler = Scheduler(hub=hub)

# Health check every 5 minutes
scheduler.add(IntervalWatch(300), target="monitor", task="Check all systems.")

# When monitor reports results, wake reporter
scheduler.add(
    EventWatch(DelegationResult.source == "monitor"),
    target="reporter",
    task_factory=lambda e: f"Summarize this report: {e.result}",
)

# Weekly summary every Monday at 9am
scheduler.add(
    CronWatch("0 9 * * MON"),
    target="reporter",
    task="Generate weekly status summary.",
)

await scheduler.start()
```

### Full Network (Convenience)

```python
from autogen.beta.network import Network, IntervalWatch, EventWatch, Pipeline

network = Network(
    topology=Pipeline(AuthPlugin(), TelemetryPlugin()),
)

await network.register(monitor, capabilities=["monitoring"])
await network.register(alerter, capabilities=["alerting"])

network.schedule(IntervalWatch(300), target="monitor", task="Check all systems.")
network.schedule(
    EventWatch(Signal.severity == "critical"),
    target="alerter",
    task_factory=lambda e: f"Alert team: {e.message}",
)

async with network:
    # Actors now run autonomously — scheduler triggers, actors delegate, Hub routes
    reply = await network.ask(monitor, "Initial system check")
```

### Custom Watch (Composite)

```python
from autogen.beta.network import AllOf, EventWatch, IntervalWatch

# Fire only when BOTH conditions are met:
# 1. A critical signal was emitted, AND
# 2. At least 60 seconds since last trigger (debounce)
debounced_critical = AllOf(
    EventWatch(Signal.severity == "critical"),
    IntervalWatch(60),
)

scheduler.add(debounced_critical, target="alerter", task="Investigate critical signal")
```

---

## Implementation Status

### Completed: Phases 1–3 + HttpChannel + Distributed Delegation

All layers of the framework are implemented and tested (260 tests, 0 failures, 0 changes to Layer 1 core).

| Phase | What | Status |
|-------|------|--------|
| **Phase 1: Primitives** | Watch (7 types + 3 composites), Signal (4 policies), Priority, Envelope (wire format), Channel (3 types), Harness (middleware bridge), Infra (4 protocols) | **Done** |
| **Phase 2: Building Blocks** | Actor, Observer, Hub (with RegistrationHandle, serve(), state_store/priority_scheme/conflict_resolver), Scheduler, events, TokenMonitor, LoopDetector | **Done** |
| **Phase 3: Composition** | Plugin protocol, RouteDecision, Topology (Pipeline/Fanout/Conditional), WindowWatch, CronWatch, RateLimiter, TelemetryPlugin | **Done** |
| **HttpChannel** | Cross-process HTTP transport with at-least-once delivery, retry, health endpoint | **Done** |
| **Network** | Convenience class (Hub + Scheduler) with async context manager support | **Done** |
| **Distributed Delegation** | RemoteAgent, Hub.serve(), Hub.connect() — cross-server agent communication over HTTP | **Done** |

The old `satellites/` directory has been fully removed. All code lives in `autogen/beta/network/`.

---

## Cross-Server Distributed Delegation

Agents running on different servers communicate transparently through `RemoteAgent`, `Hub.serve()`, and `Hub.connect()`. No changes to Hub, Actor, Channel, or any Layer 1–3 code.

### Architecture

The distributed layer adds three components:

1. **`RemoteAgent`** — A proxy that implements the Agent protocol (`name` + `ask()`) but forwards calls over HTTP to a remote Hub's delegation endpoint. The local Hub treats it like any other agent.

2. **`Hub.serve(host, port)`** — Starts an HTTP server exposing three endpoints: `POST /delegate` (run an agent, return result), `GET /discover` (return registered local agents), `GET /health` (liveness).

3. **`Hub.connect(endpoint)`** — Fetches a remote Hub's `/discover` endpoint, creates `RemoteAgent` proxies for each discovered agent, and registers them locally with their capabilities.

```
Server A                               Server B
┌────────────────────────────┐         ┌────────────────────────────┐
│ Hub.serve(port=8901)       │         │ Hub                        │
│                            │         │                            │
│ ┌──────┐    ┌──────────┐  │  HTTP   │ ┌──────────────────────┐  │
│ │ EMS  │    │ Hospital │  │◄────────│ │ RemoteAgent("ems")   │  │
│ └──────┘    └──────────┘  │         │ │ RemoteAgent("hosp")  │  │
│                            │         │ ├──────────────────────┤  │
│ POST /delegate ← agent,   │         │ │ Dispatch (local)     │  │
│   task → run agent, return │         │ └──────────────────────┘  │
│   result                   │         │                            │
│ GET /discover → local      │         │ hub.connect("server-a:8901") │
│   agent list               │         │   → auto-discovers EMS,     │
└────────────────────────────┘         │     Hospital as RemoteAgents │
                                       └────────────────────────────┘
```

### How It Works

When a local Hub delegates to a `RemoteAgent`:

```
1. Hub._delegate("ems", task, source="dispatch")
   ├── Creates Envelope, runs local topology pipeline
   ├── Sets _delegation_source context var (for tracing)
   └── Calls RemoteAgent("ems").ask(task)

2. RemoteAgent.ask(task)
   ├── Reads _delegation_source from context var
   ├── POST http://server-a:8901/delegate
   │     body: {"agent": "ems", "task": "...", "source": "dispatch"}
   └── Returns RemoteAgentReply(content=response)

3. Remote Hub's /delegate handler
   ├── Looks up "ems" in local registry
   ├── Builds network tools (discover_agents, delegate_to)
   ├── Calls ems.ask(task, tools=network_tools)
   │     ─── EMS runs, delegates to Hospital LOCALLY ───
   └── Returns {"status": "ok", "result": "..."}

4. Result flows back through the chain
```

Key behaviors:

- **Local topology still runs.** The calling Hub's topology pipeline processes the delegation before it goes remote.
- **Remote agents get network tools.** The remote Hub injects `discover_agents` and `delegate_to` so remote agents can discover and delegate to other agents on their own server (e.g., EMS → Hospital locally).
- **Mixed local + remote.** A Hub can have both local agents and RemoteAgent proxies. Discovery returns all of them transparently.
- **Retry with backoff.** RemoteAgent retries failed HTTP calls with configurable `max_retries` and exponential `retry_delay`.

### Usage

**Server side:**

```python
hub = Hub()
await hub.register(ems, capabilities=["medical"])
await hub.register(hospital, capabilities=["trauma"])

async with hub.serve(host="0.0.0.0", port=8901):
    await asyncio.Event().wait()
```

**Client side:**

```python
hub = Hub()
await hub.register(dispatch, capabilities=["dispatch"])

# Auto-discover remote agents
await hub.connect("http://server-a:8901")
# hub now has RemoteAgent("ems") and RemoteAgent("hospital")

reply = await hub.ask(dispatch, "Emergency: car accident...")
# dispatch → discover_agents() sees ems, hospital
# dispatch → delegate_to("ems", ...) → HTTP → server A → EMS runs
# EMS → delegate_to("hospital", ...) → LOCAL on server A
```

**Manual RemoteAgent (without connect):**

```python
from autogen.beta.network import RemoteAgent

remote = RemoteAgent("ems", "http://server-a:8901", timeout=120.0)
await hub.register(remote, capabilities=["medical"])
```

### On A2A and Future Protocol Support

AG2 uses its own protocol for distributed AG2-to-AG2 communication. This is intentional:

| Concern | AG2 Protocol | A2A |
|---------|-------------|-----|
| **State** | Stateful by default — Envelope traces full causation chain | Ambiguous — `contextId` is optional, agents can be stateless |
| **Topology** | Envelopes flow through plugin pipeline on the calling side | Opaque — no plugin/routing layer |
| **Observability** | Full event stream, observers, signals built in | Nothing built in |
| **Latency** | Direct HTTP POST, minimal overhead | JSON-RPC 2.0 with Agent Card negotiation |

A2A support will be added later as an adapter — `A2ARemoteAgent` that wraps the A2A protocol (Agent Cards, Tasks, Messages) behind the same `RemoteAgent` interface. This means A2A agents plug into the same Hub topology, get the same observability, and work alongside native AG2 agents without special handling.

### Extending to Other Transports

`RemoteAgent` currently uses HTTP. The same pattern extends to any transport:

```python
# Future: WebSocket for streaming
class WebSocketRemoteAgent(RemoteAgent):
    async def ask(self, msg, **kwargs):
        # WebSocket connection for real-time streaming
        ...

# Future: gRPC for high-throughput
class GrpcRemoteAgent(RemoteAgent):
    async def ask(self, msg, **kwargs):
        # gRPC stub call
        ...

# Future: A2A for cross-ecosystem interop
class A2ARemoteAgent(RemoteAgent):
    async def ask(self, msg, **kwargs):
        # Send A2A SendMessage, poll/stream for Task completion
        ...
```

Each variant only overrides `ask()` and `_post()`. The Hub, topology, plugins, observers, and Scheduler work identically regardless of transport — they only see an agent with a `name` and an `ask()` method.

Similarly, `Hub.serve()` can be extended with additional protocol bindings:

```python
# Future: Hub serves both HTTP and gRPC simultaneously
async with hub.serve(host="0.0.0.0", port=8901, grpc_port=50051):
    ...

# Future: Hub exposes A2A Agent Card
async with hub.serve(host="0.0.0.0", port=8901, a2a=True):
    # GET /.well-known/agent.json returns Agent Card
    # POST /tasks accepts A2A SendMessage
    ...
```

The key design principle: **the distributed layer is additive.** `RemoteAgent` and `Hub.serve()` sit above the existing Hub/Actor/Channel architecture without modifying it. New transports are new implementations of the same pattern, not new abstractions.

---

## Roadmap

### Near-term: End-to-end validation

The framework is functionally complete for single-process and distributed multi-server deployments. Priority is proving it works with real agents and real LLM calls through comprehensive examples. Distributed demos exist in `playground/04_emergency/` and `playground/05_smart_building/`.

### Deferred: Cloud Backends (AG2 Cloud)

These items are deferred until AG2 Cloud infrastructure is ready. The protocol contracts are already defined — implementation is swapping backends behind existing interfaces.

| Item | What | Blocked on |
|------|------|------------|
| `channels/redis.py` | RedisChannel — persistent at-least-once transport via Redis Streams | AG2 Cloud infrastructure |
| `channels/nats.py` | NatsChannel — high-throughput NATS JetStream transport | AG2 Cloud infrastructure |
| `infra/redis.py` | RedisStateStore, RedisCache, RedisLock | AG2 Cloud infrastructure |
| `infra/etcd.py` | EtcdRegistry — distributed service registry | AG2 Cloud infrastructure |
| Durable Scheduler | Temporal/Celery backend for scheduler watches | AG2 Cloud infrastructure |
| Hub replication | Replicated Hub with leader election | AG2 Cloud infrastructure |

### Deferred: Protocol Integration

| Item | What | Blocked on |
|------|------|------------|
| A2A integration | `A2ARemoteAgent` wraps A2A protocol behind Agent interface — Agent Cards, Tasks, Messages | A2A spec stabilization |
| gRPC transport | `GrpcRemoteAgent` + `Hub.serve(grpc_port=)` for high-throughput deployments | Concrete performance requirements |
| WebSocket streaming | `WebSocketRemoteAgent` for real-time result streaming during delegation | Streaming use case validation |
| MCP integration | Actors expose tools via MCP server protocol | MCP spec stabilization |
| OpenTelemetry export | Envelope trace_id/correlation_id → OTel spans for Jaeger/Grafana | Concrete deployment requirements |

### Design principle for deferred items

The framework already defines the complete programming model. Every deferred item is a backend swap behind an existing protocol:

- `RedisChannel` implements `Channel` (same as `LocalChannel` and `HttpChannel`)
- `EtcdRegistry` implements `Registry` (same as `LocalRegistry`)
- `RedisStateStore` implements `StateStore` (same as `MemoryStateStore`)
- `A2ARemoteAgent` extends `RemoteAgent` (same Hub integration, different wire protocol)
- `GrpcRemoteAgent` extends `RemoteAgent` (same Hub integration, different transport)

**Same application code, different backends.** No user code changes needed when cloud backends or new transports ship.

---

## Extension Points for AG2 Cloud

| Protocol | OSS Default (In-Memory) | Cloud Backend (Distributed) |
|----------|------------------------|---------------------------|
| `Channel` | `LocalChannel` | `HttpChannel`, `RedisChannel`, `NatsChannel` |
| `Registry` | `LocalRegistry` (dict) | `EtcdRegistry`, `ConsulRegistry` |
| `StateStore` | `MemoryStateStore` (dict) | `RedisStateStore`, `PostgresStateStore` |
| `Cache` | `MemoryCache` (dict + TTL) | `RedisCache` |
| `Lock` | `LocalLock` (asyncio.Lock) | `RedisLock`, `EtcdLock` |
| `Storage` | `MemoryStorage` (existing) | `RedisStorage`, `PostgresStorage` |
| `RemoteAgent` | HTTP (built-in) | gRPC, WebSocket, A2A |
| `Hub.serve()` | HTTP (built-in) | + gRPC port, + A2A Agent Card |
| Tracing | `Envelope` fields + `_delegation_source` | OpenTelemetry export |
| Scheduler | `asyncio` timers | Durable scheduler (Temporal, Celery) |
| Hub | Single-process | Replicated with leader election |

The OSS framework defines the complete programming model. Cloud swaps the infrastructure backends. **Same application code, different backends.**
