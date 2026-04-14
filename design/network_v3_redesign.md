# Network V3 Redesign

Status: proposal. Supersedes `ag2_network_framework.md` (V2) for the network layer.
Framework core (Agent, Stream, Events, Middleware, KnowledgeStore, AssemblyPolicy, Observers) is retained; this document only touches the network surface and the actor/hub boundary.

---

## 1. Problem Statement

Critical issues with the current network architecture:

1. **Stateless delivery.** Every delegation is fire-and-forget. A caller waits on a synchronous `ask()` result or receives nothing at all. There is no shared, durable conversation between actors.
2. **Scattered features, incoherent API.** Topics, delegation, background jobs, cross-store queries, and progress reporting were bolted onto `Hub.network(...)` one action at a time. The surface is wide and the semantics are inconsistent.
3. **Topology + plugin overkill.** The Pipeline/Fanout/Conditional/Plugin/RouteDecision stack is powerful but hostile to humans and coding agents. Most real use cases are declarative (auth, rate-limit, audit); the pipeline abstraction has 4x the concepts needed.
4. **No durable task lifecycle, no streaming.** Background delegations exist but there is no first-class task type with phases, checkpoints, TTL enforcement, or progress subscription. Streaming relies on an ad-hoc `ask_stream` race.
5. **No access control.** Any registered actor can call any other actor with no per-actor rules. Cross-actor knowledge queries use an `exposed_paths` side channel that is outside the normal routing path.
6. **Remote is a second-class citizen.** `RemoteAgent` is a special `Agent` subclass that owns its own HTTP client; local and remote routing go through different code paths. Cross-server delegation loses almost all observability, metadata, and rule enforcement.

V3 rebuilds the network around **stateful sessions, hub-owned state machines, rule-based access, and one uniform transport** so that local and remote actors are identical to the hub.

---

## 2. Core Principles

1. **Sessions are the unit of communication, not individual calls.** Every exchange happens inside a session with a defined type, participants, lifecycle, and durable log.
2. **The hub is a virtual file system.** Hub state (actors, rules, sessions, tasks, inboxes) lives on an extended `KnowledgeStore`. Persistence is not bolted on — it IS the hub.
3. **Identity is structured and pluralized.** Every registration presents a single `ActorIdentity` (profile + capability surface + auth). A standalone `Actor` has no identity and no `actor_id`. One Python `Actor` may carry several identities and present a different one to each hub it joins — each registration is independent.
4. **Two paired clients, not one binding.** The actor process holds two peer clients of the same `Link`: a `HubClient` (outbound — registration, discovery, session creation, sending) and an `ActorClient` (one per registered identity — holds the `Actor`, runs the inbox, executes per-envelope transforms locally). The hub never runs tenant transform code in its own address space.
5. **Rules, not topology.** Access control, quotas, and pre/post transforms are declared per-actor. No per-hub pipeline concept, no RouteDecision, no Pipeline/Fanout/Conditional. Rule enforcement is split: the hub evaluates `access` and `limits` (cross-call aggregation), the recipient's `ActorClient` runs `transforms` (per-envelope, isolated).
6. **One transport, two protocols.** WebSocket for everything stateful (session envelopes, streaming, notify, subscriptions). HTTP for stateless CRUD (registration, discovery, session creation, rule changes, read APIs). Local actors use an in-process transport that honors the same contract.
7. **There is no "remote". There are only addresses.** A registered identity has a runtime binding stored alongside its identity record. The hub dispatches by binding. Local, in-cluster, cross-server — all identical from the actor's point of view.
8. **IDs everywhere, UUID7, hub-stamped, per identity.** Every resource — including `actor_id` itself — gets a UUID7 from the hub at the moment of creation. `actor_id` is assigned at *registration*, not when an `Actor` is constructed; two identities of the same Python `Actor` registering with the same hub get two distinct `actor_id`s. Names are for humans; IDs are for everything internal.
9. **Hub owns state machines; actors report transitions.** Tasks and sessions have hub-enforced lifecycles. Actors cannot silently drop a session or leak a task — TTLs and invariants are the hub's responsibility.

---

## 3. Identity

Every registration presents a single **`ActorIdentity`** — a structured profile that combines who the actor is (profile), what it can do (capabilities), how to use it (skill), and how it authenticates (auth). A standalone framework-core `Actor` (not registered with any hub) has no identity and no `actor_id`. Identity is **registration input**, not actor state.

A single Python `Actor` may carry multiple identities and present a different one to each hub it joins — for example, `ag2:researcher:1` on a community hub and `acme:senior_research_lead` on an enterprise hub, each with its own capabilities, skill prompts, and credentials. Each registration is independent, and each yields a fresh `actor_id`.

### 3.1 File layout

```
hub/actors/{actor_id}/
  identity.json          # ActorIdentity (profile + capabilities + auth)
  SKILL.md               # optional sidecar, free-form usage guide for LLMs
  rule.json              # access/limits/transforms (see §4)
  runtime.json           # hub-owned binding/heartbeat (NOT part of identity)
  inbox/ ...             # hub-owned envelope queue
```

`identity.json` is uploaded by the actor at registration and is **immutable for the life of the registration** — modifying it requires unregistering and re-registering, which produces a new `actor_id`. `runtime.json` is owned and mutated by the hub on every heartbeat / reconnect, so identity reads stay cache-friendly and audit history stays clean. Splitting runtime out of identity is a deliberate inversion of V2's combined `passport.json`: the actor never writes runtime, the hub never rewrites identity.

### 3.2 ActorIdentity

```jsonc
// hub/actors/{actor_id}/identity.json
{
  // -- system-stamped --
  "actor_id":      "01932f8a...",          // UUID7, hub-stamped at registration

  // -- profile (who the actor is) --
  "name":          "ag2:researcher:1",     // human/LLM-facing address, unique within hub
  "display":       "Research Agent",
  "owner":         "ag2",
  "version":       "1",
  "framework":     "ag2-beta",             // what built this actor
  "runtime_kind":  "python",               // python | node | browser | external | human
  "model_hint":    "anthropic/claude-opus-4-6",
  "locale":        "en-US",
  "timezone":      "America/Los_Angeles",
  "pricing":       { "per_1k_input_tokens": 0.015,
                     "per_1k_output_tokens": 0.075,
                     "currency": "USD" },
  "restrictions":  {                        // self-declared hard limits
    "max_tokens_per_call": 200000,
    "data_regions": ["us-west"],
    "pii": "redact"
  },

  // -- capability surface (what the actor can do) --
  "capabilities":  ["research", "summarization", "citation-check"],
  "summary":       "Produces cited, multi-source literature reviews on a topic.",
  "domains":       ["biomedicine", "climate-science"],
  "strengths":     "Strong at synthesizing long sources. Weak at real-time data.",
  "history": [                              // prior work the actor is proud of
    { "session_id": "01932...", "task_id": "01932...", "title": "CRISPR 2025 review" }
  ],
  "knowledge_bases": [                      // paths into its private store
    { "path": "/memory/biomed/", "description": "100+ cached biomed abstracts" }
  ],
  "tools": [
    { "name": "web_search", "visibility": "public" },
    { "name": "pubmed_api",  "visibility": "exclusive" }
  ],
  "session_limits": {
    "concurrent_sessions": 3,
    "session_types": ["consulting", "conversation"]
  },

  // -- auth (how the hub validates this identity at handshake) --
  "auth": {
    "scheme":          "jwt",               // none | api_key | jwt | mtls | signed_challenge
    "issuer":          "https://auth.ag2.cloud",
    "audience":        "hub-prod",
    "key_fingerprint": "sha256:...",
    "claim":           { /* scheme-specific extra fields */ }
  },

  // -- provenance --
  "provenance": {
    "registered_at": "2026-04-12T18:22:01Z",
    "registered_by": "sirentropy"
  }
}
```

Three logical blocks:

| Block        | What it carries                                                       | Used by                                |
|--------------|-----------------------------------------------------------------------|----------------------------------------|
| profile      | `name`, `display`, `owner`, `version`, `framework`, `pricing`, etc.   | Discovery; UI rendering; pricing audit |
| capabilities | `capabilities`, `summary`, `domains`, `strengths`, `tools`, `session_limits` | Discovery index; LLM judgment in `describe_actor` |
| auth         | `scheme`, `claim`, `key_fingerprint`                                  | Hub-side `AuthAdapter` validates this at handshake (§13) |

`name` is unique within a hub. `actor_id` is unique across time. Two registrations from the same Python `Actor` using two different identities — even on the same hub — produce two different `actor_id`s; from the hub's point of view they are independent registrations sharing nothing. Renaming an identity is not a thing: change the name, you must re-register, and you get a new `actor_id`.

The hub-side `AuthAdapter` (§13) is a pluggable validator selected by `auth.scheme`. The identity carries the credential claim; the adapter knows how to verify it. A hub may install multiple adapters (one per scheme it accepts) and they coexist.

### 3.3 SKILL.md

`hub/actors/{actor_id}/SKILL.md`: a plain markdown file the actor owner ships alongside the identity, explaining prompting conventions, session-type preferences, message format, what not to ask, how to interpret results. Returned verbatim when another actor calls `describe_actor(name)`. Kept as a sidecar file because it is free-form prose; the in-memory `ActorIdentity` carries it as a `skill_md: str | None` field, so application code sees one logical object even though the on-disk shape is two files.

### 3.4 Runtime

```jsonc
// hub/actors/{actor_id}/runtime.json    (hub-owned, written on every heartbeat)
{
  "actor_id":       "01932f8a...",
  "binding":        "local",                // local | ws | http | external
  "target":         "inproc://actor-01932...",
  "ws_url":         "wss://edge-01.ag2.cloud/actors/01932.../link",
  "http_url":       "https://edge-01.ag2.cloud/actors/01932.../notify",
  "reachable":      true,
  "last_heartbeat": "2026-04-12T18:25:00Z"
}
```

The hub uses `runtime.binding` to pick a transport when dispatching envelopes. Local in-process actors get a direct queue wired to the same notify protocol. Cross-host actors get a long-lived WebSocket. The caller never sees these details — it just calls `session.send(...)`.

Runtime lives outside identity for three reasons: it changes far more often (every heartbeat), it is hub-managed (the actor never writes it), and keeping it out of identity means identity reads can be cached forever without invalidation churn.

---

## 4. Rules

Rules replace the V2 topology/plugin pipeline. They are declarative per-actor with a structured escape hatch for custom logic. A rule file is three layers: **access**, **limits**, **transforms**. No God Object.

**Two enforcement sites.** `access` and `limits` are evaluated by the hub at handshake / dispatch time — they need cross-call aggregation (concurrent counts, rate windows, delegation depth, peer eligibility, token/cost ceilings) that only the hub can do. `transforms` are evaluated by the recipient's `ActorClient` at the inbox boundary — every transform runs inside the actor's address space, never in the hub. The actor uploads one `rule.json` at registration; the hub stores the whole file but only enforces the access+limits half, while pushing the transforms half down to the `ActorClient` via the `rule_changed` frame for local execution. This split is what makes the `apply` forms safe to use in a hosted multi-tenant hub: tenant code never executes in the hub process.

```jsonc
// hub/actors/{actor_id}/rule.json
{
  "version": 1,
  "access": {
    "inbound_from":  ["owner:*:*", "ag2:*:*"],         // who may open sessions to this actor
    "outbound_to":   ["*"],                              // who this actor may reach
    "session_types": {                                   // session types this actor may join
      "initiate": ["consulting", "conversation", "discussion"],
      "accept":   ["consulting", "conversation", "discussion", "broadcast"]
    },
    "subscribe": {                                       // subscription eligibility
      "sessions": "member-only",                         // member-only | public-within-hub | public
      "tasks":    "owner-or-member"
    },
    "knowledge": {                                       // cross-actor KnowledgeStore reads
      "expose":   ["/artifacts/public/**"],
      "readers":  ["ag2:*:*"]
    }
  },

  "limits": {
    "max_concurrent_sessions": 5,
    "max_concurrent_tasks":    20,
    "session_ttl_default":     "2h",
    "task_ttl_default":        "15m",
    "rate": { "per_minute": 60, "burst": 10 },
    "tokens": { "per_hour": 2_000_000 },
    "cost":   { "per_day_usd": 50 },
    "delegation_depth": 5
  },

  "transforms": [
    // Phase 5a ships named + python + http ↓
    { "stage": "pre_receive",
      "when":  { "event": "SessionInvite" },
      "apply": "redact_pii" },                                                          // named
    { "stage": "pre_send",
      "when":  { "session_type": "consulting" },
      "apply": { "python": { "module": "myorg.guards", "class": "PromptGuard",
                             "config": {"max_tokens": 8000} } } },                      // in-process Python
    { "stage": "post_receive",
      "apply": { "http":   "http://localhost:9000/audit" } },                           // sidecar HTTP
    // Phase 5b ships exec + ws ↓ (a 5a hub stores these verbatim and passes the envelope through)
    { "stage": "pre_send",
      "apply": { "exec":   ["/usr/local/bin/policy-filter", "--mode=strict"] } },       // long-lived subprocess
    { "stage": "pre_receive",
      "apply": { "ws":     "ws://localhost:9001/transform" } }                          // sidecar WebSocket
  ]
}
```

### 4.1 Stages

Transforms run at four well-defined stages, analogous to middleware on an envelope. Every stage runs inside an `ActorClient` — never in the hub process:

| Stage          | When                                                                | Where it runs                          | Can do                                   |
|----------------|---------------------------------------------------------------------|----------------------------------------|------------------------------------------|
| `pre_send`     | Sender's `ActorClient.send`, before the link `send` frame leaves    | sender's `ActorClient`                 | mutate envelope, reject, enrich metadata |
| `post_send`    | After the hub returns `accept` for the outbound envelope             | sender's `ActorClient`                 | side-effects (audit, local counters)     |
| `pre_receive`  | Recipient's `ActorClient` after `notify` arrives, before the handler | recipient's `ActorClient`              | mutate, reject, inject guidance          |
| `post_receive` | After the handler returns / receipt is posted                        | recipient's `ActorClient`              | side-effects (telemetry, counters)       |

A reject from `pre_send` short-circuits before the WS frame is sent (the local `send` raises). A reject from `pre_receive` short-circuits before the actor's handler is invoked, and the `ActorClient` posts a structured `nack` receipt back to the hub so the hub's WAL records the rejection.

The core `Transform` protocol is one async callable, `(envelope, ctx) → envelope | None` (return `None` to reject). `apply` carries one of five pluggable forms. Phase 5 ships the first three (**named / python / http**) as the MVP — those cover the majority of real use cases (reusable stateful logic, in-process Python, and language-agnostic sidecars), without the subprocess / bidirectional-stream lifecycle complexity of `exec` and `ws`. The remaining two forms ship in Phase 5b.

| Form     | Shape                                                  | Adapter            | Ships in | Use case                                   |
|----------|--------------------------------------------------------|--------------------|----------|--------------------------------------------|
| named    | `"redact_pii"` (string)                                 | `NamedTransform`     | **5a**   | Hub-registered reusable logic              |
| python   | `{ "python": { "module": "...", "class": "...", "config": {} } }` | `PythonTransform`    | **5a**   | In-process Python, zero-overhead            |
| http     | `{ "http": "http://localhost:9000/redact" }`             | `HttpTransform`      | **5a**   | Local sidecar, language-agnostic            |
| exec     | `{ "exec": ["./policy", "--mode=strict"] }`              | `ExecTransform`      | 5b       | Long-lived subprocess, JSON-lines stdio     |
| ws       | `{ "ws":   "ws://localhost:9001/transform" }`            | `WsTransform`        | 5b       | Low-latency sidecar with bidirectional flow |

Sidecar HTTP and subprocess forms are spawned **once** and reused for every envelope — per-envelope fork would be too slow. The framework owns lifecycle (start on first use, restart on crash, drain on hub shutdown). Transforms run in declaration order; any form may return a modified envelope, `None` to reject, or raise to fail-fast (the hub logs and treats unexpected errors as reject). Named transforms are how you ship reusable, stateful logic (rate limiter counts, circuit breaker state) without rebuilding the Plugin protocol; the other forms let you write transforms in any language without changing the hub.

A `rule.json` may reference an `exec` or `ws` transform before Phase 5b ships — the hub stores it verbatim per §4.3 rule-changed events. Until Phase 5b, an `ActorClient` that sees an unknown `apply` form logs a warning and treats the transform as pass-through (it does **not** reject envelopes that flow through it), so a rule written today does not become a blocker on day 5a ships.

This preserves the power of the old Plugin/Topology — stateful processing, composition, rejection — without the Pipeline/Fanout/Conditional/RouteDecision surface. A transform that needs to fan-out uses the `send()` API in its `post_receive` hook; there is no pipeline-level RouteDecision.

### 4.2 Hub defaults

The hub carries a default rule template. New registrations inherit it unless the registrant provides a rule explicitly. Changing defaults rewrites the defaults file, not individual rules.

### 4.3 Rule changes are events

Any write to `rule.json` emits a `RuleChanged` event on the hub stream so that in-memory caches, clients, and downstream observers invalidate. Actors may subscribe to rule changes on their own identity to react.

---

## 5. Sessions

A session is a stateful, durable, multi-turn container. Every communication between actors happens inside exactly one session.

### 5.1 Session types

Six session types. Delivery semantics only — what each participant's LLM actually *sees* per turn is an assembly concern (see §5.5).

| Type             | Recipients | Semantics                                                                                    | Replaces                         |
|------------------|------------|----------------------------------------------------------------------------------------------|----------------------------------|
| `notification`   | 1          | Sender fires a message. Recipient must ack. No reply.                                         | fire-and-forget event             |
| `broadcast`      | N          | Sender fires a message to many recipients. Each acks. No reply channel.                      | TopicMessage (one-shot)          |
| `consulting`     | 1          | Strict 1-question 1-response. Recipient cannot proactively send. Closes after the reply.     | `Hub._delegate` (current)        |
| `conversation`   | 1          | Bidirectional. Either side may send. Explicit close.                                          | multi-turn chats                 |
| `discussion`     | N          | Multi-actor turn-taking. Speaking order is parameterized (dynamic / static / round-robin).   | multi-agent discussions + `Pipeline` topology |
| `auction`        | N          | Sender posts task. Recipients bid. Sender selects one winner to continue 1:1.                | new — request-for-proposal       |

**Discussion parameters** make the previously-separate `pipeline` type disappear:

- `ordering`: `dynamic` (raise-hand; sender picks next speaker per turn) | `static` (pre-declared ordered list of participants) | `round_robin`
- `on_failure`: `abort` (stop the session on any participant failure) | `continue` (skip failures and proceed)
- `initial`: the actor that starts (defaults to the initiator)

What used to be `pipeline` — A→B→C sequential, each stage sees only the previous result, fail-fast — is now `discussion(ordering="static", on_failure="abort")` with a `PreviousOnlyInboxPolicy` attached to participants via assembly. The session WAL still holds everything; assembly decides each stage's LLM view. This separates delivery (session type) from visibility (assembly policy), so pipeline-style processing is a *configuration*, not its own concept.

Topics are likewise not a separate concept — a topic is a long-lived `broadcast` session with persistent subscription and per-subscriber cursors. Joining a topic is joining the session.

These 6 types are not a closed set — future types (e.g., `tournament` for elimination rounds) can ship as additional session-type adapters without changing the envelope surface.

### 5.2 Session state

```jsonc
// hub/sessions/{session_id}/metadata.json
{
  "session_id":   "01932f8b...",      // UUID7, hub-stamped
  "type":         "consulting",
  "creator_id":   "01932...",
  "participants": [
    { "actor_id": "01932...", "role": "initiator", "joined_at": "..." },
    { "actor_id": "01932...", "role": "respondent", "joined_at": "..." }
  ],
  "visibility":   "members-only",     // members-only | hub-public | discoverable
  "state":        "active",           // pending | active | paused | closing | closed | expired
  "created_at":   "...",
  "expires_at":   "...",
  "labels":       { "team": "research", "project": "abc" },
  "ordering":     null,               // discussion only: "dynamic" | "static" | "round_robin"
  "on_failure":   null,               // discussion only: "abort" | "continue"
  "parent_session_id": null           // nested sessions
}
```

WAL layout:

```
hub/sessions/{session_id}/
  metadata.json
  wal/
    00001.jsonl                 # chunked append-only log
    00002.jsonl
  tasks/
    {task_id}.jsonl             # task-specific index (subset of WAL)
  cursors/
    {actor_id}.json             # per-participant read position
  subscribers/
    {actor_id}.json             # non-participating observers
```

### 5.3 Session handshake

Session creation is an explicit protocol so that rules and types are enforced up front.

```
Initiator                           Hub                             Participants
   │                                 │                                   │
   │  POST /sessions {type,              │                               │
   │   participants, intent, ttl}        │                               │
   │────────────────────────────────▶│                                   │
   │                                 │ validate rules (initiate+accept) │
   │                                 │ allocate session_id (UUID7)      │
   │                                 │ write metadata.json              │
   │                                 │ create WAL                       │
   │                                 │   ── SessionInvite envelope ──▶  │
   │                                 │                                   │── notify()
   │                                 │                                   │
   │                                 │◀───── InviteAck/Reject ──────────│
   │                                 │                                   │
   │  session_id + session handle    │                                   │
   │◀────────────────────────────────│                                   │
   │                                 │                                   │
```

- For `consulting`/`conversation`/`notification` (single-recipient types), the hub waits for the recipient ack before returning a ready session.
- For multi-recipient types, the hub returns an envelope-streaming session handle that surfaces invite acks/rejects as they arrive; the initiator decides the quorum policy in the session request.
- Rejected invites do not fail the session by default; the initiator sees who accepted and can proceed or abort.
- For `discussion` with `ordering="static"`, the initiator provides the participant order in the creation request (`participants: [{actor_id, order}, ...]`); the adapter notifies participant N+1 only after participant N's response envelope has landed in the WAL (or is skipped per `on_failure`).

### 5.4 Delivery semantics per type

The hub ships one `SessionAdapter` per type that enforces delivery rules. The adapter's job is tiny:

- Validate envelope direction (who can send to whom, when).
- Enforce one-shot semantics (notification, consulting).
- Manage speaking order (discussion — including its static/pipeline mode — and auction).
- Decide close conditions.

### 5.5 Adapter extensibility

The six built-in session types are **not a closed set**. An operator or a third-party package can ship a new type by implementing the `SessionAdapter` protocol and registering it with the hub via an explicit API — no entry_points magic, no auto-discovery, no config-file loaders. Registration is in Python because adapters are Python code.

```python
class TournamentAdapter:
    session_type = "tournament"              # plain string — arbitrary name

    def validate_create(self, metadata: SessionMetadata) -> None: ...
    def validate_send(self, metadata, envelope, prior) -> None: ...
    def on_accepted(self, metadata, envelope, prior) -> AdapterResult: ...

hub = Hub(store)
hub.register_adapter(TournamentAdapter())
```

Rules for operators adding a type:

1. **`session_type` is a plain string**, not a `SessionType` enum member. The built-in enum (`SessionType.CONSULTING` etc.) is kept as a canonical namespace of built-in names; its members double as strings because the enum subclasses `str`, so existing code referencing `SessionType.X` continues to work untouched.
2. **Registration replaces collisions.** If a name is already registered, `hub.register_adapter(...)` replaces the prior adapter and logs a warning. Built-in adapters are pre-registered when the `Hub` is constructed; overriding one is allowed (an operator can ship a stricter `ConsultingAdapter` for their deployment).
3. **`SessionMetadata.type` is a plain string** on the wire and on disk. The hub looks up the adapter by string at envelope-post time; an unknown type raises `SessionTypeError("no adapter registered for 'X'")`. The wire format is unchanged (the field has always been serialized as `.value`, i.e. a string).
4. **Rule patterns already work.** `rule.access.session_types.initiate/accept` are lists of glob strings, so `"tournament"` or `"ag2:*"` just works without any rule-schema change.
5. **No runtime discovery.** Operators import the adapter class and call `register_adapter`. Third-party packages ship their adapters as ordinary Python classes; the operator decides which ones to load.

This mirrors the AuthAdapter extensibility in §13 (explicit registration of pluggable validators) — same shape, same explicit-beats-implicit stance.

### 5.6 Sessions vs agent memory

Crucial separation from V2: **a network session is not an agent session.** The session's WAL is the source of truth for what happened between actors. It does **not** decide what an LLM sees in each round.

It is the **agent harness** (AssemblyPolicy) that decides what to feed the LLM. A discussion participant may choose to see only its own turns + a summary of others, or the whole transcript, or a filtered subset. Framework core already owns this via `AssemblyPolicy`. V3 adds a `SessionInboxPolicy` that reads the session WAL and produces model events for the assembly chain.

---

## 6. Tasks (network-level)

A **Task** in V3 is a network-level, session-owned unit of long-running work with a hub-managed state machine. It is **not** the same thing as the actor-local `run_subtask` feature that already exists in framework-core `Actor` — that one is untouched and stays out of the network layer (see §6.5).

Tasks are first-class but **live inside a session** — you cannot have a network Task without a session.

### 6.1 Task shape

`TaskMetadata` is the hub's durable per-task record. It is rewritten on every state transition at `hub/tasks/{task_id}/metadata.json`; the in-memory `Hub._tasks` cache is the fast path and is rebuilt from disk on cold restart.

```jsonc
{
  "task_id":         "01932f8c...",        // UUID7, hub-stamped at create time
  "session_id":      "01932f8b...",        // owning session (every task lives in one session)
  "owner_id":        "01932...",           // actor executing the task
  "requester_id":    "01932...",           // actor that called Hub.create_task
  "spec": {
    "title":       "Fetch and summarize PubMed papers",
    "description": "Pull the top-10 papers and synthesize…",
    "spec_type":   "research",            // routes to client.on_task("research")
    "phases": [
      { "id": "fetch",     "description": "Retrieve source papers"},
      { "id": "summarize", "description": "Produce per-paper summaries"},
      { "id": "synthesize","description": "Final synthesis"}
    ],
    "payload":     { "top_k": 10 }        // arbitrary JSON-serializable body
  },
  "state":           "running",            // created | running | paused | completed | failed | cancelled | expired
  "current_phase":   "summarize",
  "created_at":      "...",
  "expires_at":      "...",                // hub-enforced TTL
  "started_at":      "...",                // stamped on first running transition
  "completed_at":    null,                 // stamped on terminal transition
  "last_progress_at":"...",                // refreshed on every ag2.task.progress event
  "progress":        { "docs": 4, "pct": 0.5 },  // merged from progress events
  "result":          null,                 // terminal value for completed tasks
  "error":           null                  // terminal message for failed tasks
}
```

`blocking` is **not** stored on the task — it is a client-side argument to `Session.create_task(spec, blocking=True)` that controls whether the call awaits the terminal state or returns the `Task` handle immediately. The hub runs every task the same way regardless of how the caller is waiting.

### 6.2 Hub + actor cooperation

- **Hub owns the state machine.** Every transition (`created → running → completed`, etc.) is applied by the hub. Direct hub methods (`Hub.create_task`, `Hub.cancel_task`, `Hub.expire_due_tasks`) drive hub-initiated transitions; actor-emitted task envelopes (`ag2.task.*`) drive owner-initiated ones. Both paths share the same `_apply_task_event` mutation + on-disk rewrite.
- **Actors report progress.** The owner emits `ag2.task.phase_entered`, `ag2.task.phase_completed`, `ag2.task.progress`, `ag2.task.result`, and `ag2.task.error` envelopes into the session via the `Task` client handle. The hub updates the task record in place and the same envelope is fanned out to every session subscriber.
- **Task events bypass the session adapter.** The hub's `post_envelope` recognizes `ag2.task.*` and routes them through a task-event branch that runs the access / rate / depth / inbox checks but **skips** `adapter.validate_send` and `adapter.on_accepted`. This is what keeps consulting's 1Q1R rule (and every other adapter delivery rule) orthogonal to task lifecycle: a task owner can emit unlimited phase / progress envelopes inside a consulting session without the adapter trying to auto-close it.
- **Hub enforces TTL.** `Hub.expire_due_tasks(now=...)` walks the in-memory cache, transitions every non-terminal task whose `expires_at` is past `now` to `expired`, and emits a broadcast `ag2.task.expired` envelope. Operators register the sweeper as an `IntervalWatch` callback on a framework-core `Scheduler` instance — the hub does not own a scheduler of its own, which keeps Phase 3a's session TTL sweeper and Phase 4's task TTL sweeper as two independent registrations against the same scheduler.
- **Blocking vs background is the caller's choice.** `Session.create_task(spec, blocking=True)` opens a session subscription from the post-create WAL cursor, filters by `task_id`, and resolves on the first envelope whose event type is in `TASK_TERMINAL_EVENT_TYPES`. `blocking=False` returns a `Task` handle immediately. The blocking path is layered purely on top of the non-blocking path — no new frames, no new hub primitives, just the same subscription mechanism `Session.ask` uses with a `task_id` predicate.
- **Phases are the checkpoint boundary.** `current_phase` is updated on every `ag2.task.phase_entered` event and persisted on disk. A restarting hub reads the task back through `Hub.hydrate()` and the owner can resume from the last committed phase. If the task spec declared a phase plan up front, the hub additionally stamps `started_at` / `completed_at` on each declared phase as the events flow in.
- **Session-close task cascade.** When a session transitions to `CLOSED` (explicit close, TTL expiry, or adapter-driven auto-close), every non-terminal task in the session is transitioned to `cancelled` with `reason="session_closed"` *before* the `SessionClosed` broadcast lands in the WAL. Subscribers see a clean `task.cancelled → session.closed` ordering on replay.

### 6.3 Task event namespace

Eight stable event names ride the session envelope stream. Every event carries `task_id` on the envelope-level field (Phase 1 reserved `Envelope.task_id` for exactly this) plus an `event_data` payload specific to the event type. All eight names are pre-registered as `EventRegistry` built-ins so a strict-mode hub accepts them without operator registration.

| Event                         | Direction                            | Purpose                                                         |
|-------------------------------|--------------------------------------|-----------------------------------------------------------------|
| `ag2.task.assigned`           | hub → owner (session-wide fan-out)    | Hub-emitted after `Hub.create_task` succeeds. Carries `task_id` + `spec`. |
| `ag2.task.phase_entered`      | owner → session                       | Advances `current_phase`. Stamps `started_at` on first transition. |
| `ag2.task.phase_completed`    | owner → session                       | Marks a declared phase done (timestamp only).                  |
| `ag2.task.progress`           | owner → session                       | Merge-updates the `progress` dict. State stays.                |
| `ag2.task.result`             | owner → session                       | Terminal. Transitions state → `completed`.                      |
| `ag2.task.error`              | owner → session                       | Terminal. Transitions state → `failed`, stores `error`.         |
| `ag2.task.cancelled`          | hub → session (on requester call)     | Terminal. Transitions state → `cancelled`.                       |
| `ag2.task.expired`            | hub → session (on TTL sweep)          | Terminal. Transitions state → `expired`.                         |

Task creation is a **direct `Hub.create_task` method** (symmetric with `Hub.create_session`), not a wire envelope — auditors and subscribers see the task starting from the `ag2.task.assigned` envelope the hub posts as a side-effect of creation.

### 6.4 Subscriptions

Tasks do not have a separate subscription channel — they ride the session WAL like every other envelope. Any actor with the right `rule.access.subscribe.sessions` policy on the owning session opens a normal `Session.subscribe(...)` and filters envelopes by `task_id`. The blocking-wait path on `Task.wait()` uses exactly this — a session subscription with a `task_id` predicate that resolves on the first terminal-event envelope. Polling without a subscription is `Hub.peek_task(task_id)` for in-process callers; `GET /v1/tasks/{id}` lands as part of Phase 3b's full HTTP surface.

### 6.5 Task (network) vs subtask (actor-local)

These are two unrelated features that happen to share the word "task." V3 does not collapse them.

| Concept              | Where it lives           | What it does                                                        | Requires Hub?       |
|----------------------|--------------------------|---------------------------------------------------------------------|---------------------|
| `run_subtask` / `run_subtasks` | `autogen/beta/actor.py` (framework core) | Spawns a private child `Agent` against a `MemoryStream`, bridges chunks, returns text. Zero network, zero envelopes. | No. Actor runs standalone. |
| `Task` (this section)          | `autogen/beta/network/`   | Session-owned, hub-tracked lifecycle with phases, TTL, subscriptions. Envelope-carried events. | Yes. Session + hub required. |

**Framework-core subtasks are explicitly preserved.** The existing `run_subtask` / `run_subtasks` tools on `Actor` remain the way an actor runs a self-contained child Agent without touching the network. They are not reimplemented on top of sessions, because forcing hub dependence on a feature that should work standalone would be a regression.

When an actor is registered with a hub, it gains the `run_task` / `start_task` LLM tools in *addition* to the local `run_subtask`. The LLM picks based on intent:

- **local compute, private context** → `run_subtask` (no hub, no observability, no envelopes)
- **delegated work to a known actor, with lifecycle/observability/TTL** → `run_task` / `start_task`

The names are similar but the mechanisms are orthogonal.

---

## 7. Envelopes, Delivery, and the Inbox

Envelopes still exist — they wrap events with network metadata — but they are simpler and universal:

```jsonc
{
  "envelope_id": "01932f8d...",        // UUID7, hub-stamped on accept
  "session_id":  "01932f8b...",
  "task_id":     null,                 // optional, task-scoped events set this
  "sender_id":   "01932...",
  "recipient_id":"01932...",           // null for broadcast within session
  "causation_id":"01932f8d-prev...",
  "trace_id":    "01932...",
  "priority":    "normal",
  "created_at":  "...",
  "ttl_seconds": 60,
  "event":       { "type": "ag2.msg.text", "data": { ... } },
  "signatures":  [ ... ]               // optional, for cross-hub federation
}
```

Notes:

- `priority` is a string enum (`background | normal | urgent`) by default. Custom priority schemes remain an opt-in per hub, but the framework no longer exposes `PriorityScheme`/`ConflictResolver` in the public API.
- Wire format is version-tagged, same as V2, but event types are **stable registered names** (`ag2.msg.text`) instead of Python-qualified names. Each hub keeps an event registry; custom events register a canonical name at startup.
- Envelopes carry only WAL-friendly data — no Python types, no runtime-only fields.

### 7.1 Inbox

Each actor has a hub-owned inbox at `hub/actors/{actor_id}/inbox/`. When the hub delivers an envelope to an actor:

1. Hub writes the envelope into the inbox (`pending/{envelope_id}.json`).
2. Hub pushes a notify() frame to the actor over the actor's live transport (local queue, WebSocket).
3. Actor's receipt is recorded (`received/{envelope_id}.json`) — the notify frame is not `ack`; the actor posts an explicit receipt back when it has durably accepted the envelope (for retries and idempotence).
4. Actor processes on its own schedule. When done, it writes envelopes back via the session.

The inbox being a first-class file system structure means:

- Crash-recovery is trivial — the actor rebuilds its queue from `pending/` minus `received/`.
- The hub can GC received envelopes per retention policy.
- Debugging is `ls`.

**Inbox configuration absorbs the V2 channel knobs.** What V2 modeled as `BufferedChannel(max_buffer, overflow_policy)` and `PriorityChannel(scheme)` is now per-actor inbox configuration declared inside `rule.limits` (or inherited from the hub default rule):

```jsonc
"inbox": {
  "max_pending":      1000,
  "overflow":         "reject",          // reject | spool | drop_oldest | drop_newest
  "ordering":         "priority",        // fifo | priority
  "priority_scheme":  "default"          // hub-registered scheme name
}
```

The hub picks delivery order from this config when draining `pending/`. There is no separate `Channel` object — `Inbox + Link` is the only transport pair.

### 7.2 notify() — the inversion

**Crucial change:** the hub no longer calls `agent.ask(task)` directly. The hub never knows what an actor does to produce an answer. Instead:

1. Hub delivers an envelope via `notify(envelope) → receipt`.
2. The actor's default notify handler (per session type) picks up the envelope, runs whatever agent logic it wants (a single `Agent.ask`, a tool chain, nothing at all), and writes response envelopes back into the session.
3. For `consulting`/`notification`, the handler must produce exactly one response and close. For `conversation`, it produces a response and leaves the session open. For long-running work, it creates a task and streams `TaskProgress`.

This makes local and remote actors symmetric: both receive envelopes, both post responses, both get the same rule/transform pipeline. `RemoteAgent` disappears.

**Blocking `send()` is implemented on top of notify().** The initiator's `send()` posts an envelope into the session, then awaits the correlated response envelope (by `causation_id`) on its own session subscription, with a timeout. If the caller wants non-blocking, it calls `send_async()` and subscribes separately.

Concretely, `Session.ask(content)` in the actor API (section 10) expands to: (a) build an envelope from `content`, (b) post it via a `send` frame and read back the `envelope_id` the hub stamps on `accept`, (c) open a temporary `subscribe` frame filtered by `causation_id == envelope_id`, (d) await the first matching envelope (streaming `chunk` frames are yielded in between), (e) close the subscription and return the accumulated response. The timeout defaults to the session's remaining TTL. This is the single mechanism that bridges the high-level `ask()` ergonomics with the low-level notify/receipt protocol — there is no separate "synchronous" path.

### 7.3 Default handlers

The framework ships default notify handlers per session type that wrap the framework-core `Actor.ask` for the simple case. Handlers receive an `ActorClient` (which holds the `Actor`, the registered identity, and the session API), not a raw Actor, so they can post replies back into the session without knowing the transport. The `ActorClient` has already run `pre_receive` transforms before invoking the handler:

```python
async def handle_consulting(envelope, client: ActorClient):
    session = client.session_for(envelope)
    reply = await client.actor.ask(envelope.event.content)
    await session.send(reply.body, causation_id=envelope.envelope_id)

async def handle_conversation(envelope, client: ActorClient):
    session = client.session_for(envelope)
    # A SessionInboxPolicy on the actor's assembly chain feeds the session WAL
    # into the LLM view, so actor.ask sees prior turns without any special wiring.
    reply = await client.actor.ask(envelope.event.content)
    await session.send(reply.body, causation_id=envelope.envelope_id)
```

Advanced users override these via `client.on(SessionType.DISCUSSION)` to implement custom session protocols without touching the hub.

---

## 8. Persistence — Extended KnowledgeStore

The hub is backed by a `KnowledgeStore` (the same protocol framework core already uses for actors). No new persistence protocol. `KnowledgeStore` is extended with three new methods:

```python
class KnowledgeStore(Protocol):
    # existing
    async def read(self, path: str) -> str | None: ...
    async def write(self, path: str, content: str) -> None: ...
    async def list(self, path: str = "/") -> list[str]: ...
    async def delete(self, path: str) -> None: ...
    async def exists(self, path: str) -> bool: ...

    # new, required
    async def append(self, path: str, content: str) -> int: ...       # atomic append, returns offset
    async def read_range(self, path: str, start: int, end: int | None) -> str: ...

    # new, optional (see note)
    async def on_change(self, path: str, callback) -> ChangeSubscription: ...  # filesystem change notifications
```

`append`/`read_range` make the WAL efficient without a new abstraction. `on_change` powers live subscriptions, cache invalidation, and cross-process coherence: when a session WAL file grows, subscribers see the new lines; when a rule file is rewritten, all caches invalidate.

> **Naming note:** the knowledge-store `on_change` / `ChangeSubscription` API is deliberately named *away* from "watch" so it does not collide with `autogen.beta.watch.Watch` — the event- and time-pattern trigger type used by the framework-core `Scheduler`. The two are orthogonal: `on_change` is filesystem reactivity; `Watch` drives scheduled and event-pattern-based callbacks.

**`on_change` is optional.** Backends that cannot observe changes efficiently (S3, flat memory dicts) return a :class:`NoopChangeSubscription`, and the hub automatically falls back to short-interval polling for those paths. Implementations signal this by either omitting `on_change` or returning the sentinel. Local disk, SQLite, and Redis stores implement it natively. Actor-local stores that do not need it simply ignore the method.

Implementations:

| Backend          | Use case                              |
|------------------|---------------------------------------|
| `MemoryStore`      | dev, tests                            |
| `DiskStore`        | single-host persistent                |
| `SqliteStore`      | small production, desktop apps        |
| `RedisStore`       | low-latency multi-process             |
| `S3Store`          | durable, large-scale                  |
| `FsdbStore` (fn)    | foundationdb — strong multi-tenant    |

The same store serves both actor private knowledge (unchanged) and hub state. The scoping is purely by path prefix; a hub using a shared store does `hub/...` and actors do `actor/{id}/...` without collision.

### 8.1 In-memory cache

Every hub read of a rule, identity, session metadata, or inbox cursor on the hot path would otherwise hit the store. The hub maintains a small in-memory mirror of these hot paths, populated lazily and invalidated via `on_change`. Identity records are especially cache-friendly because they are immutable per registration; only `runtime.json` changes on heartbeat, and that file lives outside the identity cache. The cache is **not** authoritative — a cold restart rebuilds it by reading the store.

### 8.2 Archival / GC

Closed sessions, expired tasks, and acknowledged inbox entries are moved to cold storage (`hub/archive/...`) by a background sweeper. The sweeper reuses `CompactStrategy` from framework core — the WAL is reduced to a summary + final result, and the original is archived. Retention is configured per hub, not per session.

---

## 9. Communication Protocol

Everything runs over two wire protocols.

### 9.1 HTTP (stateless)

JSON-over-HTTP for CRUD and queries:

| Method | Path                                      | Purpose                                               |
|--------|-------------------------------------------|-------------------------------------------------------|
| POST   | `/v1/actors`                                | Register an actor (`ActorIdentity` + `rule` + optional `SKILL.md`) |
| PATCH  | `/v1/actors/{id}/runtime`                   | Update mutable runtime binding (heartbeat / address)  |
| DELETE | `/v1/actors/{id}`                           | Unregister                                            |
| GET    | `/v1/actors?capability=&query=`             | Discover                                              |
| GET    | `/v1/actors/{id}`                           | Describe                                              |
| PUT    | `/v1/actors/{id}/rule`                      | Replace rule                                          |
| POST   | `/v1/sessions`                              | Create session (handshake initiator half)             |
| GET    | `/v1/sessions/{id}`                         | Get session metadata                                  |
| POST   | `/v1/sessions/{id}/close`                   | Close                                                 |
| GET    | `/v1/sessions/{id}/wal?since=`              | Read WAL range (bulk)                                 |
| GET    | `/v1/sessions?state=&participant=&type=`    | List sessions (activity / operational view)           |
| POST   | `/v1/sessions/{id}/force-close`             | Admin-only force close                                |
| GET    | `/v1/tasks/{id}`                            | Task status                                           |
| GET    | `/v1/tasks?owner=&state=`                   | List tasks (activity / operational view)              |
| POST   | `/v1/tasks/{id}/cancel`                     | Cancel                                                |
| GET    | `/v1/actors/{id}/activity`                  | Recent sessions + tasks for an actor                  |
| GET    | `/v1/actors/{id}/knowledge/{path}`          | Cross-actor knowledge read (rule-gated)               |
| GET    | `/v1/admin/health`                          | Hub liveness                                          |
| GET    | `/v1/admin/metrics`                         | Counts of active sessions, tasks, connections         |

All endpoints accept an actor identity header (API key / JWT / mTLS cert fingerprint) and check against the caller's rule.

### 9.2 WebSocket (stateful)

Each actor holds one long-lived WebSocket to the hub (`wss://hub/v1/actors/{actor_id}/link`). The protocol is a framed bidirectional stream:

| Frame             | Direction       | Purpose                                                                      |
|-------------------|-----------------|------------------------------------------------------------------------------|
| `hello`           | actor → hub     | authenticate (presents `identity.auth` claim), state last-known sequence     |
| `notify`          | hub → actor     | deliver envelope (carries session_id + envelope_id)                          |
| `receipt`         | actor → hub     | ack envelope, or nack with reason (rejected, throttle, retry)                |
| `send`            | actor → hub     | post an envelope into a session (carries `idempotency_key` if supplied)      |
| `accept`/`error`  | hub → actor     | response to `send`                                                           |
| `chunk`           | hub → actor     | streaming content token (for consulting / conversation replies)              |
| `subscribe`       | actor → hub     | open a push subscription to a session or task; optional `since` cursor for resume |
| `unsubscribe`     | actor → hub     | close subscription                                                           |
| `event`           | hub → actor     | subscription delivery                                                        |
| `rule_changed`    | hub → actor     | push updated `rule.transforms` to the `ActorClient` for local execution      |
| `ping`/`pong`     | both            | heartbeat                                                                    |

`subscribe` carries an optional `since` cursor (WAL offset or `event_id`); when present the hub replays everything from that cursor before switching to live push. This collapses what would otherwise be two operations (HTTP `GET wal?since=` followed by `subscribe`) into one frame, so reconnect-and-resume is a single round trip.

Local, in-process actors use the same frames over an in-memory duplex transport — the notify handler never knows the difference.

### 9.3 Local transport parity

```python
from autogen.beta.network import LocalLink, WsLink

# Same surface, different transport
link = LocalLink(hub)                       # in-process
link = WsLink("wss://hub.example.com")      # cross-host
await actor.connect(link)
```

A locally-registered actor is identical to a remotely-registered actor modulo the `Link` it was constructed with.

### 9.4 Streaming

Chunks (`ModelMessageChunk` in framework core) become `chunk` frames inside the receiver's notify flow. For `consulting`, streaming is a sequence of `chunk` frames followed by a final `send` with the complete response envelope. The receiver returns the final envelope; `chunk` frames are emitted onto the session WAL but marked transient so they do not blow up the log (same `__transient__` mechanism framework core already uses).

---

## 10. Network Client API

V3 does **not** define a new `Actor` class. The framework-core `autogen.beta.Actor` (and plain `autogen.beta.Agent`) is unchanged — it already has observers, knowledge, assembly, compaction, aggregation, subtasks, and HITL, and it works standalone without a hub. The network layer's job is to *attach* an existing `Actor` to a hub through a pair of thin clients.

### 10.1 Two paired clients

The actor process holds **two** network clients. They are peers, not nested:

| Client          | Owns                                                                                          | Direction                       | Transport surface                                                      |
|-----------------|-----------------------------------------------------------------------------------------------|---------------------------------|------------------------------------------------------------------------|
| `HubClient`     | Connection lifecycle, registration, discovery (`find`, `describe`), session creation, outbound `send` | actor → hub                     | HTTP CRUD endpoints + WS `hello` / `send` / `subscribe` / `receipt`    |
| `ActorClient`   | One per registered identity. Holds the underlying `Actor`, the `ActorIdentity`, the cached `rule.transforms`, the inbox, the per-session-type handler registry. **Runs all four `transforms` stages locally.** | hub → actor (and replies)       | Inbound WS `notify` / `event` / `chunk` / `rule_changed` + outbound receipts and follow-up `send`s via the same link |

Both live in the actor's address space. They share the underlying `Link` to the hub but expose disjoint APIs. The split makes the rule/transform isolation point obvious: every transform — Python class, exec subprocess, sidecar HTTP/WS — runs inside `ActorClient`, never in the hub. The hub only enforces `rule.access` and `rule.limits` (§4).

A single `HubClient` owns one connection and may produce many `ActorClient`s (one per identity registered through that connection). A single Python `Actor` may own many `ActorClient`s across different hubs and different identities.

### 10.2 Registration

```python
from autogen.beta import Actor                                    # unchanged framework-core class
from autogen.beta.network import (
    HubClient, ActorClient, LocalLink, WsLink,
    ActorIdentity, Rule, SessionType,
)

# 1. Build a normal framework-core Actor exactly as today.
#    The Actor itself has no actor_id and knows nothing about hubs.
actor = Actor(
    "ag2:researcher:1",
    config=OpenAIConfig(model="gpt-4o"),
    knowledge=KnowledgeConfig(store=DiskKnowledgeStore("/var/agents/researcher")),
    observers=[TokenMonitor(50_000)],
)

# 2. Open a hub connection.
hub = HubClient(link=LocalLink(local_hub))             # or WsLink("wss://hub.example.com")

# 3. Build an identity. Profile + capabilities + auth in one object.
research_id = ActorIdentity(
    name="ag2:researcher:1",
    owner="ag2", version="1",
    capabilities=["research", "summarization"],
    summary="Produces cited literature reviews.",
    auth={"scheme": "api_key", "key_fingerprint": "sha256:..."},
    skill_md="## Researcher\n\nPrefers consulting sessions with clear scope...",
)

# 4. Register the actor under that identity. The hub stamps a fresh actor_id
#    and returns an ActorClient bound to (actor, identity, hub).
client: ActorClient = await hub.register(
    actor,
    identity=research_id,
    rule=Rule(access=..., limits=..., transforms=...),
)

# 5. Open a session via the ActorClient.
async with client.open(SessionType.CONSULTING, target="ag2:writer:1", ttl="5m") as s:
    reply = await s.ask("Here are the findings. Write a 500-word summary.")
    print(reply.content)

# 6. Unregister when done.
await client.unregister()
```

### 10.3 Multi-identity, multi-hub

The same Python `Actor` can simultaneously hold a second `ActorClient` against another hub under a different identity, with different capabilities and credentials:

```python
enterprise_hub = HubClient(link=WsLink("wss://hub.acme-corp.com"))

enterprise_id = ActorIdentity(
    name="acme:senior_research_lead",
    owner="acme", version="2",
    capabilities=["research", "compliance-review"],
    auth={"scheme": "mtls", "key_fingerprint": "sha256:..."},
    skill_md="## Senior Research Lead\n\nFollows ACME compliance protocols...",
)

enterprise_client = await enterprise_hub.register(
    actor,
    identity=enterprise_id,
    rule=Rule(access=..., limits=..., transforms=...),
)
```

The result: **one Python `Actor`, two `ActorClient`s, two `actor_id`s** (one per registration), two independent rules, two independent inboxes. Each `ActorClient` enforces its own `transforms` independently in the actor's process. Neither hub is aware of the other.

### 10.4 API surface

| Object          | Method                                                    | Purpose                                                                  |
|-----------------|-----------------------------------------------------------|--------------------------------------------------------------------------|
| `HubClient`     | `register(actor, identity, rule)`                          | Register an `Actor` under an `ActorIdentity`. Returns `ActorClient`.    |
| `HubClient`     | `find(query, capability)`                                  | Discovery                                                                |
| `HubClient`     | `describe(name)`                                           | Pull `ActorIdentity` + SKILL for another actor                           |
| `HubClient`     | `close()`                                                  | Tear down the underlying link                                            |
| `ActorClient`   | `actor`                                                    | The bound `Actor` (read-only attribute)                                  |
| `ActorClient`   | `identity`                                                 | The registered `ActorIdentity` (read-only attribute)                     |
| `ActorClient`   | `open(type, target, ...)`                                  | Open a new session. Returns async context `Session`.                     |
| `ActorClient`   | `accept_invites(handler)`                                  | Register a notify handler for incoming invites                           |
| `ActorClient`   | `on(session_type)`                                         | Decorator: override the default notify handler for a session type        |
| `ActorClient`   | `inbox_iter()`                                             | Low-level inbox polling for custom handlers                              |
| `ActorClient`   | `unregister()`                                             | Leave the hub                                                            |
| `Session`       | `send(envelope_or_content)`                                | Post into the session (`pre_send` transforms run before the link frame)  |
| `Session`       | `ask(content)`                                             | `send` + await correlated reply (see §7.2)                               |
| `Session`       | `subscribe(since=None)`                                    | Stream envelopes from the session, optionally resuming from a cursor     |
| `Session`       | `create_task(spec)`                                        | Create a network Task inside this session                                |
| `Session`       | `close()`                                                  | Explicit close                                                           |
| `Task`          | `progress(phase, message)` / `result(v)` / `fail(err)`     | Lifecycle transitions reported to the hub                                |

`ActorClient` never owns or mutates `Actor` internals — it calls `actor.ask(...)` from notify handlers like any other caller. The Actor stays a plain framework-core object; everything network-aware is layered on top via the two clients.

### 10.5 LLM tool surface

The LLM surface — the tools the agent actually calls inside a turn — is a **thin** auto-injected wrapper. The network verbs are only injected when the actor is running inside an `ActorClient`-driven notify handler. Framework-core tools (`run_subtask`, `knowledge`) remain available the whole time regardless of hub attachment.

Network verbs (added when an `ActorClient` is dispatching a turn):

| Tool             | Action                                                        |
|------------------|---------------------------------------------------------------|
| `find_actors`    | search by capability/query                                    |
| `describe_actor` | get `ActorIdentity` + SKILL                                   |
| `open_session`   | open a new session (type, target, intent)                     |
| `say`            | send content into current session                             |
| `listen`         | peek the inbox for pending messages                           |
| `run_task`       | create + block on a network Task inside current session       |
| `start_task`     | create + return task_id (non-blocking)                        |
| `track_task`     | query network Task state                                      |
| `read_session`   | read bounded WAL slice                                        |
| `leave`          | close / leave session                                         |

Framework-core tools (always available on `Actor`, hub or no hub):

| Tool            | Action                                                       |
|-----------------|--------------------------------------------------------------|
| `run_subtask`     | Spawn a private child `Agent` against a local stream (§6.5) |
| `run_subtasks`    | Parallel variant                                             |
| `knowledge`       | Read/write/list/delete in actor's own `KnowledgeStore`       |

Ten verbs, one for each primary operation. Each verb has a typed argument schema so that coding agents can produce valid calls without format drift. No more mega `network(action, target, topic, message)` string switch. These verbs are **auto-injected by the `ActorClient` when the actor executes inside a notify handler** — they are not on the `Actor` class. If the actor runs standalone (no hub), none of these verbs appear and the actor uses only its framework-core tools (`run_subtask`, `knowledge`, user tools).

Framework-core DI makes session/task context available to user-defined tools during a handler turn:

- `Session` (current session handle) — `Annotated[Session, Inject()]`
- `Task` (current task, if running inside one) — `Annotated[Task, Inject()]`
- `HubClient` — `Annotated[HubClient, Inject()]`
- `ActorClient` (actor + identity + rule view) — `Annotated[ActorClient, Inject()]`

These injections are populated by the `ActorClient` setting up the `Context` before calling `actor.ask(...)`; they vanish outside network turns so standalone runs stay unaffected.

---

## 11. Hub API Surface (Summary)

Responsibilities:

1. **Registry.** Identity lifecycle, `actor_id` allocation, discovery index, `AuthAdapter` validation at handshake.
2. **Rule storage and access/limits enforcement.** Stores the full `rule.json` per actor; enforces `access` (peer eligibility, session-type eligibility, knowledge exposure) and `limits` (concurrent sessions/tasks, rate, tokens, cost, delegation depth, TTL defaults). Pushes the `transforms` portion to each `ActorClient` via `rule_changed` for local execution.
3. **Session lifecycle.** Handshake, state machine, WAL, close, expire.
4. **Task lifecycle.** State machine, TTL enforcement, subscription fan-out.
5. **Inbox.** Envelope routing, receipts, retries.
6. **Transport.** Local and WS link management.
7. **Persistence.** KnowledgeStore-backed FS, cache, `on_change`-driven invalidation.
8. **Scheduler (system).** Runs TTL sweeps, archival, watch-based triggers. Still framework-core `Scheduler`, driven by its own `Watch` primitives (time/event triggers) — distinct from the store's `on_change` filesystem reactivity.
9. **Admin.** Health, metrics, audit log.

The hub does **not**:

- Call `Agent.ask()` directly — that is handled by the notify handler attached via the `ActorClient`.
- **Run `rule.transforms`** — those execute inside each recipient's `ActorClient`, in the actor's address space, so tenant code never enters the hub process.
- Subclass or replace framework-core `Actor` — it only holds registration state and dispatches envelopes.
- Know anything about LLM providers, tools, assembly, prompt templates.
- Own the agent's context window — the assembly policy does.
- Participate in turn-by-turn model calls.

This is the strict separation the V2 review already started but didn't finish. The network layer is **additive** on top of the existing `Agent`/`Actor` classes.

---

## 12. What Gets Removed

| Removed                                                                    | Replaced by                                                                                                              |
|----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `Topology`, `Pipeline`, `Fanout`, `Conditional`                            | `rule.transforms` (Phase 5a MVP: named / python / http · Phase 5b: exec / ws)                                            |
| `Plugin` protocol, `HubContext`                                            | `Transform` protocol + hub-registered named transforms; transforms get a local `TransformContext` from their `ActorClient` |
| `RouteDecision`                                                            | A transform may emit follow-up envelopes via `ActorClient.send`                                                          |
| `TopicPlugin`, topic events                                                | Broadcast sessions with cursor subscription                                                                              |
| `pipeline` session type                                                    | `discussion(ordering="static", on_failure="abort")` + `PreviousOnlyInboxPolicy`                                          |
| `RemoteAgent`                                                              | Every registered identity's `runtime.json` carries a binding; hub dispatches by binding                                  |
| `Passport` + `Resume` + separate SKILL                                     | Single `ActorIdentity` (profile + capabilities + auth) with `SKILL.md` as a sidecar; `runtime` split out into `runtime.json` |
| `ActorInfo` (V2 dataclass)                                                 | `ActorIdentity`                                                                                                          |
| `ActorBinding`                                                             | `ActorClient` — paired with `HubClient` as the two-client narrative; rule transforms execute here                        |
| Cross-actor knowledge via `_exposed_paths`                                 | `rule.access.knowledge` + `GET /v1/actors/{id}/knowledge/...`                                                            |
| `Channel` / `LocalChannel` / `BufferedChannel` / `PriorityChannel` (V2)    | `Inbox` + `Link`. Buffer policy and priority ordering become per-actor `inbox` config under `rule.limits` (§7.1).        |
| `PriorityScheme` / `ConflictResolver` public API                           | Priority is a tagged enum on the envelope; schemes are internal                                                          |
| `network(action, ...)` mega-tool                                           | 10 typed network verbs                                                                                                   |
| `Hub._delegate` / `Hub.ask`                                                | `session.send` / `session.ask` + notify flow                                                                             |
| `HttpChannel`                                                              | `WsLink` — WebSocket duplex is the remote transport                                                                      |
| Envelope's Python-qualified event type names                               | Stable registered names + `EventRegistry`                                                                                |

**Explicitly kept (unchanged, not reinvented):** framework-core `Agent`, `Actor` (with all its current observers / knowledge / assembly / compaction / aggregation / HITL / `run_subtask` / `run_subtasks` behavior), `KnowledgeStore` (extended with `append`/`read_range`/optional `watch`), `StateStore`, `AssemblyPolicy`, `AssemblerMiddleware`, `CompactStrategy`, `AggregateStrategy`, `Observer`, `Watch`, `Scheduler`, `Middleware`, `Stream`. The redesign is **network-only**. The network layer attaches an `Actor` to a hub via the paired `HubClient` / `ActorClient`; it does not subclass `Actor`.

---

## 13. Open Questions / Follow-ups

1. **Federation across hubs.** Envelopes carry `signatures`. Deferred until single-hub is solid, but the intended shape is:
   - Each hub publishes a **federation manifest**: a curated subset of its registry (selected actors), signed with the hub's key. Peer hubs fetch the manifest and mount the federated namespace under a prefix (e.g. local addressing `hub-a:ag2:researcher:1`).
   - Peering is an **explicit operator action**, not automatic discovery. You add a peer by URL + public key, not by wildcard broadcast.
   - Envelopes crossing a hub boundary accumulate a signature chain in `envelope.signatures[]`. The receiving hub verifies the chain against its trusted peer set.
   - Sessions may span hubs. Both hubs' rules must validate (most-restrictive wins, same policy as intra-hub §13.3). A cross-hub session has a primary hub (the initiator's) that owns the WAL; the other hub receives envelopes and routes them to its local participant.
   - Knowledge reads across hubs require explicit `rule.access.knowledge.federated_readers` entries.

2. **Auth.** *Resolved (§3.2).* Each `ActorIdentity` carries an `auth` block (`scheme` + `claim` + `key_fingerprint`). The hub-side `AuthAdapter` is a pluggable validator selected by `auth.scheme`; shipped implementations are `NoAuth`, `ApiKeyAuth`, `JwtAuth`, `MtlsAuth`, and `SignedChallengeAuth`. A hub may install multiple adapters and accept multiple schemes simultaneously — the matching adapter is selected per-identity at handshake. Default validator set is `NoAuth + ApiKeyAuth` so OSS hubs run locally without friction. Production deployments install additional adapters via `Hub(auth=[...])`. Auth runs at the WS `hello` frame and at the HTTP CRUD front door, in both cases before rule evaluation.

3. **Rule conflict resolution.** Most-restrictive wins; any explicit denial wins over any allow. Rate limits take the minimum of the intersecting sides. This is written into the rule evaluator and covered by tests.

4. **Idempotency on send.** Every `send` — both WS frames and HTTP `POST /v1/sessions/{id}/send` — accepts an `idempotency_key` (client-provided UUID). The hub stores `(session_id, idempotency_key) → envelope_id` in the state store for a configurable window (default 10 min). Repeat sends return the same envelope without re-running transforms.

5. **Human-in-the-loop.** *Resolved.* **Humans are first-class actors.** A human has an `ActorIdentity` (with `runtime_kind: "human"`, capabilities like `approval`, `decision`, `annotation`, and an `auth` block describing how the operator signs in) plus a `rule`. Its `runtime.json` points at a UI client (CLI, web, textual app). The `HumanClient` is a specialization of `ActorClient` whose notify handler routes envelopes to a pluggable "UI surface" instead of an `Actor.ask`. Any session that needs human input simply lists the human as a participant; the delivery, WAL, and lifecycle are the same as for any other actor. No HITL primitive, no special session type — HITL is just a different client class wired to a human-facing surface. `HumanClient` + `HumanCliClient` ship in **Phase 3** (alongside `WsLink`) so the "HITL is just an actor" claim is stress-tested by a real non-LLM participant before Phase 6 starts taking dependencies on it; Phase 6 adds `HumanTextualClient` and `HumanWebClient` surfaces on top of the same `HumanClient`.

6. **Audit format.** `hub/admin/audit.jsonl` schema needs a companion doc. Minimum shape: `{ts, actor_id, action, resource_type, resource_id, decision, reason, trace_id}`.

7. **Backpressure.** When an inbox is full, the hub **rejects at send** by default — the initiator's `send` returns an `InboxFull` error synchronously. Rule may override to `buffer` (bounded, TTL-bound) or `overflow_to_disk` (spool to `hub/actors/{id}/inbox/overflow/`). No silent dropping.

---


## 14. Implementation Phases

V3 ships in discrete phases. **Governing principle:** every phase establishes load-bearing contracts; later phases are strictly additive and never rewrite earlier work. Phase 0 wipes the V2 network layer — AG2 Beta carries no backward-compatibility obligations, so we start V3 on a clean slate instead of coexisting with a dead layer. Every phase ships with tests proportional to the surface it adds.

**Forbidden shortcuts at every phase** (listed so the principle has teeth):

- Hub calling `Actor.ask()` directly (bakes in the wrong ownership — notify inversion must hold from day one).
- Skipping `ActorClient` for in-proc fixtures and wiring `Actor` straight to `Hub` (retrofitting `transforms` later becomes a rewrite).
- Holding a live `agents: dict[str, Agent]` on the hub instead of a FS-backed registry on `KnowledgeStore`.
- Inlining `ActorIdentity` or `Rule` on the `Actor` class (couples framework core to network).
- Returning a session handle before the recipient has acknowledged the invite (changes ordering guarantees users will rely on).

### Phase 0 — Cleanup ✅ **Done**

Goal: wipe the V2 network layer in one pass.

Action items:

- ✅ Deleted `autogen/beta/network/` in full (hub, topology, plugins, channels, primitives, policies, logs, events, remote, convenience, desktop_proxy — 26 files, ~4,400 LoC).
- ✅ Deleted V2-specific network tests under `test/beta/network/` (topology / plugins / remote / convenience / priority integration / hub / hub topics / telemetry / metadata propagation / scheduler_hub / background delegation / review coverage / event serialization / primitives/channel / primitives/envelope / primitives/infra / primitives/priority / channels/http / …).
- ✅ Recovered framework-core tests that had been misplaced under `test/beta/network/` (knowledge, watch, advanced_watches, aggregate, compact, signal, observer, observers, scheduler, scheduler_watch_coverage, assembly, actor_integration, plus extracted framework-core nuggets from the former grab-bag `test_bugfixes.py` and `test_edge_cases.py`) and moved them to `test/beta/framework/`.
- ✅ Removed V2 network re-exports from `autogen.beta` public API; `autogen/beta/knowledge.py`'s dangling `EventRegistry` type-check import dropped.
- ✅ Confirmed: framework-core test suite stays green with `network/` removed (682 pre-existing + 171 recovered = 853 framework tests passing).

Note: the legacy playground demos (02–13) still import the V2 network surface and are currently broken. They are retained as-is and will be rewritten against the V3 API when Phase 6's LLM-tool surface lands.

### Phase 1 — Foundation ✅ **Done**

Goal: minimum end-to-end architecture with every load-bearing contract in place. One process, in-proc hub, N actors, `consulting` + `conversation` + `notification` session types, disk-durable WAL.

Action items:

- ✅ Extended `KnowledgeStore` with required `append(path, content) → int` and `read_range(path, start, end) → str`; added optional `on_change(path, callback) → ChangeSubscription` returning a `NoopChangeSubscription` by default. Implemented on `MemoryKnowledgeStore` (active subscribers + change notification) and `DiskKnowledgeStore` (no-op — real inotify/FSEvents ships with Phase 3); `LockedKnowledgeStore` proxies all three new methods. The name is deliberately distinct from `autogen.beta.watch.Watch` (scheduler-side triggers) to avoid the "two unrelated things both named watch" confusion.
- ✅ Hub-stamped UUID7 id helper in `autogen/beta/network/ids.py` with strict monotonicity (intra-millisecond counter + ms rollover).
- ✅ Primitives: `ActorIdentity` (profile + capabilities + auth blocks + `skill_md` sidecar, immutable post-registration, with full `to_dict`/`from_dict` round-trip including `skill_md`), `Rule` (access / limits / transforms dataclasses — Phase 1 enforces only `access.{inbound_from, outbound_to, session_types}` and `limits.max_concurrent_sessions` / `limits.session_ttl_default` applied at session creation; transforms stored verbatim; `parse_duration` helper on `LimitsBlock` turns `"2h"` / `"15m"` / `"30s"` into seconds), `Envelope` with stable registered event names (`ag2.msg.text`, `ag2.session.invite`, …) and a tagged `Priority` literal (`background | normal | urgent`) validated on construction, `SessionMetadata` + `Participant` + `SessionState`, `SessionType` enum (all six declared — only three adapted in Phase 1).
- ✅ `AuthAdapter` protocol + `NoAuth` implementation + `AuthRegistry`. The plugin point is wired through `Hub.register` / the Link `hello` frame, so Phase 3 auth adapters are purely additive.
- ✅ `Link` protocol declaring the full frame vocabulary (`hello` / `welcome` / `notify` / `receipt` / `send` / `accept` / `error` / `chunk` / `subscribe` / `unsubscribe` / `event` / `rule_changed` / `ping` / `pong`) with JSON-round-trippable encoders in `transport/frames.py`; `LocalLink` in-process duplex that runs the hub's connection handler as a background task per client. Both `LinkClient` and `LinkEndpoint` protocols declare the full surface (including `frames()` on the server side), so a Phase 3 `WsLink` endpoint has a complete contract to conform to.
- ✅ `AcceptFrame` carries a `wal_offset` cursor — the session WAL byte position immediately after the accepted envelope. `Session.ask` uses this as `since` when opening its correlated subscription, so the hub does **not** replay the whole WAL on every `ask` (what would otherwise be an O(N) cost per call).
- ✅ `Hub` (new module, FS-backed on `KnowledgeStore` at `hub/core.py`): registry, rule storage, session handshake, WAL append, inbox dispatch. Enforces `access`, `limits.max_concurrent_sessions`, and stamps `expires_at = created_at + limits.session_ttl_default` on every session. **Never** calls `Actor.ask` — it only delivers envelopes through the Link. Session creation waits for invite acks before returning an ACTIVE metadata.
- ✅ `SessionAdapter` protocol + `ConsultingAdapter` (strict 1Q1R), `ConversationAdapter` (bidirectional, explicit close), `NotificationAdapter` (fire-and-ack one-shot). User envelopes go through the adapter; `ag2.session.*` system envelopes bypass it.
- ✅ `HubClient` (owns `Link` + a direct `Hub` reference for the Phase 1 in-process control plane, produces `ActorClient`s via `register(actor, identity, rule)`; exposes `find` / `describe`; has separate `close` (disconnect all) vs `shutdown` (unregister all) paths).
- ✅ `ActorClient` (inbox loop in a background task, empty transforms pipeline as the Phase 5 seam, handler registry per session type, default per-type handlers that call `Actor.ask` and post correlated replies, bounded `_recent_sends` ring to suppress the default handler on replies to our own sends so `session.ask` never pingpongs against itself). Clean internal split: `inbox/`, `transforms/`, `handlers/`, `session/`.
- ✅ `Session` client handle (`send`, `ask` via `causation_id` correlation **combined** with a WAL `since` cursor for race-free and O(1) replay, `subscribe(since=None)` async iterator, `close`).
- 🟡 `SessionInboxPolicy` — deferred to Phase 2 as a trivial optimization. Phase 1 relies on the framework-core `Actor`'s own history to preserve multi-turn context in conversation sessions; the WAL stays authoritative on the hub side. Adding an explicit `SessionInboxPolicy` in Phase 2 is purely additive and does not change any Phase 1 contract.

Test coverage: 191 tests under `test/beta/network/` covering every primitive (ids, envelope, identity with `skill_md` round-trip, rule with `parse_duration`, session metadata, auth), `KnowledgeStore.append` / `read_range` / `on_change` on every backend (Memory / Disk / Locked), full `Link` frame round-trips (including `AcceptFrame.wal_offset`), `LocalLink` bidirectional delivery and close semantics, per-adapter delivery rules (consulting / conversation / notification), `Hub` registry CRUD, rule access allow/deny on both creation and send, `max_concurrent_sessions` enforcement, session TTL stamping from `limits.session_ttl_default`, session handshake happy-path + invite timeout, consulting 1Q1R close-after-reply, conversation multi-turn, notification one-shot + ack, `HubClient` / `ActorClient` end-to-end, custom handler override via `client.on(session_type)`, `Session.ask` causation-id correlation with `wal_offset` cursor, `Session.subscribe` WAL replay, three-actor crosstalk, disk durability (write → drop hub → reopen store → reconstruct WAL state), `read_range`-based WAL tailer, envelope priority validation across every valid / invalid value, `LinkClient` and `LinkEndpoint` protocol conformance, and an end-to-end run of a real framework-core `Actor` backed by `TestConfig` as a canned LLM.

Proof point: **1,050 tests pass** across the full `test/beta/` suite (682 pre-existing framework-core + 177 recovered/added framework-core + 191 V3 network), zero failures.

### Phase 2 — Multi-participant and Fan-out ✅ **Done**

Goal: all six session types and fan-out semantics. Introduces streaming, broader limit enforcement, and the adapter-extensibility surface so any later phase (or any operator) can ship new session types without rewriting the hub.

**Foundation debts from Phase 1 folded in along the way** (all items the Phase 1 audit flagged as 🟡 or follow-ups): `SessionInboxPolicy`, `Hub.peek_session` accessor, collapsed `_append_user/_append_system_envelope` helpers, and `invite_ack_timeout_s` moved from a per-call kwarg to `HubConfig`. One additional bug surfaced during refactor and was fixed with the type-name migration: `str(SessionType.CONSULTING)` returns `"SessionType.CONSULTING"` (not `"consulting"`) in Python 3.11+, so the adapter registry + `ActorClient._type_handlers` both now key off `.value` via a shared `_type_name` helper.

Action items:

- ✅ **Adapter extensibility per §5.5.** `Hub._adapters` is now `dict[str, SessionAdapter]`. `SessionAdapter.session_type` is a plain `str` (built-in adapters use `SessionType.X` enum members, which the hub normalizes through `_type_name(...)` at registration — defense against the Python 3.11+ `str(Enum)` footgun). `SessionMetadata.type` is a plain `str` on the wire, on disk, and in memory — unknown types round-trip through `from_dict` so custom adapters stay readable after re-registration drift. `Hub.register_adapter(adapter)` replaces existing names with a log warning; re-registering the same object is a no-op. Unknown types at `create_session` raise `SessionTypeError("no adapter registered for 'X'")`. The three Phase 2 adapters land through this API path, not through a constructor special case — same code operators will use.
- ✅ **Multi-participant handshake with quorum.** `_PendingInvite` now tracks per-actor `pending_ids` / `acked_ids` / `rejected_ids` sets with a `required` target. `create_session` accepts `required_acks: int | None` (default = all) and `participant_role` (defaults: `RESPONDENT` for 2-party, `PARTICIPANT` for multi-party). Each `Participant` is stamped with a numeric `order` index (0 for the initiator, then insertion order) so static / round-robin discussions and auction bid order stay deterministic. On partial reject, the hub computes `len(acked) + len(pending)` — if that can no longer reach the quorum, the handshake fails with `InviteRejectedError`.
- ✅ `BroadcastAdapter` (1→N, initiator-only sends, no auto-close), `DiscussionAdapter` (dynamic / static / round-robin orderings — static mode is the pipeline replacement, with `on_failure="abort" | "continue"` validated at create time; turn advancement is re-derived from the WAL on every send so the adapter stays stateless for hydrate + replay), `AuctionAdapter` (RFP → per-bidder one-shot bids → initiator-posted `ag2.auction.select` → 1:1 continuation between initiator and winner; all phase transitions derived from the WAL).
- ✅ **Broadcast fan-out** via new `Hub._fanout_to_participants` + `_deliver_to`. An envelope with `recipient_id=None` is cloned once per non-sender participant and delivered through each recipient's inbox; each clone has its own `recipient_id` stamped so the receiver's dispatcher routes it normally.
- ✅ **Subscription-replay race fixed via `Hub._wal_lock`.** The Phase 1 audit flagged this as the highest-priority foundation fix before Phase 2 features exercise `Session.subscribe`. Two symmetric critical sections — `(WAL append + subs snapshot)` in `post_envelope`/`_handle_system_envelope`/`_post_adapter_follow_up`, and `(WAL read + sub register)` in `_handle_subscribe` — serialize with each other so every envelope is delivered to every matching subscription **exactly once**, either via replay (sub registered after the append) or via fan-out against a locked snapshot (sub registered before the append). The invariant is covered by a dedicated regression test blasting 20 concurrent sends at a live subscriber.
- ✅ **Cold-restart rebuild path.** `Hub.hydrate()` walks `hub/actors/*/identity.json`, `hub/actors/*/rule.json`, and `hub/sessions/*/metadata.json`, rebuilds every in-memory cache (`_identities`, `_rules`, `_name_to_id`, `_sessions`, `_active_sessions`), and reconciles **half-written PENDING sessions** by transitioning them to `EXPIRED` with `close_reason="hydrate_orphaned_pending"` — an ack arriving after restart has no live `_PendingInvite.future` to resolve against, so those sessions are unrecoverable and must not leak participant slots. Because `hydrate` is async, it ships as an `async classmethod Hub.open(store, ...)` rather than a sync constructor kwarg — call sites that want fresh stores keep `Hub(store)` unchanged.
- ✅ **Non-participant subscriptions.** `rule.access.subscribe` is a new `SubscribeAccess` block carrying `sessions: "member-only" | "public-within-hub" | "public"` + `tasks` (Phase 4). `_handle_subscribe` delegates to `Hub._can_observe_session(metadata, observer_id)`, which implements the §13.3 most-restrictive-wins rule: participants always qualify; non-participants need **both** their own rule AND every participant's rule to allow public observation. One `member-only` vote from any participant vetoes the subscription.
- ✅ **`SessionInboxPolicy` + `PreviousOnlyInboxPolicy`.** New module `autogen/beta/network/policies/`. `SessionInboxPolicy` reads the session WAL via `Hub.read_wal`, converts each text envelope into a `ModelRequest` (from someone else) or `ModelMessage` (from self), and prepends the translated events to the current turn's model events — so a framework-core `Actor` without its own history still sees the full session transcript. `PreviousOnlyInboxPolicy` is the V2 pipeline-stage replacement: it injects only the most recent cross-actor text envelope, giving static discussions the "each stage sees only the previous stage's output" semantic. Both discover the ambient session via two injection points wired by the `ActorClient` notify handlers: `context.variables[SESSION_ID_VAR]` and `context.dependencies[HUB_DEP]`. Standalone actors (no hub) get pass-through behavior. The handler wiring uses `try/except TypeError` fallback so user-supplied test doubles with a 1-arg `ask` signature continue to work.
- ✅ **Streaming `chunk` frames end-to-end.** `ChunkFrame` now carries `session_id` / `sender_id` / `recipient_id` so the hub can route without threading extra state. `Hub._handle_chunk` validates (session exists, sender is a participant, session state allows writes), stamps `sender_id` from the authenticated endpoint to prevent spoofing, and dispatches via `_deliver_chunk` (unicast or broadcast fan-out). `ActorClient` maintains a per-envelope `_chunks: dict[str, Queue[ChunkFrame]]` — chunks landing before the recipient opens its iterator are buffered lazily rather than dropped. `Session.send_chunk(envelope_id, ...)` + `Session.iter_chunks(envelope_id)` are the public API; a `_Unset` sentinel distinguishes "default to the other participant" from "explicit broadcast" for `recipient_id=None`. Phase 2 keeps chunks as transient frame-level relays — not persisted to WAL — which is consistent with how framework core treats `__transient__` events.
- ✅ **Extended limits.** New `RateBlock(per_minute, burst)` on `LimitsBlock`; hub's `_rate_limiter` is a per-actor token bucket (`hub/_limits.py`) checked in `post_envelope` after access but before the adapter. `per_minute=0` disables. `delegation_depth` is enforced by carrying a `depth: int = 0` field on every envelope; `ActorClient._post_text_reply` auto-increments `reply.depth = original.depth + 1` so reply chains naturally count hops. `delegation_depth=0` on the sender's rule disables the ceiling. `tokens_per_hour` and `cost_per_day_usd` are **stored verbatim** on `LimitsBlock` for forward compatibility with Phase 4 (tasks), where LLM usage attribution has a natural owner — Phase 2 does not enforce either counter.
- ✅ **`idempotency_key` on send.** `SendFrame.idempotency_key` and `Envelope.idempotency_key` are both honored. The hub keeps `(session_id, idempotency_key) → _IdempotencyEntry(envelope_id, wal_offset, expires_at)` in an in-memory dict scoped per session, default TTL = 600s (tunable via `HubConfig.idempotency_ttl_s`). Lookup happens at `_handle_send` before `post_envelope` is even called, so a retry completely bypasses adapters, access checks, and rate limits — including the "retry storm doesn't burn the rate bucket" property that's otherwise a silent footgun. Lazy GC at write-time caps the dict at ~1024 entries. Same keys across different sessions are independent.
- ✅ **`EventRegistry`** (`autogen/beta/network/events.py`). Stable-name registry with pre-registered Phase 1/2 built-ins (`ag2.msg.text`, `ag2.session.*`, `ag2.error`, `ag2.auction.select`). Operators register custom names via `hub.register_event_type(spec_or_string)` or pass a pre-built registry to `Hub(event_registry=...)`. Permissive by default (unknown names pass — forward compatibility for custom actors); `strict=True` rejects unknown names at `post_envelope` time with `SessionTypeError`. Phase 3 WsLink/HTTP wire-format validation can promote this to a hard enforcement.
- ✅ **Envelope schema additions**: `depth: int = 0` (for delegation-depth enforcement) and `idempotency_key: str | None` (the Phase 1 field is now actually used). Both round-trip through `to_dict`/`from_dict`.

Test coverage: **295 tests under `test/beta/network/`** (191 Phase 1 + 104 new Phase 2). New test modules:

- `test_adapter_registry.py` (12) — `register_adapter`, collision warning, plain-string + enum interchangeable at dispatch, unknown type raises `SessionTypeError`, custom `_TournamentAdapter` satisfies the protocol.
- `test_subscription_race.py` (3) — exactly-once under 20 concurrent sends against a live subscriber, `since` cursor replays only after cursor, regression `Session.ask` smoke.
- `test_broadcast_adapter.py` (12) — create-time shape, initiator-only sends, fan-out to every non-sender with `recipient_id` stamped per clone, default-all vs `required_acks=1` quorum, timeout when quorum not reached, no auto-close after first message.
- `test_discussion_adapter.py` (9) — dynamic chat-room mode, static auto-close after last speaker, static out-of-turn reject, round-robin cycles, rejecting sends from non-participants.
- `test_auction_adapter.py` (13) — RFP-first enforcement, single-bid-per-bidder, select shape validation (winner must be a participant, cannot be the initiator, cannot be a second select), post-select initiator↔winner limitation, full lifecycle happy path.
- `test_hub_hydrate.py` (8) — identities + rules + skill_md sidecar rebuild, active/closed slot accounting, half-written PENDING transition to EXPIRED, idempotent second hydrate, partial corruption (missing rule.json) skipped with warning, post-hydrate `post_envelope` works, empty store no-op, `Hub.open` classmethod.
- `test_non_participant_subscriptions.py` (9) — `SubscribeAccess` round-trip + unknown-policy rejection, member-only default denies observer, hub-public allows observer, observer's own rule can still deny, most-restrictive-participant vetoes public observation.
- `test_inbox_policies.py` (8) — direction-aware translation (ModelRequest vs ModelMessage), `include_own=False` filter, standalone-actor pass-through (no session id / no hub), system envelopes skipped in translation, `PreviousOnlyInboxPolicy` injects last foreign envelope, own-only session → pass-through.
- `test_streaming.py` (6) — `ChunkFrame` round-trip with new fields, ordered delivery, broadcast fan-out to every non-sender, non-participant spoofing rejected, chunks landing before the iterator is opened are still buffered.
- `test_limits.py` (10) — token bucket unit math, refill over simulated time, rate rebuild on rule change, rule round-trip carrying `rate`/`tokens_per_hour`/`cost_per_day_usd`, envelope `depth` round-trip, hub rate-limit rejects excess, per-actor isolation, `delegation_depth` enforcement + `=0` disables.
- `test_idempotency.py` (4) — repeat send returns cached envelope without re-appending to WAL, same-key-different-sessions independence, TTL expiry with lazy GC, retry does not consume rate bucket.
- `test_event_registry.py` (13) — built-ins pre-registered, string + spec shorthand, `allowed_in` metadata preserved, unregister, strict mode rejection with `UnknownEventTypeError`, hub integration permissive vs strict, `hub.event_registry` property.

Proof point: **1,154 tests pass** across the full `test/beta/` suite (853 framework-core + 295 V3 network + 6 other recovered), zero failures, zero regressions against Phase 1.

Small lingering items carried forward to Phase 3a (noted for transparency, none are blockers):

- `autogen/beta/network/transport/frames.py` has two cosmetic linter warnings (`typing.Union` unused, unreachable branch in `_frame_to_dict`) that pre-date Phase 2; cleanup is a small follow-up pass.
- `tokens_per_hour` and `cost_per_day_usd` are stored on `LimitsBlock` but not enforced yet — Phase 4 tasks are where LLM usage attribution becomes natural.
- `ActorClient.disconnect` still awaits handler tasks without cancelling first (inherited from the Phase 1 debt list). Fine for in-proc / `LocalLink`; worth a pass before Phase 3a `WsLink` where network I/O can block indefinitely.
- `autogen/beta/scheduler.py` still carries three dead V2 references (`self._hub.stream`, `self._hub._delegate`, and a `from .network.events import SchedulerTriggerFired` import pointing at a V2 class that doesn't exist in V3). They are lazy imports inside a function body, so Phase 2 tests pass, but Phase 3a's TTL sweeper needs Scheduler's callback mode cleanly usable. Phase 3a Step 0 drops the hub-delegation branch entirely — a ~20 LoC deletion pass that keeps only the standalone callback path.
- Phase 2's hub delivers envelopes to a flat `hub/actors/{id}/inbox.jsonl` log and never GCs it; §7.1's structured `pending/` + `received/` layout with receipt-driven GC and `max_pending` / overflow enforcement lands in Phase 3a as the foundation for `WsLink` reconnect replay. Phase 2 was correct for a single-process, always-connected `LocalLink`; it breaks the moment a client can disconnect.
- `ActorClient` subscription cursors are not checkpointed across connection drops — fine under `LocalLink` (no drops) but Phase 3a's `WsLink` reconnect needs a cursor replay mechanism so a reconnecting actor does not miss envelopes that landed during the drop. Added to the Phase 3a action list explicitly.
- `runtime.json` is written once at registration with `binding: "local", reachable: false` and never updated after `hello`. Phase 3a's `_handle_hello` path rewrites it on every handshake so §3.4's dispatch-by-binding story actually holds for mixed local/ws deployments.

### Phase 3a — Foundation: Cross-process Transport and Real Reactivity ✅ **Done**

Goal: a WebSocket client on another host can register, authenticate, subscribe with resume, and exchange sessions with local actors. Hub state survives process restarts with active sessions intact. First non-LLM actor (`HumanClient`) stress-tests actor symmetry against something that is neither a Python `Actor` nor an LLM-backed handler.

Phase 3 is split into **3a (foundation)** and **3b (operations)** because the transport / inbox / reconnect / runtime work is the critical path — everything an operator needs to run V3 across processes on one host, or across hosts — while the store backends, full admin HTTP surface, and archival are additive on top of that foundation and can slip without blocking 3a's validation story. After Phase 3a ships, a real WebSocket client on another host can join a hub, authenticate, reconnect through drops without losing envelopes, and include human participants in sessions. After Phase 3b ships, that same hub is deployable against multiple store backends with full audit / admin tooling.

Action items:

- ✅ **Step 0 — Scheduler cleanup (Phase 0 leftover).** Removed the hub-delegation branch from `autogen/beta/scheduler.py`. `Scheduler.__init__` no longer takes a `hub` kwarg; `_handle_fire` lost the hub-mode path; the dead `SchedulerTriggerFired` import and `self._hub.stream` / `self._hub._delegate` call sites are gone. `_WatchEntry` shed its `target` / `task` / `task_factory` / `priority` fields — the scheduler is now a pure lifecycle manager that fires user-supplied callbacks, matching the spirit/letter of "it is a watch lifecycle manager, not a rigid scheduling engine" from the module docstring. ~20 LoC deleted net. Framework-core scheduler tests stayed green without modification (every existing test used standalone callback mode already).
- ✅ **Real `on_change()` on `DiskKnowledgeStore`.** Added a `_DiskChangeHandler` + `_DiskChangeSubscription` pair driven by `watchdog` (inotify on Linux / FSEvents on macOS / `ReadDirectoryChangesW` on Windows, with `PollingObserver` fallback if the native backend fails to initialize). Lazy imports `watchdog` so the V3 network package stays install-optional — callers without `watchdog` get `NoopChangeSubscription` and the hub transparently polls. The handler bridges the background watchdog thread to the asyncio loop via `run_coroutine_threadsafe`. `close()` runs stop+join in an executor so teardown does not pin the event loop. `on_change` auto-mkdirs the target path so "subscribe first, then write" works the same way it does on `MemoryKnowledgeStore`. `DiskKnowledgeStore.on_change` virtual-path-filters events so a subscription on `/wal` does not leak events from `/rules` when the observer is rooted higher up. `watchdog>=4.0,<7` is a new optional dep.
- 🟡 **Hub subscription delivery driven by `on_change`** — deferred to Phase 3b. Phase 2's in-memory `_subscriptions` dict is authoritative for single-process deployments (the fast path — sub-millisecond fan-out); the `on_change`-driven cross-process observer path is additive and pairs naturally with Phase 3b's `SqliteKnowledgeStore` / `RedisKnowledgeStore` work where the shared-store story has a real use case. Not a blocker for anything else in 3a; marked as an open item on the 3b scope.
- ✅ **Structured inbox layout** (§7.1 compliance). Hub writes to `hub/actors/{id}/inbox/pending/{envelope_id}.json` on `_deliver_to`. `ReceiptFrame(status="ack")` moves the file to `hub/actors/{id}/inbox/received/{envelope_id}.json`; `ReceiptFrame(status="nack", reason=...)` appends a structured entry to `hub/actors/{id}/inbox/nacks.jsonl` and removes the pending file. `rule.limits.inbox.max_pending` / `overflow` enforced at **pre-flight** time (before the WAL append, so a rejected broadcast leaves no half-persisted state) — `reject` raises `InboxFullError` synchronously; `spool` writes to `hub/actors/{id}/inbox/overflow/{id}.json` without bumping the pending counter or pushing a notify frame. `drop_oldest` / `drop_newest` round-trip through the rule but fall through to reject semantics in 3a (land for real in 3b). The pending counter is a per-actor dict rebuilt on `Hub.hydrate()` by listing `pending/`. **System envelopes (`ag2.session.*`) bypass the structured inbox entirely** — they are ephemeral handshake signals and must not consume user workload budget; this keeps `max_pending` separate from session chatter. Phase 2's flat `inbox.jsonl` log is gone. New `InboxBlock` dataclass on `LimitsBlock` round-trips through rule JSON with every overflow mode.
- ✅ **`ActorClient` subscription cursor checkpoint.** New `_ClientSubscription` dataclass replaces Phase 2's raw `asyncio.Queue`-per-sub dict, carrying `(subscription_id, queue, session_id, causation_id, since)`. `EventFrame` gains a `wal_offset` field stamped by the hub on every outbound event (both the live fan-out path and the initial replay via a new `_read_wal_with_offsets` helper). The client advances `since` with every delivery; on `ActorClient.reconnect()` each live subscription rotates to a fresh `subscription_id` and re-subscribes with its saved `since` so the hub replays only envelopes that landed during the drop. Queue identity is preserved across the rotation so callers holding a reference (`Session.ask`, `Session.subscribe`) do not need to re-open anything. **New `AcceptFrame(envelope_id="", request_id=<sub_id>)` subscribe-ack path** closes the ordering race where `_open_subscription` used to be fire-and-forget — `_open_subscription` and `reconnect` now both wait for the hub's "subscribe applied + replay drained" confirmation before returning, eliminating the "client thinks it's subscribed but the hub hasn't registered yet" window. `reconnect()` raises `LinkClosedError` on stopped clients.
- ✅ **`ActorClient.disconnect` handler-task cancel (Phase 2 debt pay-off).** Handler tasks are now cancelled before being awaited — Phase 2 left this as a lingering item ("fine for in-proc `LocalLink`; worth a pass before Phase 3a `WsLink`"). The fix unblocks the blocking-handler test pattern for inbox `max_pending` observability and prevents a hung disconnect when a real WebSocket handler blocks on I/O.
- ✅ **Runtime binding update on `hello`.** `_EndpointSide` gained `binding` (default `"local"`) + `ws_url` + `http_url` class attrs; `_WsEndpointSide` overrides `binding = "ws"`. New `Hub._write_runtime` helper rewrites `hub/actors/{actor_id}/runtime.json` on every hello with `{binding, target: endpoint_id, ws_url, http_url, reachable: true, last_heartbeat}`. `connection_handler`'s finally block flips `reachable` to `false` on disconnect — **but only if this endpoint is still the current one** (a reconnect that already re-stamped runtime with a new endpoint is not clobbered by the old connection's cleanup). This closes the §3.4 dispatch-by-binding loop for mixed local/ws deployments.
- ✅ **`WsLink` over WebSocket.** New `autogen/beta/network/transport/ws.py` module with `WsLinkClient` (actor-side factory pointed at a hub URL), `_WsClientSide` (lazy-connecting client handle), `WsLinkServer` (hub-side listener wrapping `websockets.serve` on `host:port`, binds `port=0` to get a random free port and surfaces it via `.url`), and `_WsEndpointSide` (server-side peer). Frames use the existing `encode_frame` / `decode_frame` JSON-line helpers — no new wire format. `websockets>=14.0,<17` is a new optional dep. Lazy import so the whole network layer stays install-optional; missing `websockets` raises `TransportError` with a clear install hint. End-to-end consulting, conversation, chunk streaming, subscription delivery, disconnect cleanup, and reconnect replay all pass over `WsLink` using the exact same client code that works on `LocalLink`.
- ✅ **Minimum HTTP surface — `Hub.serve()` 3a slice.** New `autogen/beta/network/http/` subpackage with `build_app(hub)` (returns a Starlette app mountable in any larger ASGI project) and `HttpServer(hub, host, port)` (uvicorn runner with lazy import). Seven endpoints:

  ```
  POST   /v1/actors                          register
  GET    /v1/actors                          discover (?capability=)
  GET    /v1/actors/{id}                     describe
  POST   /v1/sessions                        create (handshake initiator half)
  GET    /v1/sessions/{id}                   describe
  POST   /v1/sessions/{id}/close             explicit close
  GET    /v1/sessions/{id}/wal?since=        bulk WAL read
  ```

  A single `_ERROR_STATUS` table maps every `NetworkError` subclass to the right HTTP code (`AuthError → 401`, `AccessDeniedError/RuleViolationError → 403`, `UnknownActorError/UnknownSessionError → 404`, `DuplicateRegistrationError/InviteRejectedError → 409`, `SessionClosedError → 410`, `LimitExceededError/InboxFullError → 429`, `SessionTypeError → 400`); every route shares a `{error, message}` body shape. `starlette` and `uvicorn` are lazy imports. The 10 remaining endpoints from §9.1 (activity, force-close, admin, metrics, knowledge-read-through) slip to Phase 3b where they pair with the archival + audit work.
- ✅ **`ApiKeyAuth` + `JwtAuth` adapters.** Added to `autogen/beta/network/auth.py`. `ApiKeyAuth` takes a per-hub fingerprint allowlist and uses `hmac.compare_digest` for timing-safe comparison; `add_fingerprint` / `revoke_fingerprint` lifecycle hooks for operator tooling; an empty allowlist means "any correctly-matching fingerprint is accepted" which makes tests painless without diluting the production semantics. `JwtAuth` is PyJWT-backed, takes `key` + `algorithms` + optional `required_issuer` / `required_audience` / `leeway`, enforces `sub == identity.name` to prevent cross-actor token replay, and lazy-imports PyJWT. New `dev_registry(api_key_allowlist, jwt_key, jwt_algorithms)` convenience factory installs the full Phase 3a adapter zoo in one call. Both adapters run at both entry points (WS `hello`, HTTP front door) before rule evaluation.
- ✅ **TTL sweeper via Scheduler callback mode.** `Hub.sweep_expired_sessions()` walks `_sessions`, transitions every ACTIVE entry whose `expires_at` has passed to `EXPIRED`, releases participant slots, cascades task cancellation (Phase 4), and broadcasts `SessionClosed` with `reason="ttl_expired"`. `Hub.ttl_sweep_callback()` returns an `(events, ctx) → None` async callable directly usable via `Scheduler.add(IntervalWatch(30), callback=...)`. **Deliberately not bundled** — the hub does not create a scheduler of its own; the operator owns both instances and wires them explicitly, keeping the hub network-only and letting Phase 4 task TTL reuse the same sweeper without restructuring.
- ✅ **`HumanClient` + `HumanCliSurface` + `HumanScriptedSurface`.** New `autogen/beta/network/client/human.py` module. `HumanSurface` is a Protocol with `on_envelope(envelope, client) → str | None` (None means "observe silently") and `on_close(client)`. `HumanClient` is an `ActorClient` subclass that wires a single surface into every built-in session type via `_type_handlers`; users can still override per-type with `client.on("discussion")` without disturbing the default surface routing. `HumanCliSurface` reads from stdin via `run_in_executor` so the asyncio loop stays responsive, handles `EOFError`/`KeyboardInterrupt` gracefully, ignores non-text envelopes silently, and is fully configurable (`input_fn`, `output_fn`, `prompt_format`) for tests. `HumanScriptedSurface` yields a pre-declared list of responses and records every envelope seen so tests can assert on what the operator saw. `human_cli_client(...)` factory matches the design doc's `HumanCliClient = HumanClient + HumanCliSurface` shape. New `HubClient.register_human(surface, identity, rule)` method mirrors `register()` but produces a `HumanClient` and stamps `runtime_kind="human"` on the identity (unless explicitly overridden to `"browser"` / `"external"` etc). The hub never branches on `runtime_kind` — it's informational metadata that discovery / describe endpoints surface unchanged. End-to-end consulting and conversation sessions work on both `LocalLink` and `WsLink` — the first non-LLM validation of the "HITL is just an actor" claim.

Test coverage: **206 new tests under `test/beta/network/`** across nine new modules plus updates to existing suites:

- `test_knowledge_store_extensions.py` (+10) — new `TestDiskOnChange` class: fires on append, fires on write, multiple files under same prefix, nested subdirectory events, close stops delivery, close is idempotent, events outside prefix are filtered, two independent subscriptions, virtual path is store-relative not physical, `LockedKnowledgeStore` proxies `on_change`.
- `test_inbox_structured.py` (20) — `TestInboxBlock` round-trip matrix (defaults / reject / spool / drop_* / unknown policy / rule JSON), `TestStructuredInboxDelivery` (deliver lands in pending/, ack moves pending→received, counter tracks delivery through blocking handlers, counter decrements on consulting round-trip), `TestReceiptHandling` (double-ack idempotent, nack writes log + clears pending, ack for unknown envelope is no-op), `TestMaxPendingReject` (reject raises before WAL append, reject consumes rate bucket — ordering is documented, reject frees slot after ack), `TestMaxPendingSpool` (spool writes to overflow without bumping counter, spool skips NotifyFrame push), `TestFanoutPreflight` (one full recipient rejects whole broadcast atomically), `TestInboxHydrate` (cold restart rebuilds pending counter).
- `test_reconnect_resume.py` (11) — `TestEventFrameWireFormat` (wal_offset round-trips explicit + default), `TestCursorTracking` (since advances on every delivery via both live fan-out and initial replay), `TestReconnectResume` (reconnect replays sub with saved cursor, envelopes sent during a drop replayed exactly once, queue identity preserved, multiple subscriptions rotated independently, reconnect on stopped client raises, reconnect without any subs works), `TestAskSurvivesReconnect` (`Session.ask` before and after a reconnect).
- `test_runtime_binding.py` (10) — `TestRegistration` (runtime.json exists after register with placeholder shape), `TestHelloStampsRuntime` (marks runtime reachable with binding, stamps endpoint_id, persists to disk), `TestDisconnectMarksUnreachable` (flips reachable=false preserving last-known fields, reconnect repaints reachable=true without the old cleanup clobbering it), `TestMultiActorRuntime` (two actors have independent runtime entries, disconnecting one leaves the other alive), `TestRuntimeFreshnessAcrossSession` (last_heartbeat advances on reconnect via mock clock, session open does not clobber runtime).
- `test_wslink.py` (14) — `TestAvailability` (server requires handler, URL exposes bound port), `TestRawFrameRoundTrip` (SendFrame ↔ AcceptFrame over a bare `websockets.serve` echo handler, EventFrame with wal_offset=99), `TestEndToEnd` (register over WsLink stamps runtime.json with binding="ws" and server URL, consulting round-trip, conversation multi-send, subscription delivery advances cursor), `TestWsDisconnectAndReconnect` (disconnect flips reachable preserving binding="ws", reconnect replays envelopes around a simulated drop), `TestClientLifecycle` (closed client rejects new .client(), client-side close is idempotent, connect failure raises cleanly), `TestChunkStreamingOverWs` (end-to-end streaming over a real WebSocket).
- `test_http_server.py` (23) — every route's happy + at least one failure path: `TestRegisterActor` (×5), `TestFindActors` (×3), `TestDescribeActor` (×3), `TestCreateSession` (×4), `TestSessionLifecycle` (×6) + `TestRealUvicornIntegration` (×1 end-to-end over a real uvicorn instance on a random port, via `httpx.AsyncClient`). 22 of the 23 tests run in-process via `httpx.ASGITransport` for sub-millisecond dispatch.
- `test_auth_adapters.py` (24) — `TestApiKeyAuth` (8): scheme mismatch, missing key_fingerprint, missing claim fingerprint, fingerprint mismatch, match without allowlist, match with allowlist, match not in allowlist, add/revoke lifecycle. `TestJwtAuth` (9): scheme mismatch, missing token, valid HS256, wrong key, expired, wrong audience, wrong issuer, subject mismatch (cross-actor replay protection), required_audience override. `TestRegistryIntegration` (7): dev_registry ships NoAuth + ApiKeyAuth, dev_registry + jwt_key ships JwtAuth, unknown scheme raises, hub registration accepts valid API key, rejects invalid API key, accepts valid JWT, rejects tampered JWT.
- `test_ttl_sweeper.py` (10) — `TestSweepExpiredSessions` (6): fresh session not expired, session with short TTL expires past deadline, expired releases participant slots, closed session untouched, sweeper idempotent, expired broadcast reaches subscribers. `TestMultiSessionSweep` (1): only expired sessions touched. `TestSchedulerIntegration` (2): `ttl_sweep_callback` signature is callable, end-to-end `Scheduler + IntervalWatch` fires the sweeper and transitions a session to EXPIRED. `TestExpiresAtStamping` (1): `expires_at = created_at + ttl` via a deterministic fake clock.
- `test_human_client.py` (19) — `TestScriptedSurface` (3): sequential responses, None when exhausted, skips non-text without consuming slots. `TestCliSurface` (4): calls input_fn with prompt, empty response returns None, ignores non-text events, handles EOFError. `TestRegisterHuman` (3): `runtime_kind="human"` stamped on disk, explicit override preserved, surface accessor live. `TestConsultingOverLocalLink` (2): human answers consulting, None reply causes session.ask timeout. `TestConsultingOverWsLink` (1): full consulting round-trip over real WebSocket with a scripted human participant. `TestConversationMultiTurn` (1): human replies to every turn. `TestBroadcastObserver` (1): human observes broadcast without replying. `TestDisconnectHook` (2): on_close fires on disconnect and unregister. `TestHandlerOverride` (1): `client.on("conversation")` overrides surface routing. `TestHumanCliClientFactory` (1): factory produces wired client.

Proof point: **1,360 tests pass** across the full `test/beta/` suite (Phase 2 baseline 1,154 + 206 Phase 3a tests added by this work, plus parallel Phase 4 task-layer tests that landed on the same branch), zero failures, zero regressions against Phase 2. The 505-test V3 network suite is up from Phase 2's 295 (+72% growth).

Small lingering items carried forward to Phase 3b (none are blockers for Phase 3a's validation story):

- Cross-process `on_change`-driven hub subscription delivery — deferred as noted above; pairs naturally with 3b's Sqlite/Redis backend work where multi-process hubs actually make sense.
- Phase 2's two cosmetic `frames.py` linter warnings (`typing.Union` unused, unreachable branch in `_frame_to_dict`) — pre-date 3a, still harmless, untouched. Easy Phase 3b cleanup.
- `tokens_per_hour` / `cost_per_day_usd` on `LimitsBlock` — still stored verbatim, still not enforced. Phase 4 attribution is still the natural home; unchanged from Phase 2.

### Phase 3b — Operations: Store Backends, Full Admin Surface, Archival ✅ **Done**

Goal: a complete multi-user OSS deployment story on top of 3a. Operators can pick another store backend, query audit and metrics endpoints, share read-only knowledge between actors, and rely on background archival keeping the hub's hot state bounded.

Phase 3b is strictly additive on top of 3a — no 3a contract is revisited. Scope is deliberately kept small for an OSS framework: the simplest thing that matches the §9.1 HTTP surface, works on a single machine, and stays approachable for contributors. Four follow-ups are explicitly **deferred** (see below), so the scope stays honest.

Action items:

- ✅ **`SqliteKnowledgeStore` + `RedisKnowledgeStore` backends** (`autogen/beta/knowledge.py`). Both conform to the full `KnowledgeStore` protocol (`read` / `write` / `list` / `delete` / `exists` / `append` / `read_range` / `on_change`) that Memory and Disk already satisfy, with no surface changes to the protocol itself. Implementation notes: Sqlite is one `entries(path, content, version)` table with an asyncio lock serializing writes through `loop.run_in_executor`; Redis uses a per-path string + a companion sorted set `{prefix}:__index` for O(1) version diffing. Both backends implement `on_change` via the same new `_PollingChangeWatcher` — a polling watcher keyed off a per-path monotonic version index, default 500 ms interval, configurable per-instance. **No Redis keyspace notifications** (see Deferred). Drivers are lazy-imported: `sqlite3` is stdlib so no new optional dep; `redis.asyncio` stays an optional install under `ag2[redis]` pinned `redis>=5.0,<7`. The `test_knowledge_store_extensions.py` protocol conformance suite is now parameterized across Memory / Disk / Locked / Sqlite, so every future backend automatically picks up the contract surface.
- ✅ **`KnowledgeAccess` block on `AccessBlock`** (`autogen/beta/network/rule.py`). New dataclass with `expose: list[str]` (glob patterns over the target actor's own `KnowledgeStore` paths) and `readers: list[str]` (glob patterns over requesting actor names) — structurally identical to `SubscribeAccess` and sitting next to it on `AccessBlock`. Defaults are empty (no exposure) so new actors get zero cross-actor reachability without explicit opt-in. Path matching is handled by a new `_match_path` helper that understands `**` (any depth including zero), `*` (single segment), and `?` (one non-`/` character) — strictly more powerful than `fnmatch.fnmatchcase`, which was a mismatch for filesystem-shaped patterns. Full `Rule` JSON round-trip covers the new block. Phase 5a's scope stays pure: named/python/http transforms only, no access-layer work.
- ✅ **Remaining HTTP surface (§9.1)** — six new routes plus three task 404 stubs, wired in `autogen/beta/network/http/server.py`:

  ```
  PUT    /v1/actors/{id}/rule                             replace an actor's rule
  GET    /v1/actors/{id}/activity                         recent sessions + tasks for an actor
  GET    /v1/actors/{id}/knowledge/{path:path}            KnowledgeAccess-gated cross-actor read
  GET    /v1/sessions?state=&participant=&type=&limit=    list sessions with filters
  POST   /v1/sessions/{id}/force-close                    admin-only force close
  GET    /v1/admin/health                                 hub liveness
  GET    /v1/admin/metrics                                counter dict (see below)
  GET    /v1/tasks                                        explicit 404 — Phase 6
  GET    /v1/tasks/{id}                                   explicit 404 — Phase 6
  POST   /v1/tasks/{id}/cancel                            explicit 404 — Phase 6
  ```

  The knowledge-read endpoint requires the caller to stamp `X-Ag2-Reader: <actor_name>` in the request headers — operators wire auth middleware in front of the app to project a verified JWT/mTLS claim into that header. Task stubs return a stable `{error: "NotImplemented", message: ...}` 404 body shape so Phase 6 clients can swap real implementations in without any wire changes. The `_ERROR_STATUS` table from 3a needed no new entries — every new route raises through existing `NetworkError` subclasses.
- ✅ **`/v1/admin/metrics` from in-memory hub state — not from the audit log.** `Hub.metrics()` returns a flat counter dict computed from live cache state (no audit-log scan per scrape):

  ```json
  {
    "actors":   {"registered": 12, "connected": 9},
    "sessions": {"active": 4, "pending": 0, "closed_total": 87},
    "tasks":    {"running": 2, "completed_total": 41, "failed_total": 3},
    "inbox":    {"pending_total": 3},
    "uptime_s": 4210
  }
  ```

  New monotonic counters land on the hub in `__init__`: `_started_at`, `_sessions_closed_total`, `_tasks_completed_total`, `_tasks_failed_total`. `close_session` and `sweep_expired_sessions` both bump `_sessions_closed_total`; `EV_TASK_RESULT` bumps `_tasks_completed_total`; `EV_TASK_ERROR` bumps `_tasks_failed_total`. Active session / active task / inbox totals are derived on demand from the existing in-memory maps. Uptime is computed from the injected `clock` (real or fake) via a small `_iso_seconds_since` helper. `GET /v1/admin/metrics` serializes the dict verbatim — operators who want Prometheus shape wire an external adapter.
- ✅ **Audit log writer** — `hub/admin/audit.jsonl`, append-only, one JSON line per mutation, schema `{ts, actor_id, action, resource_type, resource_id, decision, reason, trace_id}` (§13.6). The hub owns an `asyncio.Queue` populated by a non-blocking `_audit(...)` method and drained by a background `_audit_writer_loop` task started via `_start_audit_writer()` from `Hub.open`. Writes are strictly best-effort: failures are logged but never block a mutation, and tests that construct `Hub(store)` directly (no `open`) never start the writer so the `_audit` call sites short-circuit cleanly. `close()` drains the queue through a sentinel `None` entry and awaits the task with a 1s timeout. Every mutation path now audits: `register_actor`, `unregister_actor`, `update_rule` (via new `Hub.set_rule`), `create_session` (allow / timeout / rejected), `close_session`, `expire_session`, `force_close_session`, `subscribe_session` (allow / deny), `inbox_drop_newest`, `inbox_drop_oldest`, `archive_session`, `read_knowledge` (allow / deny / not-found). **Metrics do not derive from this file** — the two surfaces are intentionally independent so the audit file can be rotated externally (see Deferred) without affecting the liveness view.
- ✅ **Archival sweeper** — `Hub.archive_closed_sessions(*, age_threshold_s, now=None) -> list[str]`. Walks closed/expired sessions whose `closed_at` is at least `age_threshold_s` older than `now`, computes a compact summary (envelope count, first/last timestamps, distinct senders, participants, close reason), writes `hub/archive/sessions/{id}/summary.json` + `wal.jsonl`, deletes the live WAL at `hub/sessions/{id}/wal.jsonl`, and stamps the session metadata with a new `archived_at` field. New `SessionMetadata.archived_at: str | None` round-trips through `to_dict` / `from_dict`. Idempotent (skips already-archived), safe against active sessions (skipped by state check). `Hub.archive_sweep_callback(age_threshold_s=3600.0)` returns an async `(events, ctx) → None` callable wired through the same `Scheduler` the Phase 3a TTL sweeper uses — operators add both callbacks to one `Scheduler` instance and the hub stays network-only. **Deliberate simplification**: the archive sweeper does **not** reuse framework-core `CompactStrategy`. That protocol is oriented at mid-stream LLM compaction (`BaseEvent` + `Context` + per-turn token budget), not cold archival of past sessions. Our summary is plain-text and LLM-free, which matches the OSS framework ethos and lets operators subclass the hub if they want a richer summary.
- ✅ **Inbox `drop_oldest` / `drop_newest`** — finished the Phase 3a skeleton. `_preflight_inbox_capacity` now lets all three non-raising overflow modes through; `_deliver_to` branches on the mode and either evicts (`drop_oldest`), silently drops (`drop_newest`), or spools (`spool`). Eviction reads `pending/`, picks the oldest file by lexicographic order (UUID7 envelope ids are time-sorted), deletes it, decrements the pending counter, and writes an `inbox_drop_oldest` audit line. Drop is a no-op on the WAL (the envelope still gets recorded at post-envelope time) and on the recipient's inbox (no write, no NotifyFrame push, no counter bump), plus an `inbox_drop_newest` audit line. Both modes emit the audit entries through the same `_audit` helper the new writer consumes. `drop_oldest` survives hydrate via the existing pending-counter rebuild; `drop_newest` never touches the pending directory so there's nothing to rebuild.
- ✅ **`on_change`-driven identity/rule cache invalidation.** `Hub._start_cache_invalidation()` (called from `Hub.open`) subscribes to `hub/actors/*` via `store.on_change`. The callback inspects the changed path, dispatches to either `_reload_actor_identity` or `_reload_actor_rule`, and atomically refreshes the in-memory cache under `_cache_invalidation_lock`. Identity reloads also refresh the `_name_to_id` index (handling rename by dropping the old name pointer and adding the new one) and re-read the SKILL.md sidecar. Deletions drop cache entries to match. Subscribes succeed on every backend that supports `on_change` (Memory natively, Disk via watchdog, Sqlite/Redis via the new polling watcher); backends that return `NoopChangeSubscription` silently no-op. `close()` closes the subscription idempotently. `Hub.set_rule` also writes a `RuleChangedFrame` to the live endpoint so Phase 5a's transforms pipeline has a landing surface. **Scope intentionally limited to cache invalidation** — see Deferred for the cross-process subscription fan-out story.

Deferred (with rationale):

1. **Redis keyspace notifications.** Polling works for both new stores and keeps them structurally identical. Keyspace notifications require `notify-keyspace-events` in the Redis server config (many managed providers disable it by default), are fire-and-forget, and would add a second code path for a latency gain most OSS users will not notice. Revisit if a user asks for sub-500ms cross-process delivery.
2. **Cross-process subscription fan-out.** A multi-process hub deployment sharing a store is interesting but introduces multi-writer WAL coordination, on-change-driven fan-out loops, and failure-mode questions that do not belong in a "ship another store backend" phase. Full treatment lands under Phase 7 federation, where multi-hub coordination already has a home. 3b only wires `on_change` for local cache invalidation.
3. **Audit log rotation.** Unix has `logrotate`; an OSS framework should not reinvent it. The audit file is append-only, metrics do not depend on it, and operators who care can rotate externally. A docstring on the layout helper says so explicitly.
4. **`tokens_per_hour` / `cost_per_day_usd` enforcement.** Unchanged from Phase 2 — still stored verbatim on `LimitsBlock`, still not enforced. Natural home is Phase 6 when the LLM verb surface makes usage attribution concrete.

Test coverage: **166 new tests under `test/beta/network/`** (510 Phase 4 baseline → 676 total) across nine new / expanded modules:

- `test_rule.py` (+19) — new `TestKnowledgeAccessDefaults`, `TestKnowledgeAccessExposeMatching`, `TestKnowledgeAccessReaderMatching`, `TestKnowledgeAccessCombined`, `TestKnowledgeAccessRoundTrip` covering the `KnowledgeAccess` block. Plus `TestPathNormalization` and `TestPathGlobMatch` for the new `_match_path` / `_normalize_store_path` / `_path_glob_match` helpers: exact paths, directory prefixes, `*` vs `**` scoping, `?` single-char semantics, reader/path combined allow/deny logic.
- `test_inbox_drop_policies.py` (12) — `TestDropOldest` (at-capacity eviction, below-capacity no-op, sustained pressure cycle, new envelope delivered after eviction, disk store round-trip, hydrate-after-eviction counter preservation), `TestDropNewest` (silent drop at capacity, `send` does not raise, WAL still records the post, handler never sees dropped envelope, slot frees on ack), `TestDropPoliciesFanout` (per-recipient policy mixing).
- `test_audit_log.py` (16) — `TestAuditWriterDisabled` (sync constructor doesn't start writer, close is no-op), `TestRegistrationAudit` (register/unregister write), `TestSessionAudit` (create/close/TTL-expire write), `TestRuleAudit` (`set_rule` writes + updates cache + disk round-trip + unknown actor raises), `TestInboxDropAudit` (both drop modes emit structured entries), `TestAuditWriterMechanics` (schema round-trip, disk persistence, close drains queue, sort_keys stability).
- `test_hub_metrics.py` (13) — `TestMetricsShape` (empty hub shape, idempotent reads), `TestActorCounters` (register/unregister +/- 1, connected tracks live endpoints), `TestSessionCounters` (active/closed_total, monotonic closed_total, TTL-expired sessions bump closed_total), `TestTaskCounters` (running + completed_total on result, failed_total on crash), `TestInboxCounter` (pending_total sums per-actor), `TestUptime` (fake clock advances uptime, real clock works by default).
- `test_http_admin.py` (30) — `TestUpdateRule` (happy path replaces rule, 404 on unknown actor, 400 on malformed body, 400 on invalid rule), `TestListSessions` (list all, filter by type, filter by state, filter by participant, limit cap, unknown participant 404), `TestForceClose` (active session closed, unknown 404, default reason), `TestActorActivity` (sessions + tasks shape, name lookup, 404), `TestKnowledgeRead` (allowed reader + path 200, missing header 403, denied reader 403, denied path 403, missing file 404, unknown actor 404, empty-default denies, nested path preserved), `TestAdminHealth`, `TestAdminMetrics` (shape + counter movement after registration), `TestTaskStubs` (three 404 stubs with stable body).
- `test_cache_invalidation.py` (11) — `TestMemoryStoreInvalidation` (rule / identity rewrite invalidates, rule / identity delete drops, name change updates index), `TestSetRuleThroughCacheLayer` (set_rule is observed and post-state is stable), `TestCacheInvalidationLifecycle` (close is idempotent, no-op on sync constructor, no further invalidation after close), `TestDiskStoreInvalidation` (watchdog-driven real filesystem rewrite invalidates, close releases observer).
- `test_archival_sweeper.py` (10) — `TestArchiveClosedSessions` (happy path moves WAL + writes summary, active untouched, idempotent, age threshold skips fresh, deterministic fake-clock `now`), `TestExpiredSessionArchival` (TTL-expired eligible), `TestArchiveSweepCallback` (async callable, end-to-end), `TestArchiveDiskRoundTrip` (survives hydrate), `TestArchiveSummaryContent` (basic stats preserved).
- `test_sqlite_knowledge_store.py` (15) — `TestSqliteBasics` (read/write/list/delete/append/read_range/exists round-trip), `TestSqliteOnChange` (fires on write / append / delete, prefix scope isolates siblings, close stops delivery, close idempotent, two independent subscriptions), `TestSqlitePersistence` (reopen preserves writes and append offsets), `TestSqliteConcurrency` (parallel appends serialize through the asyncio lock without byte interleaving).
- `test_redis_knowledge_store.py` (18) — `TestRedisBasics`, `TestRedisAppend`, `TestRedisOnChange` (same six on_change cases as Sqlite), `TestRedisKeyPrefix` (tenant isolation via custom `key_prefix`), `TestRedisConcurrency`. Entire module uses `fakeredis.aioredis` via `pytest.importorskip` so it skips cleanly if the dep is missing.
- `test_knowledge_store_extensions.py` (+13) — the existing parameterized conformance suite now covers Memory / Disk / LockedMem / Sqlite (53 rows, up from 40) so every KnowledgeStore method is held to the same contract on the new backend.

Proof point: **676 tests pass** under `test/beta/network/` (510 Phase 4 baseline + 166 new Phase 3b), zero failures, zero regressions against Phase 3a. Full beta suite (`test/beta/`) reports **1,258 passed, 12 skipped, 0 failures** — framework-core stays clean.

Lingering items carried forward (none are blockers):

- Cross-process subscription fan-out stays deferred to Phase 7 as noted above. The Phase 3a `_subscriptions` in-memory map is still the sole fan-out path; multi-hub coordination is a federation concern.
- Archived sessions have their WAL at `hub/archive/sessions/{id}/wal.jsonl` but `Hub.read_wal` still reads from the live path only — callers that want archived bytes read them directly from the store. A transparent fallback is trivial to add if operator demand surfaces.
- Two cosmetic lint warnings in `autogen/beta/network/transport/frames.py` (`typing.Union` used via `# noqa: UP007`, defensive unreachable `raise TypeError`) — inherited from Phase 2, still harmless, still untouched.
- `RuleChangedFrame` emission from `set_rule` lands the frame on the wire but does not trigger any Phase 3b behavior on the `ActorClient` side — Phase 5a is where the transforms pipeline actually consumes it.

### Phase 4 — Network Tasks ✅ **Done**

Goal: first-class network `Task` state machine (see §6) layered on top of sessions. Phase 4 runs in parallel with Phase 3 on the same branch; its contracts are strictly additive on Phase 2's stable session/WAL/adapter surface so the two lanes do not collide. Phase 4 ships the programmatic API only — `Session.create_task`, the `Task` client handle, `Hub.create_task` / `cancel_task` / `expire_due_tasks`, and the default task handler — so Phase 6 has a stable surface to auto-inject the LLM verb wrappers (`run_task` / `start_task` / `track_task`) on top of.

Action items:

- ✅ **Task primitives** (`autogen/beta/network/task.py`). New `TaskState` enum (`created` / `running` / `paused` / `completed` / `failed` / `cancelled` / `expired`, str-subclass for wire-format stability), `TERMINAL_TASK_STATES` frozenset, `TaskPhase` (id + description + started_at/completed_at timestamps), `TaskSpec` (title, description, phases, `spec_type` for handler routing, free-form `payload` dict), and `TaskMetadata` (the durable hub record — see §6.1 for the full shape). Every dataclass round-trips through `to_dict` / `from_dict`; `TaskMetadata` additionally exposes `to_json` / `from_json` / `is_terminal()` / `copy()` so the hub can persist/load it from `KnowledgeStore` without a custom encoder.
- ✅ **`ag2.task.*` event registry.** Eight stable event names (`assigned` / `phase_entered` / `phase_completed` / `progress` / `result` / `error` / `cancelled` / `expired`) added to `BUILTIN_EVENT_TYPES` so a strict-mode `EventRegistry` accepts them without operator registration. New `TASK_EVENT_TYPES` and `TASK_TERMINAL_EVENT_TYPES` frozensets are exported for client-side filtering and the hub's task-event branch dispatch. The Phase 1 `Envelope.task_id` field — reserved but unused since day one — is now actually populated; round-trips through every wire format including `Envelope.from_json` / `to_json`.
- ✅ **Task FS layout** (`autogen/beta/network/hub/layout.py`). New `TASKS_ROOT = /hub/tasks`, `task_dir(task_id)`, and `task_metadata(task_id)` paths for the durable per-task record, plus `session_tasks_dir(session_id)` / `session_task_ref(session_id, task_id)` for the session→task back-reference pointers. Tasks live under their own root rather than nested inside `hub/sessions/{id}/tasks/` so `Hub.hydrate()` walks every task in one scan regardless of how many sessions they span; the per-session `.ref` pointer files are how `Session.track_tasks()` enumerates a session's tasks without re-scanning.
- ✅ **Task-event branch in `Hub.post_envelope`** — the load-bearing decision from the design lock-in (§6.2). Task envelopes (any `event_type in TASK_EVENT_TYPES`) flow through the same access / rate / depth / inbox checks that user envelopes do, but **bypass `adapter.validate_send` and `adapter.on_accepted`**. Implemented as an `if envelope.event_type in TASK_EVENT_TYPES: return await self._process_actor_task_event(...)` branch placed after the access checks and before the adapter is consulted. This is what keeps consulting's 1Q1R rule (and every other adapter delivery rule) orthogonal to task lifecycle: a task owner can emit unlimited `phase_entered` / `progress` / `phase_completed` / `result` events inside a consulting session without the adapter trying to auto-close it. The session WAL is still authoritative — task envelopes land in `wal.jsonl` alongside text envelopes, which is how subscribers see them and how hydrate reconstructs task state on cold restart.
- ✅ **Hub task state machine** (`autogen/beta/network/hub/core.py`). Two new in-memory caches: `Hub._tasks: dict[str, TaskMetadata]` (authoritative task record, mirrored on disk) and `Hub._session_tasks: dict[str, set[str]]` (non-terminal task ids per session, used by the close cascade and `tasks_for_session` lookups). New methods:
    - `Hub.create_task(*, session_id, requester_id, owner_id, spec, ttl_seconds=None) -> TaskMetadata` — direct call symmetric with `create_session`. Allocates a UUID7 `task_id`, validates session is `ACTIVE`, verifies both requester and owner are participants, enforces `owner.rule.limits.max_concurrent_tasks` against the owner's current non-terminal count, stamps `expires_at = created_at + (ttl_seconds or owner.rule.task_ttl_default)`, writes `metadata.json` + the session-side `.ref`, and emits an `ag2.task.assigned` envelope addressed to the owner.
    - `Hub.peek_task(task_id) -> TaskMetadata | None` / `Hub.get_task(task_id) -> TaskMetadata` — non-raising and raising read API.
    - `Hub.tasks_for_session(session_id) -> list[TaskMetadata]` — lookup over the in-memory cache for `Session.track_tasks()`.
    - `Hub.cancel_task(task_id, *, requested_by, reason="") -> TaskMetadata` — verifies `requested_by` is the requester or owner, no-ops on already-terminal tasks (returns current metadata for blocking-caller idempotency), emits a broadcast `ag2.task.cancelled` envelope.
    - `Hub.expire_due_tasks(*, now=None) -> list[str]` — TTL sweeper entry point. Walks `_tasks`, transitions every non-terminal task whose `expires_at <= now` to `expired`, emits `ag2.task.expired` per task, returns the list of expired ids. `now` defaults to `self._clock()` but tests pass an explicit ISO-Z string in the future to drive the sweep deterministically without wall-clock dependency.
- ✅ **State-machine validation** (`Hub._validate_actor_task_event`). Sender-authority check: `phase_entered` / `phase_completed` / `progress` / `result` / `error` are owner-only; `assigned` / `cancelled` / `expired` are hub-only and rejected from actors with `TaskStateError`. Terminal-state check: no transitions out of terminal — once a task is `completed` / `failed` / `cancelled` / `expired`, every further task envelope from any sender raises. Declared-phase check: if `task.spec.phases` is non-empty, `phase_entered` / `phase_completed` events must reference one of the declared phase ids; ad-hoc phases are allowed only when the spec was built without a phase plan.
- ✅ **State-machine application** (`Hub._apply_task_event`). Mutates `TaskMetadata` in place per event type and rewrites `hub/tasks/{task_id}/metadata.json`: `phase_entered` / `progress` advance `created → running` and stamp `started_at` on first transition; `phase_entered` updates `current_phase` and stamps the declared phase's `started_at`; `phase_completed` stamps the declared phase's `completed_at`; `progress` merges the `update` dict onto `task.progress` and bumps `last_progress_at`; `result` / `error` / `cancelled` / `expired` set the terminal state, stamp `completed_at`, and remove the task id from `_session_tasks` so the close cascade and `tasks_for_session` queries stay accurate. Both actor-posted and hub-emitted task envelopes go through the same shared `_apply_task_event` mutation path.
- ✅ **`Hub._post_hub_task_envelope`** — shared persistence + delivery + state-transition helper used by `create_task` (`assigned`), `cancel_task` (`cancelled`), `expire_due_tasks` (`expired`), and the session-close cascade. Hub-emitted task events go through the same `_wal_lock` + `_fanout_to_subs` invariant as actor-posted ones, so subscribers see assigned/cancelled/expired envelopes through the same race-free fan-out path that already covers progress/result.
- ✅ **Session-close task cascade.** `Hub.close_session` and `Hub._apply_adapter_result` (when an adapter transitions a session to `CLOSED`) call `_cancel_tasks_for_session(session_id, reason=...)` *before* `_broadcast_session_closed`, so subscribers see every task's `cancelled` envelope landing in the WAL ahead of the terminal `SessionClosed`. Tasks whose session is concurrently being torn down get a clean "task ended because the session closed" reason instead of a stale `running` state.
- ✅ **`max_concurrent_tasks` enforcement.** `Hub.create_task` counts the owner's non-terminal tasks and rejects with `LimitExceededError` if the count is already at `owner.rule.limits.max_concurrent_tasks`. `LimitsBlock` declared the field back in Phase 2 but left it unused — Phase 4 turns it on. Per-window task-rate budgets (`tasks_per_actor` / `tasks_per_session`) remain stored verbatim — see lingering items below.
- ✅ **Cold-restart hydrate.** `Hub.hydrate()` (Phase 2) gains a third pass after actors and sessions: walks `hub/tasks/*/metadata.json`, loads every task into `_tasks`, and re-adds non-terminal tasks to `_session_tasks` so the close cascade and `tasks_for_session` lookups still work after a process restart. Terminal tasks stay loaded for read-only inspection but are not re-added to the session→task index. Corrupt metadata files (parse failure) are logged and skipped — the same defensive stance Phase 2 takes for sessions.
- ✅ **Wire error code mapping.** Two new `ErrorFrame.code` strings: `unknown_task` (the envelope referenced a `task_id` the hub has never heard of) and `task_state` (state-machine violation: wrong sender, terminal-state reuse, bad phase id). `Hub._handle_send`'s exception ladder maps `UnknownTaskError → ErrorFrame(code="unknown_task")` and `TaskStateError → ErrorFrame(code="task_state")`; client-side `_error_for_code` translates them back so `Session.send` / `Task.result` / etc. raise typed exceptions instead of generic `SessionError`s.
- ✅ **Task error hierarchy** (`autogen/beta/network/errors.py`). New `TaskError` base, `UnknownTaskError`, `TaskStateError`, plus three terminal-resolution variants for the blocking-wait path: `TaskFailedError(reason, metadata)` / `TaskCancelledError(reason, metadata)` / `TaskExpiredError(metadata)`. The metadata-bearing variants give the caller the full terminal `TaskMetadata` for inspection.
- ✅ **`Task` client handle** (`autogen/beta/network/client/task.py`). Thin wrapper around the owning `Session` that stamps `task_id` on every envelope it emits. Methods: `phase_entered(phase_id, *, description="")`, `phase_completed(phase_id)`, `progress(**fields)`, `result(value)`, `fail(error)`, `cancel(*, reason="")`, `wait(*, timeout=None)`, `refresh()`. Owner-side methods route through `ActorClient._send_envelope` so transforms (Phase 5 seam), depth propagation, and inbox preflight all apply. `cancel` is special — it goes through `Hub.cancel_task` directly because both the requester and the owner can call it without needing the owner's network context. `wait` opens a session subscription from the current cursor, filters by `task_id`, resolves on the first envelope whose event type is in `TASK_TERMINAL_EVENT_TYPES`, and translates the terminal state into either a return value (`completed`) or one of `TaskFailedError` / `TaskCancelledError` / `TaskExpiredError`. If the task is already terminal at call time, `wait` returns synchronously without opening a subscription.
- ✅ **`Session.create_task` / `track_task` / `track_tasks`** (`autogen/beta/network/client/session.py`). `create_task(spec, *, owner=None, blocking=False, timeout=None, ttl_seconds=None)` resolves the owner (single non-initiator participant for the 2-party fast path; explicit name or actor_id otherwise; rejected if not a participant via a new `_resolve_owner` helper), calls `Hub.create_task` directly, wraps the returned `TaskMetadata` in a `Task` handle, and either returns the handle (`blocking=False`) or awaits `task.wait(timeout=...)` (`blocking=True`). The blocking path is purely composed on top of the non-blocking path — no new frames, no new hub primitives, just the same subscription mechanism `Session.ask` uses with a `task_id` predicate. `track_task(task_id)` / `track_tasks()` read the hub's task cache without touching the network.
- ✅ **`ActorClient` task handler registry + dispatch** (`autogen/beta/network/client/actor_client.py`). New `_task_handlers: dict[str, TaskHandler]` keyed by `TaskSpec.spec_type` (or `"*"` for the default fallback), populated at construction with `default_handlers.handle_task_assigned` under the `"*"` key. New `on_task(spec_type="*")` decorator mirrors the session-type `on()` decorator. Third dispatch branch in `_on_notify` for `ag2.task.*` envelopes: `assigned` envelopes go through `_dispatch_task_assignment` which builds a fresh `Task` handle from the hub's task cache, picks the handler by `spec_type` with `"*"` fallback, and runs the handler in a **background task** (mirroring the session-type handler dispatch pattern) so a handler that posts `phase_entered` / `progress` / `result` through `_send_envelope` does not deadlock against its own `AcceptFrame`s flowing back through the inbox loop. Other `ag2.task.*` events (the ones the hub already applied) are acked silently — the local `Task` instance refreshes lazily when the requester reads it.
- ✅ **Handler crash safety net.** If a registered task handler raises an exception, `_dispatch_task_assignment` catches it and posts a `TaskError` envelope on the handler's behalf via `task.fail(...)` so the requester's `task.wait()` resolves with `TaskFailedError` instead of hanging until TTL. Handles the edge case where the task may already be terminal (e.g. requester cancelled it while the handler was running) by suppressing `TaskStateError` / `UnknownTaskError` from the recovery path.
- ✅ **Default task handler** (`autogen/beta/network/client/handlers.py::handle_task_assigned`). Registered against `"*"` so any task whose `spec_type` does not match a custom registration runs the minimal useful behavior: call `actor.ask(spec.description or spec.title)` (with the same `SESSION_ID_VAR` + `HUB_DEP` injection the consulting / conversation handlers use, so `SessionInboxPolicy` still works inside a task handler), post the result through `task.result(...)`, and translate any exception into `task.fail(...)`. Custom handlers that want phases / progress / nested calls override via `client.on_task("my-spec-type")`.
- ✅ **Module exports.** `autogen/beta/network/__init__.py` adds the eight `EV_TASK_*` constants, `TASK_EVENT_TYPES`, `TASK_TERMINAL_EVENT_TYPES`, `TERMINAL_TASK_STATES`, `Task`, `TaskMetadata`, `TaskPhase`, `TaskSpec`, `TaskState`, and the full task error hierarchy. `autogen/beta/network/client/__init__.py` adds `Task` and `TaskHandler` alongside the existing `ActorClient` / `Session` exports.

Test coverage: **104 new tests under `test/beta/network/`** across nine new modules. New test modules:

- `test_task_primitives.py` (16) — `TaskState` enum values + str-subclass behavior, `TERMINAL_TASK_STATES` membership, `TaskPhase` / `TaskSpec` construction + round-trip, `TaskMetadata` round-trip in every state (created / running / completed / failed), `is_terminal()` matrix, `copy()` independence, missing-required-key rejection on `from_dict`.
- `test_task_events.py` (13) — eight `ag2.task.*` names registered as `BUILTIN_EVENT_TYPES`, four terminal events identified correctly, stable wire values pinned, permissive + strict `EventRegistry` both accept built-ins, strict mode rejects unknown `ag2.task.*` names, `Envelope.task_id` round-trip with default-None and explicit value, terminal-event payload (result value, error message) round-trip.
- `test_task_lifecycle.py` (15) — `TestCreateTask` (allocates task_id + writes metadata.json + session ref + handler sees same handle, rejects unknown / non-participant owner, enforces `max_concurrent_tasks` against an actively running task), `TestTaskEmissions` (`phase_entered` transitions to running and stamps phase started_at, `progress` merges without terminal, `result` transitions to completed with payload, `error` transitions to failed with reason, `phase_completed` stamps phase completed_at), `TestSessionBypass` (consulting session stays ACTIVE through full task lifecycles, multiple tasks per consulting session each reach completed, session close cancels active running tasks), `TestAuthorityGuards` (non-owner cannot emit task events, terminal task rejects further events, hub-only events forged from actors are rejected with `TaskStateError`).
- `test_task_blocking.py` (7) — `TestBlockingResolution` (returns terminal metadata on completion, raises `TaskFailedError` on failure with reason + metadata, raises `TaskCancelledError` when cancel arrives mid-wait, raises `NetTimeoutError` on deadline, raises `TaskExpiredError` when sweeper fires past the task's TTL), `TestNonBlockingHandle` (returns handle immediately with non-terminal state, `task.wait()` on already-terminal returns synchronously without opening a subscription).
- `test_task_ttl.py` (5) — `TestExpireDueTasks` (expires non-terminal task past deadline + persists state on disk, skips not-yet-due tasks, skips already-terminal tasks even with past `expires_at`, expiry envelope fans out to subscribers, multi-session isolation — only expired sessions touched).
- `test_task_cancel.py` (6) — requester cancel + transitions to cancelled, owner can cancel from inside its own handler, non-requester non-owner third-party rejected with `AccessDeniedError`, cancel on completed task is idempotent no-op, cancel on unknown task raises `UnknownTaskError`, cancel envelope fans out to non-participant subscribers.
- `test_task_restart_resume.py` (4) — `TestHydrateTasks` (hydrate rebuilds completed task from disk including result payload, hydrate rebuilds running task preserving `current_phase` and re-adds it to `_session_tasks` index, partial/corrupt metadata files skipped with a warning), `TestHydrateDisk` (full disk-store round-trip — drive a task to completion via real `Actor` + handler, tear hub + link down, open fresh `Hub.open` on the same disk root, confirm task is still readable with terminal state).
- `test_task_subscriptions.py` (4) — participant sees every task event from `assigned` through `result`, non-participant observer with `rule.access.subscribe.sessions = "public-within-hub"` sees task events end-to-end (most-restrictive vote requires every participant + observer to allow public observation), multiple subscribers all see the same task, cancellation envelope visible through subscription.
- `test_task_integration.py` (5) — `TestDefaultTaskHandler` (real framework-core `Actor` + `TestConfig` drives the default handler from `create → result`, default handler converts handler exceptions into `TaskFailedError`), `TestCustomTaskHandler` (multi-phase research handler running through a scripted Actor with two model calls returning distinct responses; both declared phases get their `started_at` + `completed_at` stamped; progress merges visible via `Task.refresh`), `TestTaskDurability` (full disk-store cold restart of a completed task driven by a real Actor + the default handler).

Proof point: **510 tests pass** under `test/beta/network/` (Phase 3a's 406-test baseline + 104 Phase 4 tests), zero failures. Full beta suite (`test/beta/`) reports **1,369 passed, 12 skipped, 1 xfailed**, zero regressions against framework-core. `grep -r run_subtask autogen/beta/network/` returns no matches — the framework-core/network task split from §6.5 is preserved verbatim.

Small lingering items carried forward to later phases (none are blockers, all explicitly listed for transparency):

- `TaskState.PAUSED` is declared, round-trips, and is rejected by the actor-side state machine (no `paused` envelope is wired). The `pause` / `resume` transitions slip to Phase 6 where the LLM verb surface naturally pairs with user-driven pause control.
- LLM verb wiring (`run_task` / `start_task` / `track_task` auto-injected into the agent turn surface) is Phase 6, as planned. The Phase 4 programmatic API is the contract Phase 6 verbs sit on top of.
- Cross-process task creation over `WsLink` / HTTP relies on `HubClient` holding a direct `Hub` reference (Phase 1's pattern). When Phase 3b lands the full HTTP admin surface, `HubClient.create_task` will swap its body for `POST /v1/sessions/{id}/tasks` without changing the public signature.
- `tasks_per_actor` and `tasks_per_session` rate-limit budgets remain stored verbatim on `LimitsBlock`. Phase 4 enforces `max_concurrent_tasks` only; per-window task-rate enforcement waits until the operator demand is concrete.

### Phase 5a — Rules MVP (named / python / http)

Goal: `rule.transforms` enforcement at `ActorClient` for the three forms that cover ~80% of production use cases, without the subprocess / bidirectional-WS lifecycle surface. Phase 5a is what ships to operators first.

Phase 5a is split into two sub-steps so the transforms pipeline ships independently of the cross-actor knowledge bridging, which has an open architectural question (see 5a.2 below). **Phase 5a.1 is the "transforms MVP" slice; Phase 5a.2 is the knowledge-exposure slice.** 5a.1 is the hard dependency for every downstream phase (6 depends on it to run turns through transforms; 7 depends on it to exist before federation adds a second axis of rule enforcement). 5a.2 can slip without blocking anything else.

#### Phase 5a — Step 0 (prerequisites) ✅ **Done**

Three audit findings from the Phase 5a readiness review were fixed on the planet-satellite-network branch before transforms work began, so 5a.1 starts on clean ground:

- ✅ **`Hub.set_rule` constructed `RuleChangedFrame` without the required `version` field.** Latent `TypeError` that only fired when an actor had a live endpoint at `set_rule` time; the Phase 3b `test_put_replaces_rule` coverage called `hub.register(...)` (no live endpoint) so the buggy branch was never reached. Fixed by passing `version=rule.version` through a new shared helper, and by adding live-endpoint regression coverage in `test_cache_invalidation.py::TestRuleChangedEmission`.
- ✅ **`_reload_actor_rule` refreshed the cache but never emitted `RuleChangedFrame`**, breaking §4.3 for operator-edited rules written directly to the store. Extracted `Hub._emit_rule_changed(actor_id, rule)` (`hub/core.py:937-981`) and wired it into both `set_rule` and `_reload_actor_rule` so the §4.3 "any write to rule.json emits a RuleChanged event" contract holds on both paths.
- ✅ **Double-emission on synchronous-`on_change` stores.** `set_rule` writes the rule, which fires `_reload_actor_rule` inline on `MemoryKnowledgeStore` (and any Phase 3b store whose `on_change` is synchronous), producing two `RuleChangedFrame`s per logical rule change. Fixed by deduping in `_emit_rule_changed` against a per-actor `_last_emitted_rule_json` snapshot — byte-identical re-emits are dropped, so exactly one frame fires per logical change regardless of which hub entry point observed it. The snapshot is evicted on `unregister` and on the identity-deletion branch of `_reload_actor_identity` so no stale state leaks across registrations.

#### Phase 5a.1 — Transforms pipeline + 3 adapters + standard library ✅ **Done**

Goal: ship the four-stage `ActorClient` pipeline with the three MVP adapter forms (named / python / http), a frozen `TransformContext` shape, a minimum viable `when` matcher, the `rule_changed` live-swap path, and a three-entry standard library — every piece shipping inside the actor's address space so tenant code never executes in the hub.

Action items:

- ✅ **Transforms package skeleton** (`autogen/beta/network/client/transforms/`). New package with `protocol.py` (`Transform` protocol, `TransformContext`, `TransformError`/`TransformLookupError`/`TransformRejected`), `registry.py` (`TransformRegistry` per-client factory lookup), `when.py` (minimal matcher), `adapters.py` (all three adapter classes colocated because each is small), `pipeline.py` (four-stage driver with lifecycle), `stdlib.py` (three built-in named transforms + installer), and `__init__.py` re-exporting the full public surface.
- ✅ **`Transform` protocol** (`transforms/protocol.py`). One runtime-checkable async callable, `(envelope, ctx) → envelope | None`. Return `None` to reject. No inheritance requirement — structural subtyping, matching every other protocol in the framework. Raising is reserved for unexpected errors and is treated as reject-plus-log by the pipeline (§4.1).
- ✅ **`TransformContext` dataclass — shape frozen in 5a.1.** Five fields: `stage` (`TransformStage` enum), `client` (the owning `ActorClient` for identity/rule reads), `session_id` (nullable for forward-compat with session-less envelopes), `rule_version` (bumped on every `RuleChanged` push so named transforms can decide to reset cached state), and `direction: Literal["inbound", "outbound"]` (derived from stage as filter sugar). Freezing the shape is deliberate — 5b's sidecar adapters will consume the same context and user-written named transforms live longer than any phase; later additions are strictly additive, nothing is removed or renamed.
- ✅ **`TransformStage` enum + `TransformSpec.from_dict` validation** (`autogen/beta/network/rule.py`). New `TransformStage(str, Enum)` with `PRE_SEND` / `POST_SEND` / `PRE_RECEIVE` / `POST_RECEIVE` members that compare equal to their literal wire values (same str-subclass trick as `SessionType`). Added `_VALID_STAGES` frozenset and `TransformSpec.__post_init__` plus `TransformSpec.from_dict` / `Rule.from_dict` rejection of unknown stage strings with a clear `ValueError("TransformSpec.stage must be one of [...]")`. Phase 1 stored `stage` verbatim for forward-compat; that was the right call for `apply` (new adapter forms ship over time) but wrong for `stage` (the four stages are not going to grow) — a typo silently disabling a transform is the worst kind of bug. `TransformStage` lives in `rule.py` (not in `client/transforms/`) so parser-side validation needs no client-runtime import.
- ✅ **`NamedTransform`** (`transforms/adapters.py`). Takes a `(name, registry)` pair and eagerly resolves the delegate at construction so a dangling-name rule fails at pipeline-build time, not per-envelope. Forwards `aclose()` to the delegate if the delegate defines one, so long-lived registered instances (pooled clients, rate-limit buckets) get drained on pipeline rebuild and on `ActorClient.disconnect`.
- ✅ **`PythonTransform`** (`transforms/adapters.py`). Parses `apply={"python": {"module": "...", "class": "...", "config": {...}}}`, imports the module via `importlib.import_module`, resolves the class, and instantiates it with `**config` as keyword args, falling back to a single-dict-argument constructor if the class rejects the kwargs form (so both `def __init__(self, max_tokens=8000)` and `def __init__(self, config: dict)` work without rule-author ceremony). Clear `TransformError` subclasses for every parse-time failure (missing `module`, missing `class`, non-importable module, missing attribute, config not a dict). The adapter only runs inside the recipient's `ActorClient` — the hub never calls `importlib.import_module` on tenant data (§4 isolation invariant). `aclose` forwards to the instance if defined.
- ✅ **`HttpTransform`** (`transforms/adapters.py`). Takes a local sidecar URL, lazily constructs one `httpx.AsyncClient` per transform instance with `Limits(max_keepalive_connections=10, max_connections=50)` and `timeout=5.0s`, and POSTs the envelope JSON on every call. Response decoding: `200 application/json` → `Envelope.from_dict(body)` (mutate), `204` → return the inbound envelope unchanged (pass), `409` → return `None` (reject, logs `response.text` as the reason), any other status → log + reject (matching "unexpected errors are reject" from §4.1). `httpx` import is lazy so the network package stays install-optional. Pooled client is drained in `aclose` (called from pipeline rebuild and from `ActorClient.disconnect`); `aclose` is idempotent.
- ✅ **`TransformPipeline` driver** (`transforms/pipeline.py`). Holds four `list[CompiledTransform]` buckets keyed by `TransformStage`. `TransformPipeline.build(rule, registry=...)` compiles every `TransformSpec` in declaration order — string apply → `NamedTransform`, `{"python": ...}` → `PythonTransform`, `{"http": ...}` → `HttpTransform`. Dispatch methods `run_pre_send` / `run_post_send` / `run_pre_receive` / `run_post_receive` iterate their stage list, short-circuit on `None` for the mutating stages, swallow-and-log exceptions on the side-effect stages, and stamp every invocation with a fresh `TransformContext` so transforms see the correct `stage` / `direction` / `rule_version`. `aclose()` drains every `CompiledTransform` in parallel for pipeline rebuild and shutdown.
- ✅ **Minimum viable `when` matcher** (`transforms/when.py`). Two keys only: `event: <event_type>` (exact match against `envelope.event_type`) and `session_type: <type>` (exact match against the cached session metadata's type, resolved via `ActorClient._session_type_for`). Empty `when` always matches. AND semantics across keys — OR combinators deliberately not shipped. Unknown keys are ignored (forward-compat for 5b additions). A transform that needs regex or boolean combinators writes a `PythonTransform`.
- ✅ **Unknown apply forms (`exec` / `ws`) pass through with a one-shot warning.** `pipeline._compile_spec` catches the internal `_UnknownForm` sentinel, dedupes against a per-pipeline `_warned_unknown_forms: set[tuple[stage, form]]`, logs `transform.unsupported_form stage=... form=...` exactly once per pair, and skips the transform (returns `None` from the compiler so the slot is absent from the stage list). Rules authored against the full five-form surface therefore do not break on a 5a hub and flip to real execution in 5b without touching any of the 5a.1 wiring.
- ✅ **`ActorClient` pipeline integration** (`autogen/beta/network/client/actor_client.py`). The empty Phase 1 hooks at lines 682-694 became real dispatches to `self._pipeline.run_*`. `__init__` now builds a `TransformRegistry`, optionally seeds the stdlib via the `install_stdlib_transforms=True` kwarg, builds the initial `TransformPipeline` from `rule.transforms`, and owns a `_pipeline_lock: asyncio.Lock` for atomic rebuilds. `_apply_pre_send_transforms` raises `TransformRejected(stage=PRE_SEND)` when the pipeline rejects so the local `Session.send` raises to the caller (§4.1). `_apply_pre_receive_transforms` returns `None` so the existing `_on_notify` nack path (`reason="transform_rejected"`) handles inbound rejection unchanged. `post_send` / `post_receive` stage exceptions are logged and suppressed so side-effect handlers cannot break the delivery path. `register_transform(name, factory)` and `transform_registry_names()` expose the per-client registry for tests and for tenant-side glue that wants to add reusable logic without a Python dotted path.
- ✅ **`rule_changed` frame consumed on the `ActorClient` side.** `_dispatch_frame` gains a `RuleChangedFrame` branch that re-reads the rule from `self._hub.get_rule(actor_id)` (the hub cache is already updated by `set_rule` / `_reload_actor_rule` before the frame is pushed) and schedules `_rebuild_pipeline(new_rule)` in a background task — so the inbox loop never blocks on adapter teardown (`HttpTransform.aclose`, etc). `_rebuild_pipeline` holds `_pipeline_lock` for the swap-and-drain, builds a fresh pipeline, swaps `self._pipeline` and `self._rule` atomically, then `aclose`s the old pipeline after releasing the lock. In-flight transforms read `self._pipeline` once per envelope at dispatch time, so a concurrent rebuild cannot yank the pipeline out from under an envelope mid-flight.
- ✅ **`HubClient.register(install_stdlib_transforms=True)` kwarg.** Threaded through to `ActorClient.__init__`. Default is opt-in for the Phase 5a.1 stdlib; tests and deployments that want a pristine registry pass `install_stdlib_transforms=False`.
- ✅ **`ActorClient.disconnect` drains the pipeline.** After cancelling handler tasks and closing the link, `disconnect` calls `self._pipeline.aclose()` so adapter-owned state (pooled httpx clients, stateful Python instances) is released cleanly. Idempotent; safe across reconnect.
- ✅ **Rule write API validated end-to-end.** `PUT /v1/actors/{id}/rule` landed in Phase 3b (`http/server.py:320-337`) but the 3b test suite could not observe the full round-trip because `hub.register(...)` produces no live endpoint (the buggy `RuleChangedFrame` branch never fired). Phase 5a.1's `test_transforms_integration.py::TestRuleChangedLiveSwap` runs the complete path: a live `HubClient`-registered `ActorClient`, then `hub.set_rule(...)` swaps the pipeline in place, then subsequent envelopes observe the new transforms.
- ✅ **Standard library named transforms** (`transforms/stdlib.py`). Three entries, installed by default into every `ActorClient`'s registry:
  - `redact_pii` — regex-based stripper for email addresses (`[a-zA-Z0-9._%+-]+@...`), US-style phone numbers (`\b(?:\+?1)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b`), and US SSN patterns (`\b\d{3}-\d{2}-\d{4}\b`). Replaces with `[REDACTED:email]` / `[REDACTED:phone]` / `[REDACTED:ssn]` markers. Non-text envelopes pass through so the transform is safe to install globally.
  - `truncate_long_content` — caps `event_data["content"]` at `DEFAULT_TRUNCATE_BYTES=32*1024` (character count for MVP simplicity), appending a `…[truncated]` marker. Non-text envelopes pass through.
  - `stamp_audit_header` — annotates `event_data["_audit"]` with `{stage, sender_id, trace_id, actor_id, rule_version}`. Pure annotation, never rejects. Idempotent: re-stamping overwrites the prior block so chaining in both pre and post stages is harmless.
  `install_stdlib_transforms(registry)` is the installer the `ActorClient` constructor calls; importing `autogen.beta.network.client.transforms.stdlib` gives operators direct access to install a subset on a custom registry.
- ✅ **Module exports.** `autogen/beta/network/__init__.py` re-exports 11 new names: `Transform`, `TransformContext`, `TransformStage`, `TransformRegistry`, `TransformPipeline`, `NamedTransform`, `PythonTransform`, `HttpTransform`, `TransformError`, `TransformLookupError`, `TransformRejected`. `autogen/beta/network/client/transforms/__init__.py` re-exports the same surface plus `when_matches` for operator use. The existing `TransformSpec` export is preserved and now appears alongside `TransformStage` in both the module and `__all__`.

Test coverage: **76 new tests under `test/beta/network/`** across one updated module and three new modules:

- `test_rule.py` (+15) — new `TestTransformStageValidation` class: four-stage accept matrix via `pytest.mark.parametrize`, enum member interchangeable with its value, six bad stage strings (`"presend"`, `"Pre_Send"`, `"PRE_SEND"`, …) all raise `ValueError`, `TransformSpec.from_dict` rejection, missing-`stage` field raises, `Rule.from_dict` transitively rejects bad stages, stable wire values pinned (`pre_send` / `post_send` / `pre_receive` / `post_receive`).
- `test_cache_invalidation.py` (+3) — new `TestRuleChangedEmission` class (also the Step 0 regression): `set_rule` against a live endpoint captures a `RuleChangedFrame` with `version=rule.version` and the full transform payload, an out-of-band `store.write(actor_rule)` also emits the frame (dedupes correctly so exactly one frame per logical change), `set_rule` against an offline actor is silent and does not raise.
- `test_transforms_unit.py` (48) — `TestTransformRegistry` (6): register/create/replace/unregister, unknown-name raises `TransformLookupError` listing known names, factory produces fresh instances. `TestWhenMatcher` (6): empty always-matches, `event` key exact match, `session_type` key, unknown session id rejects, AND semantics, unknown keys forward-compat. `TestNamedTransform` (3): delegation, unknown name fails at construction, rejection propagates. `TestPythonTransform` (10): kwargs constructor happy path, zero-arg constructor, config-dict fallback constructor (the single-arg form), rejection propagates, unknown module/class/missing-field/non-dict-config all raise `TransformError`, `aclose` no-op on missing hook. `TestHttpTransform` (9): empty URL rejected, `200` decodes envelope, `204` passes through, `409` rejects, `500` fails-as-reject, connection refused fails-as-reject, malformed `200` body fails-as-reject, pooled client reuse across three envelopes, `aclose` is idempotent. `TestTransformPipeline` (14): empty-pipeline passthrough, declaration order preserved, rejection short-circuits subsequent transforms, `pre_send`/`pre_receive` independent, post-stage side-effects run, post-stage exception is logged not raised, pre-stage exception is reject, unknown `exec` form logs exactly once per `(stage, form)` and passes through, unknown `ws` form logs once across two specs, `when` filter gates per-envelope, `rule_version` propagates into `TransformContext`, direction is `"outbound"` for pre_send and `"inbound"` for pre_receive, `aclose` closes wrapped adapter instances via `NamedTransform`'s forwarding, unknown-name build-time failure.
- `test_transforms_integration.py` (10) — `TestStdlibTransformsEndToEnd` (4): `redact_pii` on `pre_receive` strips email before Bob's handler sees it, `redact_pii` on `pre_send` strips phone before the envelope leaves Alice, `truncate_long_content` preserves short content, `stamp_audit_header` annotates inbound envelopes and the custom handler observes the `_audit` block. `TestRuleChangedLiveSwap` (1): `hub.set_rule` swaps Bob's pipeline in place (`rule_version` advances from 1 to 2) and a subsequent send flows through the new `redact_pii` stage. `TestTransformRejection` (2): `pre_send` reject raises `TransformRejected` to the caller and the receiver never sees the envelope, `pre_receive` reject scoped via `when={"event": "ag2.msg.text"}` nacks text envelopes without blocking the session handshake. `TestUnknownForms` (1): an `exec` apply form is logged once and behaves as pass-through end-to-end. `TestMultiTenantIsolation` (1): upload a rule with a `PythonTransform` pointing at a sentinel module name — `sys.modules` must not contain the sentinel before, during, or after `hub.set_rule` (the hub must never `importlib.import_module` tenant paths), and the cached `TransformSpec` stores the raw dict without an instantiated transform object. `TestPythonTransformE2E` (1): a module-level `_StampClass` imported via dotted path mutates the content with the current stage through a real `ActorClient` notify flow.

Proof point: **752 tests pass** under `test/beta/network/` (Phase 4 + 3b baseline of 676 + 76 Phase 5a tests), zero failures. Full beta suite (`test/beta/`) reports **1,611 passed, 12 skipped, 1 xfailed**, zero regressions against framework-core. The multi-tenant isolation test asserts the hub never calls `importlib.import_module` on tenant data even when uploading a rule with a `PythonTransform` sentinel name, keeping the "tenant code never enters the hub process" invariant from §4 provable rather than aspirational.

Small lingering items carried forward to later phases (none are blockers, all explicitly listed for transparency):

- Phase 5a.2 (cross-actor knowledge exposure) is still pending — see the dedicated sub-section below. 5a.1's `rule.access.knowledge` enforcement runs through the existing Phase 3b HTTP endpoint as-is; it just talks to `hub._store` rather than the actor's private store, which is the architectural question 5a.2 resolves.
- `TransformContext.direction` is informational today — no built-in transform branches on it. Kept in the frozen shape because stamping both `stage` and `direction` gives user transforms a legible handle for per-direction filtering without re-deriving the enum membership.
- `PythonTransform` instantiation is per-`TransformSpec`, not per-`(module, class, config_hash)`. The design doc mentioned a cache of the latter; the actual implementation instantiates one instance per spec because every compiled pipeline is short-lived (bounded by rule-version changes) and the cache would complicate `aclose` bookkeeping. If a user hits the repeat-imports cost, a follow-up can add back the shared cache without changing the adapter contract.
- The two cosmetic `frames.py` linter warnings inherited from Phase 2 (`typing.Union`, defensive unreachable branch) remain untouched — still harmless, still out of scope for a transforms pass.

#### Phase 5a.2 — Cross-actor knowledge exposure (deferred to 5b, default)

Goal: make `rule.access.knowledge.expose` a real private-store bridge, not a convention on the shared hub store.

**Architectural finding from the 5a readiness review.** Phase 3b's `GET /v1/actors/{id}/knowledge/{path}` endpoint reads from `hub._store` at the conventional prefix `/actors/{owner_id}/knowledge{path}`. The hub has zero handle to any registered actor's private `KnowledgeStore` — `Actor._knowledge_store` is actor-local and never registered with the hub. The 3b test suite works only because it pre-populates data via `hub._store.write(...)` directly; a real `Actor` that writes to its own private store never shows up through the cross-actor read endpoint. §3.2's `knowledge_bases` field says these are "paths into its private store" — current wiring contradicts that.

Three options, pick one in 5a.2 (or defer to 5b if nothing lands):

1. **Actor-to-hub knowledge bridge frame.** On register, the `ActorClient` declares its private store; the hub stores a weakref-like handle. A `KnowledgeReadFrame(path, reader_id)` is pushed to the owning `ActorClient` over the live link when an HTTP `GET /v1/actors/{id}/knowledge/...` request arrives; the client runs the access check a second time locally (defense in depth) and returns `KnowledgeReadResultFrame(content | not_found)`. The hub forwards the result to the HTTP caller. Private stores stay private; cross-actor reads are frame-routed; offline actors return 503. **Most faithful to §3.2** but adds a round-trip per read.
2. **Explicit mirror path.** The `ActorClient` declares a subset of its own store to be mirrored into the hub's store at register time (and watches for further writes). Cross-actor reads go through the hub store, same as today, but the private store stays authoritative and the mirror is tagged as stale if the actor disconnects. Simpler than (1) but duplicates data; fine for small files (SKILLs, summaries) but not for large knowledge bases.
3. **Document the current convention and move on.** Rename the endpoint's semantics in the design doc: "shared-slot knowledge" is a hub-writable prefix that any participant can use, and private stores are *not* cross-actor-readable. §3.2 and §9.1 get updated to match. Defer (1) / (2) until a concrete user ask lands.

**Default plan**: slip 5a.2 to 5b and adopt option 3 for 5a by documenting the shared-slot semantics explicitly. If a concrete 5a user needs option 1 or 2 before 5b, revisit.

Action items (if 5a.2 ships in 5a rather than slipping):

- Pick one of the three options, update §3.2 / §9.1 to match.
- Implement the chosen bridge / mirror / convention.
- End-to-end test: an `Actor` with a private `MemoryKnowledgeStore` exposes `/public/**`, a peer reads via `GET /v1/actors/{id}/knowledge/public/doc.txt` and sees the private content.
- Audit-log coverage: every read attempt writes a `read_knowledge` entry per Phase 3b shape.

If 5a.2 slips, the existing Phase 3b endpoint continues to serve the shared-slot path unchanged — 5a.1's rule write API already exercises `rule.access.knowledge` through it end-to-end.

### Phase 5b — Rules full surface (exec / ws)

Goal: land the two sidecar forms whose complexity is process / stream lifecycle, not rule semantics. Everything in Phase 5a stays as-is; this phase only adds two new adapters and their lifecycle machinery.

Action items:

- `ExecTransform` — long-lived subprocess, JSON-lines stdio. Framework owns start-on-first-use, restart-on-crash (bounded retry with backoff), drain-on-shutdown. Per-line request/response correlation so multiple envelopes can be in flight against one subprocess.
- `WsTransform` — long-lived outbound WebSocket to a sidecar. Framework owns the same lifecycle as `ExecTransform`, plus reconnect-on-drop. Per-message correlation ids.
- Remove the Phase 5a pass-through warning for unknown `exec` / `ws` forms — they now execute.
- Stress tests for sidecar restart under load.

Test coverage: each apply form end-to-end (reject, mutate, pass-through); subprocess restart-on-crash scenario; WS reconnect scenario; the Phase 5a pass-through regression is flipped into a full execution test; multi-tenant isolation regression is re-run across all five forms.

### Phase 6 — LLM Surface

Goal: make V3 usable from inside a model turn. (`HumanClient` already landed in Phase 3 — Phase 6 just wires an LLM agent into the same network-verbs surface that any actor already has.)

Action items:

- Ten auto-injected network verbs (`find_actors`, `describe_actor`, `open_session`, `say`, `listen`, `run_task`, `start_task`, `track_task`, `read_session`, `leave`) attached only when `ActorClient` is dispatching a turn.
- Framework-core DI: `Session`, `Task`, `HubClient`, `ActorClient` injectable into user-defined tools via `Annotated[..., Inject()]`.
- Additional UI surfaces for `HumanClient` beyond the Phase 3 CLI: `HumanTextualClient` (TUI), `HumanWebClient` (webhook / iframe).

Test coverage: every verb invoked from a synthetic agent turn; DI resolution inside user-defined tools; LLM-actor opening a session with a `HumanClient` participant via the verb surface; additional `HumanClient` surfaces passing the same end-to-end handshake test Phase 3 introduced.

### Phase 7 — Multi-hub and Federation

Goal: exercise multi-identity / multi-hub; cross-hub federation.

Action items:

- End-to-end multi-identity / multi-hub fixture (one `Actor` holding two `ActorClient`s on two hubs).
- Federation manifest format + signed envelope chain.
- Explicit peering (add peer by URL + public key).
- `MtlsAuth` + `SignedChallengeAuth` adapters.
- Admin endpoints + audit log format.

Test coverage: multi-hub registration; federation manifest sign / verify / mount under prefix; cross-hub session handshake under most-restrictive rule merging; signed envelope chain verification; peer trust rotation; admin audit log shape.
