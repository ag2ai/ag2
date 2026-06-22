---
title: Multi-Agent Network Overview
sidebarTitle: Overview
---

The `#!python autogen.beta.network` module turns one or more `#!python Agent` instances into a **multi-agent network** — a hub-and-spoke topology where a central registry coordinates channel-based, protocol-driven exchanges between named agents.

It is fully **opt-in**. Bare `#!python Agent` continues to work standalone with no behavioural change when this package is not imported. Adopt the network when you need any of the following:

- Multiple agents coordinating on the same task with **enforceable turn order**, expectations, and audit trails.
- A **registry of named agents** that can find each other by capability.
- **Durable messaging** with replayable channel transcripts (write-ahead log).
- **Governance**: per-agent rules (access, rate, inbox, limits), per-adapter expectations, and a hub-side audit log.
- **Sub-task observability**: `#!python agent.task(...)` lifecycle events automatically forwarded to the hub when run inside a network turn.
- **Distributed / cross-process deployment**: replace `#!python LocalLink` with `#!python WsLink` to run the hub and each agent as separate OS processes connected over WebSocket.

## The Mental Model

```
                          ┌────────────────┐
                          │      Hub       │  ←── audit log, registry,
                          │ ── adapters ── │       channels, sweepers,
                          │ ── channels ── │       expectation evaluators
                          │ ── audit log ──│
                          └────────┬───────┘
                                   │  in-process duplex (LocalLink)
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
          ┌──────────┐       ┌──────────┐       ┌──────────┐
          │AgentClient│      │AgentClient│      │AgentClient│
          │  alice    │      │   bob     │      │   carol   │
          │  Agent    │      │  Agent    │      │  Agent    │
          └──────────┘       └──────────┘       └──────────┘
```

Each `#!python Agent` is wrapped by an `#!python AgentClient`, which lives behind a `#!python HubClient` and connects to the hub through a `#!python LinkClient`. The hub holds the authoritative state; clients are thin frontends that send and receive `#!python Envelope`s through framed messages on their link.

!!! note
    The default transport is `#!python LocalLink` — same-process duplex queues. Swap it for `#!python WsLink` to run each agent in its own process or on a separate host; the `#!python HubClient` API is unchanged. See [Distributed Deployment](distributed.md).

## Core Concepts

| Concept | Lives in | Purpose |
|---|---|---|
| `#!python Hub` | One per network | Authoritative state: registry, audit log, channel table, write-ahead logs, expectation evaluators, sweepers |
| `#!python Passport` | Hub registry | Stable identity (`name`, `agent_id`, owner, model) — identifies who's on the network |
| `#!python Resume` | Hub registry | Capability claims (`claimed_capabilities`, `domains`, `summary`) plus hub-mutated `observed` track record |
| `#!python Rule` | Hub registry | Per-agent governance: access lists, rate limits, inbox caps, channel-type allowlists |
| `#!python HubClient` | One per process | Process-side connection to the hub; manages registration and bookkeeping |
| `#!python AgentClient` | One per agent | Wraps a single `#!python Agent`; sends envelopes, receives notify frames, runs handlers |
| `#!python HumanClient` | One per human | Non-LLM participant (HITL) — same envelope plumbing, no `#!python Agent`; your UI drives it via push (`on_envelope`) / pull (`next_envelope`) |
| `#!python HubListener` / `#!python HubArbiter` | Registered on the hub | `#!python HubListener` observes state transitions after the fact (the audit log is one); `#!python HubArbiter` is the gatekeeper consulted *before* register / open / send / dispatch decisions |
| `#!python Envelope` | The wire format | Hub-stamped record of one event in one channel: `event_type`, `event_data`, `sender_id`, `audience`, `causation_id` |
| `#!python Channel` | Created by `agent_client.open(...)` | A bounded multi-party exchange governed by an adapter |
| Channel adapter | One of `#!python ConsultingAdapter` / `#!python ConversationAdapter` / `#!python DiscussionAdapter` / `#!python WorkflowAdapter` | Defines the channel's allowed sends, default view policy, expectations, and termination rules |
| `#!python TransitionGraph` | Workflow only | Declarative orchestration: who speaks first, what conditions fire, when to terminate |
| `#!python TaskMirror` | Auto-attached per turn | Bridges `#!python Task` lifecycle events into the hub so capability tags update `#!python Resume.observed` |

## Lifecycle of a Channel

```
agent_client.open(type=..., target=...)
    │
    ▼
INVITED ──┬─ all targets ack ──→ ACTIVE ──┬─ adapter terminates ──→ CLOSED
          │                                │
          └─ ack timeout ──→ CLOSED       └─ explicit channel.close() ──→ CLOSING ──→ CLOSED
                  (expectation violation)         or TTL expired
```

The four built-in adapters differ in how they govern the `ACTIVE → CLOSED` transition:

| Adapter | Participants | Turn order | Termination |
|---|---|---|---|
| `#!python conversation` | Exactly 2 | Free-form (either side, any time) | Explicit close or TTL |
| `#!python consulting` | Exactly 2 | Strict 1Q1R (initiator → respondent) | Auto-closes after respondent's reply |
| `#!python discussion` | 2+ | Round-robin via `ordering="round_robin"` | Explicit close or TTL |
| `#!python workflow` | 2+ | Declarative `TransitionGraph` | Graph terminates (`TerminateTarget` / `max_turns`) |

See the [Channel Adapters](adapters_overview.md) overview for the full picture.

## Reading Order

If you're new to AG2 Beta:

1. Start with the [Quick Start](quick_start.md) page. It's the smallest end-to-end network scenario.
2. Then [Hub & Identity](hub_and_identity.md) for the registry-side primitives.
3. Then [Agent Clients](agent_clients.md) for the agent-side primitives.
4. Finally choose an adapter from the [Channel Adapters](adapters_overview.md) overview based on what you're building.

If you're migrating from the classic `GroupChat` / handoff orchestrations:

1. Skim the overview and quick start.
2. Read [Migrating from Group Chat](migration_from_group_chat.md). It maps every classic concept onto its `#!python WorkflowAdapter` counterpart with side-by-side code.

If you're operating an existing deployment:

- [Governance, Audit & Observability](expectations_and_audit.md) covers expectations, the audit log, `#!python HubListener` / `#!python HubArbiter`, and turn-failure handling.
- [Task Observation](task_observation.md) covers per-capability track records.
- [Views & Skills](views_and_skills.md) covers what each agent sees of its own channel and how its capabilities are advertised.
- [Distributed Deployment](distributed.md) covers `#!python WsLink`, `#!python serve_ws`, auth, at-least-once delivery, reconnect, and task durability.
