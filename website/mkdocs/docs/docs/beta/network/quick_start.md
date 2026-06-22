---
title: Quick Start
sidebarTitle: Quick Start
---

The smallest possible end-to-end network scenario: one in-process hub, two agents, a `#!python consulting` channel that auto-closes after a single Q-and-A.

```python linenums="1"
import asyncio

from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    EV_CHANNEL_CLOSED,
    EV_TEXT,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)

async def main() -> None:
    config = AnthropicConfig(model="claude-sonnet-4-6")

    # Hub: registry + WAL + audit log + adapters live here.
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
    link = LocalLink(hub)  # in-process duplex transport

    # Each agent gets its own HubClient (its own duplex pair to the hub).
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(
        Agent("alice", prompt="Ask one focused question and stop.", config=config),
        Passport(name="alice"),
        Resume(),
    )
    bob = await bob_hc.register(
        Agent("bob", prompt="Answer in one short sentence.", config=config),
        Passport(name="bob"),
        Resume(),
    )

    # Strict 1Q1R; the adapter auto-closes on bob's reply.
    channel = await alice.open(type="consulting", target="bob")
    await channel.send(
        "What's the single most important property of a distributed system?",
        audience=[bob.agent_id],
    )

    # Bob's default handler runs Agent.ask on the inbound EV_TEXT, sends the
    # reply, and ConsultingAdapter posts EV_CHANNEL_CLOSED.
    close_env = await alice.wait_for_channel_event(
        channel_id=channel.channel_id,
        predicate=lambda e: e.event_type == EV_CHANNEL_CLOSED,
        timeout=60.0,
    )
    print(f"closed: {close_env.event_data.get('reason')!r}")

    # Replay the conversation from the hub's write-ahead log.
    wal = await hub.read_wal(channel.channel_id)
    for env in wal:
        if env.event_type == EV_TEXT:
            speaker = "alice" if env.sender_id == alice.agent_id else "bob"
            print(f"{speaker}: {env.event_data['text']}")

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()

asyncio.run(main())
```

Expected output (Sonnet's exact words will differ on each run):

```text
closed: 'consulting_complete'
alice: What's the single most important property of a distributed system?
bob: Fault tolerance — because a system that can't survive partial failures defeats its entire purpose.
```

## What Just Happened

In order:

1. **`#!python Hub.open(MemoryKnowledgeStore())`** — boots an in-process hub. The `#!python KnowledgeStore` is where the hub persists its audit log, registry, and write-ahead logs (here in memory).
2. **`#!python LocalLink(hub)`** — a transport factory. Each `#!python HubClient` constructed against the same link gets its own duplex queue pair to the hub.
3. **`#!python HubClient(link, hub=hub)`** — one per process boundary. In a real deployment alice and bob would each live in their own process, each with one `#!python HubClient`. Here they share a process for clarity.
4. **`#!python hc.register(agent, passport, resume)`** — registers an `#!python Agent` with the hub. Returns an `#!python AgentClient` whose `#!python agent_id` is hub-stamped.
5. **`#!python alice.open(type="consulting", target="bob")`** — alice creates a consulting channel with bob as respondent. Internally: hub posts `#!python EV_CHANNEL_INVITE` to bob → bob's default handler auto-acks → hub posts `#!python EV_CHANNEL_OPENED` and `#!python alice.open(...)` returns with `#!python channel.state == ACTIVE`.
6. **`#!python channel.send(text, audience=...)`** — alice sends an `#!python EV_TEXT` envelope.
7. **Bob's default handler** — receives the `#!python EV_TEXT`, probes whether the adapter would accept a reply right now (it would — bob hasn't replied yet), runs `#!python Agent.ask(text)`, and sends bob's reply back through bob's own channel handle.
8. **`#!python ConsultingAdapter`** — sees both `initiator_sent` and `respondent_replied` are true, returns `#!python AdapterResult(next_state=CLOSED, auto_close_reason="consulting_complete")`. Hub posts `#!python EV_CHANNEL_CLOSED`.
9. **`#!python alice.wait_for_channel_event(...)`** — alice's loop wakes when she receives the close envelope.
10. **`#!python hub.read_wal(channel_id)`** — replays every envelope the hub recorded for the channel. Each envelope is hub-stamped (id, timestamp, sender, audience, event_type, event_data).

## Mental Hooks

- The `#!python Hub` is the **only authoritative state**. Every send goes through it; every observer reads from it. Clients are thin.
- A `#!python channel_id` is the unit of conversation. The hub's WAL is keyed by channel id; expectation evaluators evaluate per channel; views project per channel.
- Each `#!python AgentClient` carries a `#!python default_handler` that auto-acks invites and runs `#!python Agent.ask` on inbound text. You can replace it with `#!python agent_client.on_envelope(callback)` when you need custom logic.
- The hub assigns the `#!python agent_id` at registration. Use it (`#!python alice.agent_id`) for routing rather than the human-readable name. The name may not be unique under a multi-tenant deployment.

## Where to Next

- [Hub & Identity](hub_and_identity.md) — the registry side: `#!python Hub.open`, `#!python Passport`, `#!python Resume`, `#!python Rule`, auth.
- [Agent Clients](agent_clients.md) — the agent side: `#!python HubClient.register`, default handler, custom handlers.
- [Channel Adapters](adapters_overview.md) — pick the right one: free-form, 1Q1R, round-robin, or graph-driven.
