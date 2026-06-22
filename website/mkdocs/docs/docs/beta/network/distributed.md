---
title: Distributed Deployment
sidebarTitle: Distributed Deployment
---

The network is not limited to one process. Replace `#!python LocalLink` with `#!python WsLink` and the hub becomes a server that agents anywhere on the network can connect to over WebSocket — entirely over the wire, no shared memory, no in-process hub reference.

## Install

`#!python WsLink` and `#!python serve_ws` require the `network-ws` extra:

```bash
pip install "ag2[network-ws]"
```

## Architecture

```
        ┌─────────────── hub server ────────────────┐
        │  Hub + serve_ws  ·  registry · WAL · auth │
        └───▲──────────────────────────▲────────────┘
            │  ws://hub:8765           │  ws://hub:8765
   ┌────────┴──────────┐     ┌─────────┴──────────┐
   │  process A        │     │  process B         │
   │  HubClient(WsLink)│     │  HubClient(WsLink) │
   │  alice            │     │  bob               │
   └───────────────────┘     └────────────────────┘
```

Every control-plane call (register, open channel, post envelope, WAL read) travels as a `#!python RequestFrame` / `#!python ResponseFrame` RPC over the WebSocket. Every inbound notify arrives as a `#!python NotifyFrame` and is ack'd by a `#!python ReceiptFrame`. The `#!python HubClient` API is identical whether you pass a `#!python LocalLink` or a `#!python WsLink` — the transport is the only thing that changes.

## Starting the Hub Server

```python linenums="1"
import asyncio
import contextlib

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import Hub, serve_ws

async def main() -> None:
    hub = await Hub.open(MemoryKnowledgeStore())
    async with serve_ws(hub, "0.0.0.0", 8765) as server:
        host, port = server.sockets[0].getsockname()[:2]
        print(f"listening on ws://{host}:{port}", flush=True)
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.Future()  # serve until interrupted
    await hub.close()

asyncio.run(main())
```

`#!python serve_ws(hub, host, port)` is an async context manager. It binds a WebSocket server, hands each incoming connection its own `#!python WsLinkEndpoint`, and lets the hub dispatch from there. Pass `#!python port=0` to bind an ephemeral port and read the real one from `#!python server.sockets[0].getsockname()[1]`.

## Connecting a Remote Agent

From any other process — same machine, different container, or across a real network:

```python linenums="1"
import asyncio

from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.network import HubClient, Passport, Resume, WsLink

async def main() -> None:
    hub_client = HubClient(WsLink("ws://hub-host:8765"))
    await hub_client.open()  # WebSocket connect + handshake

    agent = Agent("bob", prompt="Answer in one short sentence.", config=AnthropicConfig(model="claude-haiku-4-5"))
    bob = await hub_client.register(agent, Passport(name="bob"), Resume())

    print(f"registered as {bob.agent_id}; awaiting channels")
    await asyncio.Future()  # stay connected; default handler answers inbound consults

asyncio.run(main())
```

`#!python HubClient(WsLink(url))` puts the client in *remote mode*: every `#!python register`, `#!python open`, `#!python send`, and `#!python read_wal` call is an RPC round-trip to the hub. The `#!python AgentClient` surface (`#!python bob.open(...)`, `#!python bob.wait_for_channel_event(...)`, `#!python channel.send(...)`) is identical to the in-process API.

Agents backed by different providers can share the same hub and the same channel — the hub is provider-neutral.

## Sending a Cross-Process Consult

```python linenums="1"
import asyncio

from autogen.beta import Agent
from autogen.beta.config import OpenAIConfig
from autogen.beta.network import EV_TEXT, HubClient, Passport, Resume, WsLink
from autogen.beta.network.adapters.consulting import CONSULTING_TYPE

async def main() -> None:
    hub_client = HubClient(WsLink("ws://hub-host:8765"))
    await hub_client.open()

    alice = await hub_client.register(
        Agent("alice", prompt="You are a coordinator.", config=OpenAIConfig(model="gpt-4o-mini")),
        Passport(name="alice"),
        Resume(),
    )

    # open() and send() travel to the hub over the wire;
    # the hub invites bob over bob's own WebSocket connection.
    channel = await alice.open(type=CONSULTING_TYPE, target=["bob"])
    await channel.send("What is 12 times 11? Reply with just the integer.")

    reply = await alice.wait_for_channel_event(
        channel_id=channel.channel_id,
        predicate=lambda e: e.event_type == EV_TEXT and e.sender_id != alice.agent_id,
        timeout=90.0,
    )
    print(reply.event_data["text"])  # 132
    await hub_client.close()

asyncio.run(main())
```

The channel invite travels hub → bob over bob's WebSocket. Bob's default handler answers, and the reply comes back hub → alice over alice's WebSocket. The hub brokers the exchange; neither agent holds a reference to the other.

All four channel adapters work cross-process without change: `#!python consulting`, `#!python conversation`, `#!python discussion`, and `#!python workflow`.

## Authentication

For deployments where agents must authenticate at the WebSocket handshake, build the hub with an `#!python AuthRegistry`:

```python linenums="1"
from autogen.beta.network import ApiKeyAuth, AuthRegistry, Hub

auth = AuthRegistry([ApiKeyAuth(keys={"token-alice", "token-bob"})])
hub = await Hub.open(store, auth=auth)
```

Each agent passes its token inside `#!python Passport.auth`:

```python linenums="1"
from autogen.beta.network import AuthBlock, Passport

passport = Passport(
    name="bob",
    auth=AuthBlock(scheme="api_key", claim={"token": "token-bob"}),
)
bob = await hub_client.register(agent, passport, Resume())
```

The hub's `#!python AuthRegistry` validates the claim before binding the connection. Raise `#!python AuthError` from a custom `#!python AuthAdapter` (a `#!python Protocol`) to reject any scheme you define.

## At-Least-Once Delivery

The hub guarantees each envelope is delivered at least once across reconnects:

- Each `#!python AgentClient` maintains an inbox cursor per channel — the `envelope_id` of the last successfully processed envelope.
- Inbound envelopes are ack'd with `#!python ReceiptFrame`; a nack causes immediate replay.
- On reconnect, pass `#!python since_envelope_id` to replay any unacked envelopes from that point:

```python linenums="1"
hub_client = HubClient(WsLink(url))
await hub_client.open()
bob = await hub_client.attach(agent, name="bob", since_envelope_id=last_acked_id)
await hub_client.resume_pending_turns(bob)
```

`#!python attach` re-binds an existing identity to a fresh connection. `#!python resume_pending_turns` re-fires any turns the protocol still expects from this agent. The default notify handler is idempotent under redelivery — causation-id deduplication short-circuits duplicate model turns without double-posting.

## Task Durability

Tasks can be checkpointed through the hub so state survives a process restart:

```python linenums="1"
from autogen.beta.network import HubBackedCheckpointStore

checkpoint_store = HubBackedCheckpointStore(hub_client)

# inside a tool or agent turn:
await task.checkpoint({"step": 3, "partial_result": "..."})

# on another node, after a restart:
recovered = await agent.resume_from(task_id, checkpoint_store)
```

`#!python HubBackedCheckpointStore` satisfies the `#!python CheckpointStore` Protocol by delegating writes and reads to the hub's `#!python KnowledgeStore`. Pass a `#!python Hub` for in-process durability or a `#!python HubClient` for cross-process. For checkpoints that survive a hub restart, use `#!python DiskKnowledgeStore(path)` on the hub.

## Production Notes

| Concern | Recommendation |
|---|---|
| TLS | Pass an `ssl_context` to `serve_ws(...)` and use `wss://` in `WsLink`. |
| Auth | Build the hub with `AuthRegistry([ApiKeyAuth(keys=...)])` and pass `Passport(auth=AuthBlock(...))` from each agent. |
| Durability | Use `DiskKnowledgeStore(path)` on the hub so the registry and channel WALs survive a restart. |
| Reconnect | On disconnect, build a fresh `HubClient`, call `open()`, then `attach(agent, name=..., since_envelope_id=last_id)`. The hub replays any unacked envelopes past that cursor. |
| Federation | Register a `RemoteAgentProxy` on the hub to route envelopes addressed to agents with `kind="remote_agent"` across hub boundaries. |

## Where to Next

- [Hub & Identity](hub_and_identity.md) — `AuthRegistry`, `ApiKeyAuth`, governance rules.
- [Agent Clients](agent_clients.md) — custom envelope handlers, replacing the default handler.
- [Governance, Audit & Observability](expectations_and_audit.md) — per-channel expectations and the hub audit log.
