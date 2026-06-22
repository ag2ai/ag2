---
title: Agent Clients and Handlers
sidebarTitle: Agent Clients
---

The agent-side of the network. A `#!python HubClient` represents one process's connection to the hub; it produces `#!python AgentClient`s — one per registered `#!python Agent`. Each `#!python AgentClient` carries a notify handler that decides what the agent does when an envelope arrives.

!!! tip
    For a **non-LLM participant** — e.g. a human in the loop — register a `#!python HumanClient` instead via `#!python hc.register_human(...)`; see [HumanClient (HITL)](human_client.md).

## HubClient — one per process

```python linenums="1"
from autogen.beta.network import HubClient, LocalLink

link = LocalLink(hub)            # transport factory bound to this hub
hc = HubClient(link, hub=hub)
```

A `#!python HubClient` is the per-process registration broker. In a single-process script you may construct one per agent (each gets its own duplex link); in a real deployment you'd typically have one `#!python HubClient` per process and register all of that process's agents on it.

| Method | Notes |
|---|---|
| `#!python await hc.register(agent, passport, resume, ...)` | Registers an `#!python Agent` with the hub. Returns an `#!python AgentClient`. |
| `#!python await hc.unregister(agent_id)` | Tears down the registration; emits `#!python AUDIT_KIND_AGENT_UNREGISTERED`. |
| `#!python await hc.close()` | Closes this process's link to the hub. |
| `#!python hc.read_wal(channel_id)` | Read the WAL for any channel this process can see. |
| `#!python hc.can_send(channel_id, agent_id)` | Probe — would the adapter accept a send from this agent right now? |
| `#!python hc.default_view_policy(channel_id, agent_id)` | The view policy this participant should use when projecting history. |

The probe and view-policy methods exist so custom handlers (next section) don't have to reach into hub internals.

## Registering an Agent

```python linenums="1"
agent_client = await hc.register(
    agent,                     # autogen.beta.Agent
    passport,                  # autogen.beta.network.Passport
    resume,                    # autogen.beta.network.Resume
    skill_md=None,             # optional — markdown describing this agent's skill
    rule=None,                 # optional — Rule(...) for governance
    attach_plugin=True,        # whether to install the default notify handler
)
```

`#!python attach_plugin=True` (the default) installs the `#!python default_handler` on this client. Pass `#!python False` if you want full control over inbound envelope handling — typical for headless workers, gateways, or custom orchestration logic.

## AgentClient

| Attribute / Method | Notes |
|---|---|
| `#!python agent_client.agent_id` | Hub-stamped id; use for routing. |
| `#!python agent_client.agent` | The wrapped `#!python Agent`. |
| `#!python agent_client.passport` / `#!python .resume` | Snapshot at registration time. |
| `#!python await agent_client.open(type=..., target=..., knobs=...)` | Open a new channel. |
| `#!python await agent_client.send_envelope(envelope)` | Direct envelope send (for custom event types). |
| `#!python await agent_client.wait_for_channel_event(channel_id, predicate, timeout=...)` | Block until a matching envelope lands in this client's inbox. |
| `#!python agent_client.on_envelope(callback)` | Replace the handler. |

`#!python channel = await agent_client.open(...)` returns a `#!python Channel` handle scoped to one channel id. Use `#!python channel.send(text, audience=...)` for ordinary text sends, `#!python channel.close()` for explicit termination, `#!python channel.info()` for the latest metadata.

## The Default Handler

When you register with `#!python attach_plugin=True`, the client installs `#!python default_handler`. It routes inbound envelopes:

| Inbound event | Behaviour |
|---|---|
| `#!python EV_CHANNEL_INVITE` | Auto-ack with `#!python EV_CHANNEL_INVITE_ACK`. |
| `#!python EV_TEXT` / `#!python EV_HANDOFF` | If `#!python hc.can_send(...)` says it's our turn: read the WAL up to this envelope, project it through this participant's view policy, pre-populate a fresh `#!python MemoryStream`, attach a `#!python TaskMirror`, run `#!python agent.ask(text, stream=stream, dependencies=...)`, and send any non-empty reply via `#!python channel.send(...)`. |
| `#!python EV_CHANNEL_*` (other) | No-op — bookkeeping is reflected in the next `#!python channel.info()`. |
| `#!python ag2.task.*` | No-op at the handler level — `#!python TaskMirror` handles these separately when attached. |

The handler wraps the **entire** turn path (WAL slice, view projection, `#!python extract_turn_input`, `#!python agent.ask`, round-envelope build, outbound send) in a single trap: a crash is routed through `#!python HubClient.report_turn_failure` → `#!python Hub.report_turn_failure`, which fans `#!python on_turn_failed` out to every `#!python HubListener` (the built-in `#!python AuditLog` records `#!python AUDIT_KIND_TURN_FAILED`). No reply is posted, but the channel stays active and the next envelope flows normally — see [Turn-failure resilience](expectations_and_audit.md).

The handler is decomposed into public hooks so you can override only the parts you care about:

```python linenums="1"
from autogen.beta.network import (
    read_wal_until,
    resolve_view_policy,
    stamp_dependencies,
)
```

| Hook | Purpose |
|---|---|
| `#!python read_wal_until(client, envelope)` | Slice the WAL up to but excluding the given envelope. |
| `#!python resolve_view_policy(client, metadata)` | The `#!python ViewPolicy` this participant should use. |
| `#!python stamp_dependencies(client, channel)` | The `#!python context.dependencies` dict the LLM turn will see (`#!python CHANNEL_DEP`, `#!python AGENT_CLIENT_DEP`, `#!python HUB_DEP`). |

## Custom Handlers

```python linenums="1"
from autogen.beta.network import Envelope, EV_TEXT

async def gateway_handler(envelope: Envelope) -> None:
    if envelope.event_type != EV_TEXT:
        return
    # forward to your own external system instead of running an LLM
    text = envelope.event_data.get("text", "")
    await my_external_queue.put({"from": envelope.sender_id, "text": text})

agent_client.on_envelope(gateway_handler)
```

Common patterns:

- **Headless worker** — register with `#!python attach_plugin=False` and install a handler that pulls work directly off the hub without running an LLM.
- **Selective override** — install a handler that handles only one event type (e.g. custom invite policy) and falls back to `#!python default_handler` for everything else.
- **Filtered forwarding** — wrap `#!python default_handler` with pre/post hooks for logging, rate limiting, or routing.

## Transport — LocalLink

```python linenums="1"
from autogen.beta.network import LocalLink, LinkClient, LinkEndpoint

link = LocalLink(hub)
client_link = link.client()  # produces a fresh LocalLinkClient bound to a fresh LocalLinkEndpoint
```

`#!python LocalLink` is the in-process transport. Each `#!python HubClient(link, hub=hub)` lazily creates one `#!python LocalLinkClient`/`#!python LocalLinkEndpoint` pair on first use:

- `#!python LocalLinkClient` — the agent-process side; sends frames toward the hub, receives notify frames from it.
- `#!python LocalLinkEndpoint` — the hub-process side; the inverse.

Both sides exchange `#!python Frame` records via async queues. Frame types are re-exported at the package level: `#!python HelloFrame`, `#!python WelcomeFrame`, `#!python SendFrame`, `#!python ReceiptFrame`, `#!python NotifyFrame`, `#!python AcceptFrame`, `#!python ErrorFrame`, `#!python PingFrame`, `#!python PongFrame`, `#!python SubscribeFrame`, `#!python UnsubscribeFrame`.

The transport layer is a `#!python Protocol`:

```python
class LinkClient(Protocol):
    async def open(self) -> None: ...
    async def send_frame(self, frame: Frame) -> None: ...
    def frames(self) -> AsyncIterator[Frame]: ...
    async def close(self) -> None: ...
```

Cross-process or cross-host transports plug in here. `#!python LocalLink` is the only built-in, but the abstraction allows for future Redis/WebSocket/gRPC implementations without changing any client code.

## Inbox & Backpressure

Every `#!python AgentClient` has an inbox bounded by its `#!python Rule.inbox.max_pending` (default unbounded). When the inbox fills, sends to that agent fail with `#!python InboxFull`. The `#!python wait_for_channel_event` and the default handler drain the inbox in order; custom handlers should do the same — never block forever in a callback.

`#!python Rule.inbox.high_water` is a soft threshold below the hard cap: when a dispatch first pushes a recipient over it, the hub fires `#!python on_inbox_pressure(agent_id, pending, cap)` on every listener (and on a `#!python Hub` subclass override) — once per crossing, not on every subsequent envelope. It defaults to `#!python None`, which resolves to 80% of `#!python max_pending`; set `#!python 0` to disable the signal. See [HubListener](expectations_and_audit.md#hublistener-observing-state-transitions).

## Closing Down

```python linenums="1"
await alice_hc.close()
await bob_hc.close()
await hub.close()
```

`#!python HubClient.close()` cancels the link's listening task and unsubscribes all clients. Always pair with the matching `#!python register` calls; otherwise the hub keeps the registration in its registry.
