# Distributed network — one hub, agents on separate servers

A single `Hub` runs behind a WebSocket listener. Agents anywhere on the
network connect to it and register, open channels, and exchange messages
**entirely over the wire** — each agent process holds no in-process hub
reference, so this is a genuinely distributed deployment, not a loopback
simulation. Agents may be backed by different providers; the hub is
provider-neutral.

```
        ┌──────────── hub server (server A) ────────────┐
        │  Hub + serve_ws  ·  registry · channels · WAL  │
        └───▲───────────────────────▲──────────────────-┘
            │  ws://A:8765           │  ws://A:8765
   ┌────────┴────────┐      ┌────────┴────────┐
   │ initiator (B)   │      │ responder (C)   │
   │ HubClient(WsLink)│     │ HubClient(WsLink)│
   │ alice           │      │ bob             │
   └─────────────────┘      └─────────────────┘
```

## Files

| File | Role |
|------|------|
| `hub_server.py` | The hub. Runs `serve_ws` on a host/port, forever. |
| `responder.py`  | An agent node that registers and answers inbound consults. |
| `initiator.py`  | An agent node that opens a consult to a peer and prints the reply. |
| `run_demo.py`   | Spawns all three as separate local processes and verifies the round-trip. |
| `_common.py`    | `.env` loading + provider→config factory shared by the scripts. |

## Run it locally (three processes, one command)

```bash
python -m examples.network_distributed.run_demo
```

This spawns the hub + `bob` + `alice` as independent OS processes on
loopback, sends a question from `alice` to `bob` through the hub, and
prints `SUCCESS` with the three distinct PIDs once the answer returns.
Pick providers with env vars (needs the matching API keys):

```bash
DEMO_PROVIDER=openai python -m examples.network_distributed.run_demo
DEMO_INITIATOR_PROVIDER=anthropic DEMO_RESPONDER_PROVIDER=gemini \
    python -m examples.network_distributed.run_demo
```

## Run it across real servers

Identical scripts — only the URL changes. On the hub server:

```bash
python -m examples.network_distributed.hub_server --host 0.0.0.0 --port 8765
```

On each agent server (replace `HUB_IP`):

```bash
python -m examples.network_distributed.responder \
    --url ws://HUB_IP:8765 --name bob --provider openai

python -m examples.network_distributed.initiator \
    --url ws://HUB_IP:8765 --target bob --provider anthropic \
    --ask "What is 12 times 11? Reply with just the integer."
```

## Taking it to production

This example is deliberately minimal. For a real deployment add:

- **TLS** — pass an `ssl_context` to `serve_ws(...)` and use `wss://`
  URLs in `WsLink`.
- **Auth** — build the hub with
  `AuthRegistry([NoAuth(), ApiKeyAuth(keys=...)])` and have each agent
  register with `Passport(auth=AuthBlock(scheme="api_key", claim={"token": ...}))`.
- **Durability** — use `DiskKnowledgeStore(path)` instead of
  `MemoryKnowledgeStore` so the registry and channel WALs survive a hub
  restart.
- **Reconnect** — `WsLink.open()` connects once. Wrap a node in a small
  backoff loop that, on disconnect, builds a fresh `HubClient` and calls
  `attach(agent, name=..., since_envelope_id="")`; the hub replays any
  unacked deliveries past the agent's cursor, so the reconnect is
  exactly-handled.
