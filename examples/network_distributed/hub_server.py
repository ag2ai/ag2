# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The hub process — one server, many remote agents.

Runs a single ``Hub`` behind a WebSocket listener. Agents anywhere on
the network connect to ``ws://<this-host>:<port>`` and register / open
channels / send entirely over the wire; the hub holds no reference to
them beyond their connections. This is the central node of the
distributed network.

Run it on its own (a real server), then point agents at it::

    python -m examples.network_distributed.hub_server --host 0.0.0.0 --port 8765

``--port 0`` binds an ephemeral port and prints the chosen one — used by
``run_demo.py`` to wire everything together on one machine.

Production notes (kept out of this minimal example): pass an
``ssl_context`` to :func:`serve_ws` for ``wss://``, and construct the
hub with an ``AuthRegistry([... ApiKeyAuth(...)])`` so connecting agents
authenticate at the handshake. Swap ``MemoryKnowledgeStore`` for
``DiskKnowledgeStore(path)`` to make the registry + WAL survive a hub
restart.
"""

import argparse
import asyncio
import contextlib

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import Hub, serve_ws


async def main() -> None:
    parser = argparse.ArgumentParser(description="distributed-network hub server")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to accept remote agents)")
    parser.add_argument("--port", type=int, default=8765, help="bind port (0 = ephemeral)")
    args = parser.parse_args()

    hub = await Hub.open(MemoryKnowledgeStore())
    async with serve_ws(hub, args.host, args.port) as server:
        host, port = server.sockets[0].getsockname()[:2]
        # Single sentinel line so an orchestrator can read the real port.
        print(f"[hub] listening on ws://{host}:{port}", flush=True)
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.Future()  # serve until interrupted
    await hub.close()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
