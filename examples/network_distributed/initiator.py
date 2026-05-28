# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""An initiator node — opens a channel to a peer across the network.

Connects to the remote hub, registers, opens a ``consulting`` channel
addressed to a peer by name (resolved over the wire), asks one
question, waits for the peer's reply to come back across the network,
prints it, and exits.

    python -m examples.network_distributed.initiator \\
        --url ws://<hub-host>:8765 --target bob \\
        --ask "What is 12 times 11? Reply with just the integer."
"""

import argparse
import asyncio

from autogen.beta import Agent
from autogen.beta.network import EV_TEXT, HubClient, Passport, Resume, WsLink
from autogen.beta.network.adapters.consulting import CONSULTING_TYPE

from ._common import load_env, make_config


async def main() -> None:
    parser = argparse.ArgumentParser(description="distributed-network initiator node")
    parser.add_argument("--url", required=True, help="hub URL, e.g. ws://127.0.0.1:8765")
    parser.add_argument("--name", default="alice")
    parser.add_argument("--provider", default="anthropic", help="anthropic | openai | gemini")
    parser.add_argument("--target", default="bob", help="peer name to consult")
    parser.add_argument("--ask", required=True, help="the question to send")
    args = parser.parse_args()
    load_env()

    agent = Agent(name=args.name, prompt="You are a coordinator.", config=make_config(args.provider))

    hub_client = HubClient(WsLink(args.url))
    await hub_client.open()
    alice = await hub_client.register(agent, Passport(name=args.name), Resume())
    print(f"[{args.name}] connected (provider={args.provider}); consulting {args.target!r}", flush=True)

    # Open the channel and ask — both cross the wire to the hub, which
    # invites the (remote, possibly different-provider) peer.
    channel = await alice.open(type=CONSULTING_TYPE, target=[args.target], intent="cross-server consult")
    await channel.send(args.ask)

    # Wait for the peer's reply to arrive back over the network.
    reply = await alice.wait_for_channel_event(
        channel_id=channel.channel_id,
        predicate=lambda e: e.event_type == EV_TEXT and e.sender_id != alice.agent_id,
        timeout=90.0,
    )
    print(f"[{args.name}] reply from {args.target!r}: {reply.event_data.get('text', '')}", flush=True)
    print("RESULT-OK", flush=True)

    await hub_client.close()


if __name__ == "__main__":
    asyncio.run(main())
