# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A responder node — an agent living on its own server.

Connects to a remote hub over ``WsLink`` (no in-process ``Hub``
reference — every call crosses the wire), registers an identity, and
then stays connected. Its default notify handler does the rest: it
auto-accepts channel invites and answers inbound questions with its
LLM. Nothing here drives a channel; this node only *responds*.

    python -m examples.network_distributed.responder \\
        --url ws://<hub-host>:8765 --name bob --provider openai
"""

import argparse
import asyncio
import contextlib

from autogen.beta import Agent
from autogen.beta.network import HubClient, Passport, Resume, WsLink

from ._common import load_env, make_config


async def main() -> None:
    parser = argparse.ArgumentParser(description="distributed-network responder node")
    parser.add_argument("--url", required=True, help="hub URL, e.g. ws://127.0.0.1:8765")
    parser.add_argument("--name", default="bob")
    parser.add_argument("--provider", default="anthropic", help="anthropic | openai | gemini")
    parser.add_argument("--capability", default="math", help="capability advertised in the resume")
    parser.add_argument(
        "--system-prompt",
        default="You are an arithmetic specialist. Reply with just the integer result, no words.",
    )
    args = parser.parse_args()
    load_env()

    agent = Agent(name=args.name, prompt=args.system_prompt, config=make_config(args.provider))

    # No hub= → remote mode: register, open, send, ack all travel as RPC
    # / notify frames over the WebSocket to the hub on the other server.
    hub_client = HubClient(WsLink(args.url))
    await hub_client.open()
    client = await hub_client.register(
        agent,
        Passport(name=args.name),
        Resume(claimed_capabilities=[args.capability], summary=f"{args.provider} {args.capability} specialist"),
    )
    print(f"[{args.name}] registered as {client.agent_id} (provider={args.provider}); awaiting channels", flush=True)

    try:
        await asyncio.Future()  # stay connected; the default handler answers
    finally:
        await hub_client.close()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
