# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``HubClient._receive_loop`` resilience to per-frame dispatch failures.

A single envelope whose handler raises must not stop the loop from
dispatching subsequent envelopes. The loop logs and keeps going.
"""

import asyncio

import pytest

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_TEXT,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)
from ag2.network.adapters.conversation import CONVERSATION_TYPE
from ag2.testing import TestConfig


def _agent(name: str) -> Agent:
    return Agent(name=name, config=TestConfig())


@pytest.mark.asyncio
async def test_handler_exception_does_not_stop_receive_loop() -> None:
    """A handler that raises on the first envelope must not block
    delivery of subsequent envelopes — per-frame dispatch failures
    are caught and logged, the loop survives.
    """
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    # Open the channel first under the default handler so bob auto-acks
    # the invite. Only then swap in the crashing handler so subsequent
    # text envelopes exercise the per-frame error path.
    channel = await alice.open(type=CONVERSATION_TYPE, target=bob.agent_id)

    async def always_raises(_envelope: Envelope) -> None:
        raise RuntimeError("boom")

    bob.on_envelope(always_raises)

    inbox = bob.ensure_channel_inbox(channel.channel_id)
    # Drain any channel-protocol envelopes already queued (invite/opened)
    # so the assertion sees only the two text envelopes we send below.
    while not inbox.empty():
        inbox.get_nowait()

    env1 = Envelope(
        channel_id=channel.channel_id,
        sender_id=alice.agent_id,
        audience=[bob.agent_id],
        event_type=EV_TEXT,
        event_data={"text": "first"},
    )
    env2 = Envelope(
        channel_id=channel.channel_id,
        sender_id=alice.agent_id,
        audience=[bob.agent_id],
        event_type=EV_TEXT,
        event_data={"text": "second"},
    )

    # Two sends in a row. Without per-frame error handling the first
    # send's handler crash would tear down bob's receive loop and the
    # second would never be delivered.
    await alice.send_envelope(env1)
    await alice.send_envelope(env2)

    seen: list[str] = []
    deadline = asyncio.get_event_loop().time() + 2.0
    while len(seen) < 2 and asyncio.get_event_loop().time() < deadline:
        env = await asyncio.wait_for(inbox.get(), timeout=1.0)
        if env.event_type == EV_TEXT:
            seen.append(env.event_data["text"])

    assert seen == ["first", "second"]

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()
