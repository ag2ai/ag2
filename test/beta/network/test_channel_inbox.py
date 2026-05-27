# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Per-channel inbox availability invariant.

Covers the rule that any ``AgentClient`` involved in a channel must
have a per-channel inbox for it. The two creation paths under test:

* ``AgentClient.open(...)`` pre-creates the inbox for the creator
  immediately after the hub returns the channel metadata.
* ``AgentClient.receive(...)`` ensures the inbox exists before queuing
  the inbound envelope, so any joiner gets one on first envelope
  regardless of handler shape (default, custom, or no handler at all).

The race-fix end-to-end check confirms a sleep between ``channel.send``
and ``wait_for_channel_event`` no longer drops envelopes — replies and
the close envelope are queued during the sleep and consumed by the
later wait.
"""

import asyncio
import contextlib

import pytest

from autogen.beta import Agent
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    EV_CHANNEL_CLOSED,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)
from autogen.beta.network.adapters.consulting import CONSULTING_TYPE
from autogen.beta.testing import TestConfig


def _agent(name: str, *events: object) -> Agent:
    return Agent(name=name, config=TestConfig(*events))


@pytest.mark.asyncio
class TestChannelInboxInvariant:
    """Every AgentClient involved in a channel always has an inbox for it."""

    async def test_creator_has_inbox_after_open(self) -> None:
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

        channel = await alice.open(type=CONSULTING_TYPE, target="bob")

        assert channel.channel_id in alice._channel_inboxes

        await alice_hc.close()
        await bob_hc.close()
        await hub.close()

    async def test_default_handler_joiner_has_inbox(self) -> None:
        """Joiner with the default handler — invite triggers receive(),
        which creates the inbox before _auto_ack_invite runs."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

        channel = await alice.open(type=CONSULTING_TYPE, target="bob")

        assert channel.channel_id in bob._channel_inboxes

        await alice_hc.close()
        await bob_hc.close()
        await hub.close()

    async def test_custom_handler_joiner_has_inbox(self) -> None:
        """Joiner with a custom handler that ignores invites still gets
        an inbox — the receive()-level hook fires before any handler.

        Regression case for the head-dev invariant: the older
        ``_auto_ack_invite``-only hook would have left this joiner
        without an inbox.
        """
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())

        async def silent(_envelope: Envelope) -> None:
            return

        bob = await bob_hc.register(
            _agent("bob"),
            Passport(name="bob"),
            Resume(),
            attach_plugin=False,
        )
        bob.on_envelope(silent)

        # Without an ack the channel won't go ACTIVE; open() blocks until the
        # invite-ack timeout fires (or we cancel). The invite envelope
        # itself reaches bob.receive() before that, which is the point.
        open_task = asyncio.create_task(alice.open(type=CONSULTING_TYPE, target=bob.agent_id))
        await asyncio.sleep(0.1)

        channels = list(hub._channels.values())
        assert len(channels) == 1
        channel_id = channels[0].channel_id

        assert channel_id in bob._channel_inboxes

        open_task.cancel()
        with contextlib.suppress(BaseException):
            await open_task

        await alice_hc.close()
        await bob_hc.close()
        await hub.close()

    async def test_ensure_channel_inbox_is_idempotent(self) -> None:
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)
        link = LocalLink(hub)
        hc = HubClient(link, hub=hub)
        ac = await hc.register(_agent("solo"), Passport(name="solo"), Resume())

        sid = "fake-channel"
        q1 = ac.ensure_channel_inbox(sid)
        q2 = ac.ensure_channel_inbox(sid)
        assert q1 is q2

        await hc.close()
        await hub.close()

    async def test_send_sleep_wait_no_race(self) -> None:
        """The race fix end-to-end: open → send → asyncio.sleep → wait
        succeeds. Without the inbox invariant, the close envelope would
        arrive during the sleep, find no inbox, and be dropped — the
        wait would then time out."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)
        bob_hc = HubClient(link, hub=hub)
        alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
        bob = await bob_hc.register(
            _agent("bob", "ok"),  # bob's TestConfig replies with "ok"
            Passport(name="bob"),
            Resume(),
        )

        channel = await alice.open(type=CONSULTING_TYPE, target=bob.agent_id)
        await channel.send("hi", audience=[bob.agent_id])

        # During this sleep: bob's handler runs, replies, ConsultingAdapter
        # auto-closes, hub posts EV_CHANNEL_CLOSED. All envelopes must
        # land in alice's pre-existing inbox.
        await asyncio.sleep(0.2)

        close_env = await alice.wait_for_channel_event(
            channel_id=channel.channel_id,
            predicate=lambda e: e.event_type == EV_CHANNEL_CLOSED,
            timeout=2.0,
        )
        assert close_env.event_data.get("reason") == "consulting_complete"

        await alice_hc.close()
        await bob_hc.close()
        await hub.close()
