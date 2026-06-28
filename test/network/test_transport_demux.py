# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Transport / NotifyFrame demux invariants — multiple agents per
HubClient, broadcast vs targeted, multi-tenant isolation, unregister
cleanup.

The hub stamps ``recipient_id`` on every dispatched ``NotifyFrame``
so the receiving HubClient can demux directly to the right
AgentClient without re-walking channel participants. Tests here
validate that demux against the real frame path (LocalLink) for the
scenarios that are otherwise easy to get subtly wrong.
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

from ._helpers import ScriptedConfig


def _agent(name: str) -> Agent:
    return Agent(name=name, config=ScriptedConfig())


@pytest.mark.asyncio
async def test_multiple_agents_share_single_hub_client_endpoint() -> None:
    """A HubClient hosts N AgentClients on a single LocalLink connection.

    The hub binds every agent_id registered through the same HubClient
    to that HubClient's one endpoint. Demux happens at the AgentClient
    level via the per-frame ``recipient_id`` stamp.
    """
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await hc.register(_agent("bob"), Passport(name="bob"), Resume())
    carol = await hc.register(_agent("carol"), Passport(name="carol"), Resume())

    shared = hub._agent_to_endpoint[alice.agent_id]
    assert hub._agent_to_endpoint[bob.agent_id] == shared
    assert hub._agent_to_endpoint[carol.agent_id] == shared
    assert hub._endpoint_to_agents[shared] == {alice.agent_id, bob.agent_id, carol.agent_id}

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_targeted_envelope_delivered_to_only_named_recipient() -> None:
    """Per-frame ``recipient_id`` stamping: an envelope addressed to
    bob is delivered ONLY to bob, even when the same HubClient hosts
    multiple identities."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await hc.register(_agent("bob"), Passport(name="bob"), Resume())
    carol = await hc.register(_agent("carol"), Passport(name="carol"), Resume())

    # Wrap each agent's notify handler with a capture that preserves the original.
    received_alice: list[Envelope] = []
    received_bob: list[Envelope] = []
    received_carol: list[Envelope] = []
    orig_alice, orig_bob, orig_carol = alice._on_envelope, bob._on_envelope, carol._on_envelope

    async def cap_alice(env):
        received_alice.append(env)
        if orig_alice is not None:
            await orig_alice(env)

    async def cap_bob(env):
        received_bob.append(env)
        if orig_bob is not None:
            await orig_bob(env)

    async def cap_carol(env):
        received_carol.append(env)
        if orig_carol is not None:
            await orig_carol(env)

    alice.on_envelope(cap_alice)
    bob.on_envelope(cap_bob)
    carol.on_envelope(cap_carol)

    # Alice opens a 3-way discussion with bob+carol.
    channel = await alice.open(type="discussion", target=["bob", "carol"])

    # Targeted to bob only.
    await channel.send("private to bob", audience=[bob.agent_id])
    await asyncio.sleep(0.05)

    bob_text = [e for e in received_bob if e.event_type == EV_TEXT and e.event_data.get("text") == "private to bob"]
    carol_text = [e for e in received_carol if e.event_type == EV_TEXT and e.event_data.get("text") == "private to bob"]
    alice_text = [e for e in received_alice if e.event_type == EV_TEXT and e.event_data.get("text") == "private to bob"]

    assert len(bob_text) == 1, "bob must receive the targeted envelope"
    assert carol_text == [], "carol must NOT receive a targeted-to-bob envelope"
    assert alice_text == [], "sender must NOT echo its own send back to itself"

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_broadcast_envelope_delivered_to_all_non_sender() -> None:
    """An envelope with ``audience=None`` reaches every participant
    except the sender."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await hc.register(_agent("bob"), Passport(name="bob"), Resume())
    carol = await hc.register(_agent("carol"), Passport(name="carol"), Resume())

    received_bob: list[Envelope] = []
    received_carol: list[Envelope] = []
    received_alice: list[Envelope] = []
    orig_b, orig_c, orig_a = bob._on_envelope, carol._on_envelope, alice._on_envelope

    async def cap_b(env):
        received_bob.append(env)
        if orig_b:
            await orig_b(env)

    async def cap_c(env):
        received_carol.append(env)
        if orig_c:
            await orig_c(env)

    async def cap_a(env):
        received_alice.append(env)
        if orig_a:
            await orig_a(env)

    bob.on_envelope(cap_b)
    carol.on_envelope(cap_c)
    alice.on_envelope(cap_a)

    channel = await alice.open(type="discussion", target=["bob", "carol"])

    await channel.send("hello everyone", audience=None)
    await asyncio.sleep(0.05)

    # Both non-sender participants see it; alice (sender) does not.
    assert any(e.event_data.get("text") == "hello everyone" for e in received_bob)
    assert any(e.event_data.get("text") == "hello everyone" for e in received_carol)
    assert all(e.event_data.get("text") != "hello everyone" for e in received_alice)

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_unregister_cleans_endpoint_binding() -> None:
    """After unregister, the agent_id no longer maps to any endpoint."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await hc.register(_agent("bob"), Passport(name="bob"), Resume())
    alice_id = alice.agent_id
    shared_endpoint_id = hub._agent_to_endpoint[alice_id]

    # Both agents on same endpoint initially.
    assert alice_id in hub._endpoint_to_agents[shared_endpoint_id]
    assert bob.agent_id in hub._endpoint_to_agents[shared_endpoint_id]

    await alice.unregister()

    # Alice's binding gone; bob still bound to the same endpoint.
    assert alice_id not in hub._agent_to_endpoint
    assert alice_id not in hub._endpoint_to_agents.get(shared_endpoint_id, set())
    assert bob.agent_id in hub._endpoint_to_agents[shared_endpoint_id]

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_two_hub_clients_share_one_hub_isolated_endpoints() -> None:
    """Two HubClients (e.g. simulating two tenant processes) on the
    same in-process Hub get separate endpoints. Envelopes for tenant
    A's agents do NOT reach tenant B."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    hc_a = HubClient(link, hub=hub)
    hc_b = HubClient(link, hub=hub)

    alice = await hc_a.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await hc_b.register(_agent("bob"), Passport(name="bob"), Resume())

    # Different endpoints.
    assert hub._agent_to_endpoint[alice.agent_id] != hub._agent_to_endpoint[bob.agent_id]

    received_bob: list[Envelope] = []
    orig_b = bob._on_envelope

    async def cap_b(env):
        received_bob.append(env)
        if orig_b:
            await orig_b(env)

    bob.on_envelope(cap_b)

    channel = await alice.open(type="conversation", target="bob")
    await channel.send("cross-tenant", audience=[bob.agent_id])
    await asyncio.sleep(0.05)

    assert any(e.event_data.get("text") == "cross-tenant" for e in received_bob)

    # And the cross-tenant envelope is registered in alice's HubClient
    # too — but only routed to her own AgentClient (none here besides
    # alice). Verify by checking that hc_a only has alice in _clients.
    assert set(hc_a._clients.keys()) == {alice.agent_id}
    assert set(hc_b._clients.keys()) == {bob.agent_id}

    await hc_a.close()
    await hc_b.close()
    await hub.close()


@pytest.mark.asyncio
async def test_hub_client_close_shuts_down_link() -> None:
    """HubClient.close() should close its LocalLinkClient, terminating
    the receive loop. Subsequent register attempts must raise."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await hc.close()

    with pytest.raises(RuntimeError, match="closed"):
        await hc.register(_agent("bob"), Passport(name="bob"), Resume())

    await hub.close()


@pytest.mark.asyncio
async def test_hub_close_idempotent_after_endpoint_attached() -> None:
    """Hub.close() with attached endpoint tasks — calling twice is safe."""
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await hc.close()
    await hub.close()
    await hub.close()  # idempotent
