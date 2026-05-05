# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""M1 foundation integration tests.

Covers the M1 exit criterion from ``design/PLAN.md``: two
``AgentClient``s register through ``LocalLink`` and exchange raw
envelopes; ``Hub.hydrate()`` rebuilds passport/resume/rule caches from
disk.

No sessions, no LLM, no expectation sweeper — those land in M2/M3.
"""

import asyncio

import pytest

from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from autogen.beta.network import (
    EV_TEXT,
    AccessBlock,
    AccessDeniedError,
    Envelope,
    Hub,
    HubClient,
    LimitsBlock,
    LocalLink,
    NotFoundError,
    Passport,
    Resume,
    Rule,
)


def _agent(name: str) -> Agent:
    return Agent(name=name, config=AnthropicConfig(model="claude-sonnet-4-6"))


@pytest.mark.asyncio
async def test_register_and_exchange_envelope_via_hub_client() -> None:
    """End-to-end M1 happy path: register both, send, receive."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    received: list[Envelope] = []

    async def on_envelope(env: Envelope) -> None:
        received.append(env)

    bob.on_envelope(on_envelope)

    envelope = Envelope(
        session_id="s1",
        sender_id=alice.agent_id,
        audience=[bob.agent_id],
        event_type=EV_TEXT,
        event_data={"text": "hello bob"},
    )
    eid = await alice.send_envelope(envelope)

    # Allow the demuxing loop to drain.
    for _ in range(20):
        if received:
            break
        await asyncio.sleep(0.01)

    assert len(received) == 1
    assert received[0].envelope_id == eid
    assert received[0].event_data == {"text": "hello bob"}

    # Hub WAL persisted.
    wal = await hub.read_wal("s1")
    assert len(wal) == 1
    assert wal[0].sender_id == alice.agent_id

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_unknown_sender_raises_not_found() -> None:
    """Posting an envelope from an unregistered sender_id raises NotFoundError."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store)

    envelope = Envelope(
        session_id="s1",
        sender_id="ghost",
        audience=["someone"],
        event_type=EV_TEXT,
        event_data={},
    )
    with pytest.raises(NotFoundError):
        await hub.post_envelope(envelope)

    await hub.close()


@pytest.mark.asyncio
async def test_hydrate_reloads_identities_from_disk(tmp_path) -> None:
    """Close hub, reopen with same store, verify identities + resumes survive."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store)
    link1 = LocalLink(hub1)
    hc1 = HubClient(link1, hub=hub1)

    alice = await hc1.register(
        _agent("alice"),
        Passport(name="alice", owner="acme"),
        Resume(claimed_capabilities=["analysis", "research"], summary="senior analyst"),
        skill_md="# Alice\nuse for research",
    )
    bob = await hc1.register(
        _agent("bob"),
        Passport(name="bob"),
        Resume(claimed_capabilities=["debate"]),
        rule=Rule(limits=LimitsBlock(session_ttl_default="4h")),
    )

    alice_id = alice.agent_id
    bob_id = bob.agent_id

    await hc1.close()
    await hub1.close()

    # Reopen — no shared in-memory state with hub1.
    store2 = DiskKnowledgeStore(str(tmp_path))
    hub2 = await Hub.open(store2)

    alice_p = await hub2.get_agent("alice")
    assert alice_p.agent_id == alice_id
    assert alice_p.owner == "acme"

    bob_p = await hub2.get_agent("bob")
    assert bob_p.agent_id == bob_id

    alice_resume = await hub2.get_resume(alice_id)
    assert "analysis" in alice_resume.claimed_capabilities
    assert alice_resume.summary == "senior analyst"

    alice_skill = await hub2.get_skill(alice_id)
    assert alice_skill is not None
    assert "Alice" in alice_skill

    # Capability filter on the rebuilt index.
    found = await hub2.list_agents(capability="debate")
    assert {p.name for p in found} == {"bob"}

    found_query = await hub2.list_agents(query="senior")
    assert {p.name for p in found_query} == {"alice"}

    await hub2.close()


@pytest.mark.asyncio
async def test_outbound_access_denied() -> None:
    """Sender's outbound_to whitelist blocks the recipient."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(
        _agent("alice"),
        Passport(name="alice"),
        Resume(),
        rule=Rule(access=AccessBlock(outbound_to=["carol"])),  # bob not allowed
    )
    bob = await hc.register(_agent("bob"), Passport(name="bob"), Resume())

    envelope = Envelope(
        session_id="s1",
        sender_id=alice.agent_id,
        audience=[bob.agent_id],
        event_type=EV_TEXT,
        event_data={},
    )
    with pytest.raises(AccessDeniedError):
        await alice.send_envelope(envelope)

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_inbound_access_silently_drops() -> None:
    """Recipient's inbound_from whitelist drops the notify silently."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await hc.register(
        _agent("bob"),
        Passport(name="bob"),
        Resume(),
        rule=Rule(access=AccessBlock(inbound_from=["carol"])),  # alice not allowed in
    )

    received: list[Envelope] = []
    bob.on_envelope(lambda env: received.append(env))  # type: ignore[arg-type, return-value]

    envelope = Envelope(
        session_id="s1",
        sender_id=alice.agent_id,
        audience=[bob.agent_id],
        event_type=EV_TEXT,
        event_data={},
    )
    eid = await alice.send_envelope(envelope)
    assert eid  # post succeeds (envelope is in WAL for audit)

    # Wait briefly to confirm bob's callback never fires.
    await asyncio.sleep(0.05)
    assert received == []

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_unregister_makes_send_fail() -> None:
    """After unregister, the AgentClient instance is inert.

    The local guard short-circuits before reaching the hub. If a
    different sender_id is used through ``Hub.post_envelope`` directly,
    the hub itself raises ``NotFoundError`` (already covered by
    ``test_unknown_sender_raises_not_found``).
    """
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await hc.register(_agent("bob"), Passport(name="bob"), Resume())

    await alice.unregister()

    envelope = Envelope(
        session_id="s1",
        sender_id=alice.agent_id,
        audience=[bob.agent_id],
        event_type=EV_TEXT,
        event_data={},
    )
    with pytest.raises(RuntimeError, match="disconnected"):
        await alice.send_envelope(envelope)

    # And the hub itself no longer knows the agent_id.
    with pytest.raises(NotFoundError):
        await hub.get_agent(alice.agent_id)
    remaining_names = {p.name for p in await hub.list_agents()}
    assert remaining_names == {"bob"}

    await hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_set_resume_updates_index() -> None:
    """Tenant-driven set_resume swaps the cached resume; queries reflect it."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store)
    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)

    alice = await hc.register(_agent("alice"), Passport(name="alice"), Resume())

    new_resume = Resume(claimed_capabilities=["new-skill"], summary="updated")
    await alice.set_resume(new_resume)

    found = await hc.list_agents(capability="new-skill")
    assert {p.name for p in found} == {"alice"}

    fetched = await hc.get_resume(alice.agent_id)
    assert fetched.summary == "updated"
    assert fetched.last_updated != ""

    await hc.close()
    await hub.close()
