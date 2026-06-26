# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Foundation integration tests.

Two ``AgentClient``s register through ``LocalLink`` and exchange raw
envelopes; ``Hub.hydrate()`` rebuilds passport/resume/rule caches from
disk. No channels, no LLM, no expectation sweeper — those are
exercised in the higher-level adapter tests.
"""

import pytest

from ag2 import Agent
from ag2.config import AnthropicConfig
from ag2.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from ag2.network import (
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


# Channel-based envelope round-trip lives in test_consulting.py — once
# adapters are wired all envelope dispatch goes through a real channel.


@pytest.mark.asyncio
async def test_unknown_sender_raises_not_found() -> None:
    """Posting an envelope from an unregistered sender_id raises NotFoundError."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store)

    envelope = Envelope(
        channel_id="s1",
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

    alice = await hub1.register(
        _agent("alice"),
        Passport(name="alice", owner="acme"),
        resume=Resume(claimed_capabilities=["analysis", "research"], summary="senior analyst"),
        skill_md="# Alice\nuse for research",
    )
    bob = await hub1.register(
        _agent("bob"),
        resume=Resume(claimed_capabilities=["debate"]),
        rule=Rule(limits=LimitsBlock(channel_ttl_default="4h")),
    )

    alice_id = alice.agent_id
    bob_id = bob.agent_id

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

    alice = await hub.register(
        _agent("alice"),
        rule=Rule(access=AccessBlock(outbound_to=["carol"])),  # bob not allowed
    )
    bob = await hub.register(_agent("bob"))

    envelope = Envelope(
        channel_id="s1",
        sender_id=alice.agent_id,
        audience=[bob.agent_id],
        event_type=EV_TEXT,
        event_data={},
    )
    with pytest.raises(AccessDeniedError):
        await alice.send_envelope(envelope)

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

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    await alice.unregister()

    envelope = Envelope(
        channel_id="s1",
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
