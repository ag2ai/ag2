# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the adapter-owned tool surface + Layer-2 envelope helpers.

* ``ChannelAdapter.tools_for`` resolves per-turn.
* ``NetworkPlugin`` attaches only identity-level tools (no ``say``).
* Workflow agents never see ``say``.
* Adapter envelope helpers produce envelopes that round-trip through
  ``Hub.post_envelope``.
* ``channels(action="open", message=...)`` seeds the first envelope.
"""

import json

import pytest

from autogen.beta import Agent
from autogen.beta.events import ToolCallEvent
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    EV_PACKET,
    EV_TEXT,
    Handoff,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)
from autogen.beta.network.adapters.consulting import ConsultingAdapter
from autogen.beta.network.adapters.conversation import ConversationAdapter
from autogen.beta.network.adapters.discussion import DiscussionAdapter
from autogen.beta.network.adapters.workflow import WorkflowAdapter
from autogen.beta.testing import TestConfig


def _agent(name: str, *replies: str) -> Agent:
    return Agent(name=name, config=TestConfig(*replies))


# ── Plugin attaches identity-level tools only ─────────────────────────────


@pytest.mark.asyncio
async def test_plugin_does_not_attach_say() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    names = {t.name for t in alice.agent.tools}
    assert "say" not in names
    # Identity-level set still present.
    assert {"delegate", "peers", "channels", "tasks", "context"} <= names

    await alice_hc.close()
    await hub.close()


# ── Adapter tools_for ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_consulting_tools_for_gates_by_turn() -> None:
    """Consulting offers ``say`` to the participant whose turn it is."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())
    channel = await alice.open(type="consulting", target="bob")

    adapter = ConsultingAdapter()
    state = hub.adapter_state(channel.channel_id)
    # Initiator's turn (hasn't sent the prompt yet).
    initiator_tools = adapter.tools_for(alice, channel.metadata, state, alice.agent_id)
    assert [t.name for t in initiator_tools] == ["say"]
    # Respondent's turn isn't active yet.
    respondent_tools = adapter.tools_for(bob, channel.metadata, state, bob.agent_id)
    assert respondent_tools == []

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_conversation_tools_for_always_offers_say() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())
    channel = await alice.open(type="conversation", target="bob")

    adapter = ConversationAdapter()
    state = hub.adapter_state(channel.channel_id)
    assert [t.name for t in adapter.tools_for(alice, channel.metadata, state, alice.agent_id)] == ["say"]
    assert [t.name for t in adapter.tools_for(bob, channel.metadata, state, bob.agent_id)] == ["say"]

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_discussion_tools_for_gates_by_round_robin() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    carol_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())
    carol = await carol_hc.register(_agent("carol"), Passport(name="carol"), Resume())

    channel = await alice.open(
        type="discussion",
        target=["bob", "carol"],
        knobs={"ordering": "round_robin"},
    )

    adapter = DiscussionAdapter()
    state = hub.adapter_state(channel.channel_id)
    # Initial expected_next_speaker is the creator (alice).
    assert [t.name for t in adapter.tools_for(alice, channel.metadata, state, alice.agent_id)] == ["say"]
    assert adapter.tools_for(bob, channel.metadata, state, bob.agent_id) == []
    assert adapter.tools_for(carol, channel.metadata, state, carol.agent_id) == []

    await alice_hc.close()
    await bob_hc.close()
    await carol_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_workflow_tools_for_returns_empty() -> None:
    """Workflow ships no adapter tools — handoff is user-authored."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())

    adapter = WorkflowAdapter()
    # No active workflow channel needed — tools_for is pure.
    assert adapter.tools_for(alice, None, None, alice.agent_id) == []

    await alice_hc.close()
    await hub.close()


# ── Layer-2 envelope helpers ──────────────────────────────────────────────


def test_build_text_envelope_default_shape() -> None:
    adapter = ConversationAdapter()
    env = adapter.build_text_envelope(
        channel_id="c1",
        sender_id="a1",
        text="hello",
        audience=["a2"],
    )
    assert env.event_type == EV_TEXT
    assert env.event_data == {"text": "hello"}
    assert env.audience == ["a2"]


def test_build_packet_envelope_with_handoff() -> None:
    adapter = WorkflowAdapter()
    env = adapter.build_packet_envelope(
        channel_id="c1",
        sender_id="a1",
        body="result",
        handoff=Handoff(target="a2", reason="needs expert"),
    )
    assert env.event_type == EV_PACKET
    assert env.event_data["body"] == "result"
    routing = env.event_data.get("routing", {})
    assert routing == {"kind": "handoff", "target": "a2", "reason": "needs expert"}


def test_build_packet_envelope_with_context_set() -> None:
    adapter = WorkflowAdapter()
    env = adapter.build_packet_envelope(
        channel_id="c1",
        sender_id="a1",
        body="step done",
        context_set={"score": 0.9},
    )
    assert env.event_type == EV_PACKET
    assert env.event_data["context"] == {"score": 0.9}


# ── channels(open, message=...) seed ──────────────────────────────────────


@pytest.mark.asyncio
async def test_channels_open_with_seed_message() -> None:
    """``channels(action="open", message=...)`` posts the seed envelope
    atomically after the channel transitions to OPENED.

    Drives the tool via the agent's `ask` so the test exercises the
    real fast_depends injection path; the agent's scripted reply
    returns a tool call to ``channels(open, message=...)``.
    """
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice_agent = Agent(
        name="alice",
        config=TestConfig(
            [
                ToolCallEvent(
                    name="channels",
                    arguments=json.dumps({
                        "action": "open",
                        "type": "conversation",
                        "target": "bob",
                        "message": "kickoff",
                    }),
                )
            ],
            "done",
        ),
    )
    alice = await alice_hc.register(alice_agent, Passport(name="alice"), Resume())
    await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    reply = await alice.agent.ask("Open a conversation with bob.")
    assert reply.body

    # Find the new channel (the only conversation in the hub).
    channels = await hub.list_channels()
    convs = [c for c in channels if c.manifest.type == "conversation"]
    assert len(convs) == 1
    wal = await hub.read_wal(convs[0].channel_id)
    text_envs = [e for e in wal if e.event_type == EV_TEXT]
    # Seed envelope present alongside any default-handler-generated turns.
    assert any(e.event_data.get("text") == "kickoff" for e in text_envs)

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()
