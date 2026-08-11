# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Real-LLM coverage for network surface that the existing smokes miss.

The other smokes cover the headline flows (consulting via delegate,
round-robin discussion, single-handoff workflow). These tests cover
the remaining tool surface against a real model so we have evidence
the LLM-driven paths actually work end-to-end:

* ``ConversationAdapter`` — 1+1 bidirectional, multi-turn, manual close.
* ``peers(action="describe")`` — SKILL.md fallback rendering reaches the LLM.
* ``channels(action="close")`` — LLM closes the channel it owns.
* ``context(action="search")`` — LLM finds an earlier turn by substring.
* Multi-tool workflow graph — graph with two ``ToolCalled`` transitions
  materialises two distinct handoff tools and the LLM picks the right one.

Uses ``claude-haiku-4-5`` for cost.
"""

import asyncio
import os
from pathlib import Path

import pytest

from ag2 import Agent
from ag2.config import AnthropicConfig
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_PACKET,
    EV_TEXT,
    Handoff,
    Hub,
    Resume,
)
from ag2.network.adapters.conversation import CONVERSATION_TYPE
from ag2.network.adapters.workflow import WORKFLOW_TYPE
from ag2.network.channel import ChannelState
from ag2.network.client.channel import Channel
from ag2.network.policies import (
    AGENT_CLIENT_DEP,
    CHANNEL_DEP,
    HUB_DEP,
)
from ag2.network.transitions import (
    AgentTarget,
    FromSpeaker,
    StayTarget,
    ToolCalled,
    Transition,
    TransitionGraph,
)
from ag2.testing import TestConfig

try:
    from dotenv import load_dotenv

    _REPO_ROOT = Path(__file__).resolve().parents[4]
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass


@pytest.fixture()
def anthropic_config() -> AnthropicConfig:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return AnthropicConfig(model="claude-haiku-4-5", api_key=api_key, temperature=0)


async def _wait_for_text_count(
    hub: Hub,
    channel_id: str,
    expected: int,
    *,
    timeout: float = 60.0,
) -> int:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        wal = await hub.read_wal(channel_id)
        count = sum(1 for e in wal if e.event_type == EV_TEXT)
        if count >= expected:
            return count
        await asyncio.sleep(0.2)
    return sum(1 for e in (await hub.read_wal(channel_id)) if e.event_type == EV_TEXT)


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_conversation_adapter_bidirectional_two_turns(
    anthropic_config: AnthropicConfig,
) -> None:
    """The ``ConversationAdapter`` runs 1+1 indefinitely until close.

    alice opens a conversation with bob, sends a question, bob's notify
    handler engages bob's LLM and replies. The default adapter does NOT
    auto-close (unlike consulting), so the channel stays ACTIVE. We
    verify both turns landed and the channel is still active.
    """
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )

    alice = Agent(
        name="alice",
        prompt="You are alice. Respond in one short sentence.",
        config=anthropic_config,
    )
    bob = Agent(
        name="bob",
        prompt="You are bob. Respond in one short sentence.",
        config=anthropic_config,
    )

    alice_c = await hub.register(alice)
    bob_c = await hub.register(bob)

    channel = await alice_c.open(type=CONVERSATION_TYPE, target=bob_c.agent_id)

    await channel.send("Hi bob. What's a good Python web framework for tiny APIs?")

    # Wait for bob's reply to land.
    count = await _wait_for_text_count(hub, channel.channel_id, expected=2, timeout=60.0)
    assert count >= 2, f"expected 2 turns (alice + bob), got {count}"

    # Conversation is still active — no auto-close.
    metadata = await hub.get_channel(channel.channel_id)
    assert metadata.state == ChannelState.ACTIVE

    await channel.close()
    metadata = await hub.get_channel(channel.channel_id)
    assert metadata.state == ChannelState.CLOSED

    await hub.close()


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_peers_describe_returns_fallback_skill(
    anthropic_config: AnthropicConfig,
) -> None:
    """``peers(action="describe")`` returns a fallback skill_md when no
    SKILL.md is registered, and the LLM can extract a fact from it.

    bob registers with a passport + resume only (no SKILL.md). alice's
    LLM calls ``peers(action="describe", name="bob")`` and is asked to
    repeat bob's claimed capability. The fallback render must contain
    enough structure for the LLM to pull "math" out.
    """
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    alice = Agent(
        name="alice",
        prompt=(
            "You are a coordinator. Use peers(action='describe', name=<peer>) "
            "to look up a peer's profile. Reply to the user with the peer's "
            "primary claimed capability, lower-cased, and nothing else."
        ),
        config=anthropic_config,
    )
    bob = Agent(
        name="bob",
        prompt="You are bob.",
        config=anthropic_config,
    )

    alice_c = await hub.register(alice)
    await hub.register(
        bob,
        resume=Resume(
            claimed_capabilities=["arithmetic"],
            domains=["mathematics"],
            summary="bob handles arithmetic word problems.",
        ),
    )

    reply = await alice_c.agent.ask("Look up bob and tell me what bob does.")

    assert reply.body is not None
    assert "arithmetic" in reply.body.lower(), (
        f"expected fallback skill render to expose 'arithmetic', got: {reply.body!r}"
    )

    await hub.close()


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_channels_close_invoked_by_llm(
    anthropic_config: AnthropicConfig,
) -> None:
    """The LLM uses ``channels(action='close')`` to terminate a channel.

    alice opens a conversation, sends a final message, then is told to
    close the channel via the tool. We verify the channel ends in
    ``CLOSED`` state.
    """
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    alice = Agent(
        name="alice",
        prompt=(
            "You are a coordinator. When the user tells you to close the "
            "current channel, call channels(action='close') and confirm "
            "you closed it in one short sentence."
        ),
        config=anthropic_config,
    )
    bob = Agent(name="bob", prompt="You are bob.", config=anthropic_config)

    alice_c = await hub.register(alice)
    bob_c = await hub.register(bob)

    channel = await alice_c.open(type=CONVERSATION_TYPE, target=bob_c.agent_id)

    deps = {
        CHANNEL_DEP: Channel(metadata=channel.metadata, client=alice_c),
        AGENT_CLIENT_DEP: alice_c,
        HUB_DEP: hub,
    }
    await alice_c.agent.ask(
        "We're done with this conversation. Please close the current channel.",
        dependencies=deps,
    )

    # Allow the close to propagate through the dispatch loop.
    deadline = asyncio.get_event_loop().time() + 30.0
    while asyncio.get_event_loop().time() < deadline:
        metadata = await hub.get_channel(channel.channel_id)
        if metadata.state == ChannelState.CLOSED:
            break
        await asyncio.sleep(0.2)

    metadata = await hub.get_channel(channel.channel_id)
    assert metadata.state == ChannelState.CLOSED, f"expected channel to be CLOSED, got {metadata.state}"

    await hub.close()


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_context_search_finds_earlier_turn(
    anthropic_config: AnthropicConfig,
) -> None:
    """The LLM uses ``context(action="search")`` to locate an earlier turn.

    alice opens a conversation, manually sends two distinct messages,
    then asks alice's LLM (within the same channel context) to search
    for the password phrase and report it.
    """
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    alice = Agent(
        name="alice",
        prompt=(
            "You are an assistant on a multi-agent network. When asked to "
            "find something earlier in the channel, call "
            "context(action='search', query=<term>, scope='channel') and "
            "report the matching text in one short sentence."
        ),
        config=anthropic_config,
    )
    bob = Agent(name="bob", prompt="You are bob.", config=anthropic_config)

    alice_c = await hub.register(alice)
    bob_c = await hub.register(bob)

    channel = await alice_c.open(type=CONVERSATION_TYPE, target=bob_c.agent_id)

    # Seed the WAL with a unique fact alice can later search for.
    await channel.send("FYI: the project codename is QUARTZSTONE-2026.")
    # Wait for bob's reply (1+1 conversation).
    await _wait_for_text_count(hub, channel.channel_id, expected=2, timeout=30.0)

    deps = {
        CHANNEL_DEP: Channel(metadata=channel.metadata, client=alice_c),
        AGENT_CLIENT_DEP: alice_c,
        HUB_DEP: hub,
    }
    reply = await alice_c.agent.ask(
        "Earlier in this channel I mentioned a project codename. "
        "Search the channel for the word 'codename' and tell me the value.",
        dependencies=deps,
    )

    assert reply.body is not None
    assert "QUARTZSTONE-2026" in reply.body or "quartzstone-2026" in reply.body.lower(), (
        f"expected reply to include 'QUARTZSTONE-2026', got: {reply.body!r}"
    )

    await hub.close()


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_workflow_graph_with_two_handoff_tools(
    anthropic_config: AnthropicConfig,
) -> None:
    """Triage with two hand-written handoff tools picks the right one.

    triage has two ``@tool``-decorated handoff functions that each
    return a typed ``Handoff``; the matching ``ToolCalled →
    AgentTarget`` transitions in the graph remain for documentation /
    auditability. The LLM is given a billing-flavoured prompt; we
    assert the ``EV_PACKET`` in the WAL carries
    ``routing.tool == transfer_to_billing``.

    The flow uses a separate ``user`` agent as the workflow's initial
    speaker so that triage's first turn runs through her notify
    handler — that's where the framework builds the ``EV_PACKET``
    capturing her routing decision.
    """
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    user_agent = Agent(name="user", config=TestConfig())
    triage = Agent(
        name="triage",
        prompt=(
            "You are the triage coordinator. Route engineering questions "
            "to engineering via transfer_to_eng, and billing/refund/payment "
            "questions to billing via transfer_to_billing. Always call exactly "
            "one of the transfer tools — never answer directly."
        ),
        config=anthropic_config,
    )
    eng = Agent(name="eng", prompt="You are engineering.", config=anthropic_config)
    billing = Agent(name="billing", prompt="You are billing.", config=anthropic_config)

    user_c = await hub.register(user_agent)
    triage_c = await hub.register(triage)
    eng_c = await hub.register(eng)
    billing_c = await hub.register(billing)

    eng_id = eng_c.agent_id
    billing_id = billing_c.agent_id

    @triage.tool
    async def transfer_to_eng(reason: str = "") -> Handoff:
        """Route an engineering/coding/infrastructure question to the engineering team."""
        return Handoff(target=eng_id, reason=reason)

    @triage.tool
    async def transfer_to_billing(reason: str = "") -> Handoff:
        """Route a billing/refund/payment question to the billing team."""
        return Handoff(target=billing_id, reason=reason)

    graph = TransitionGraph(
        initial_speaker=user_c.agent_id,
        transitions=[
            Transition(when=FromSpeaker(user_c.agent_id), then=AgentTarget(triage_c.agent_id)),
            Transition(
                when=ToolCalled("transfer_to_eng"),
                then=AgentTarget(eng_c.agent_id),
            ),
            Transition(
                when=ToolCalled("transfer_to_billing"),
                then=AgentTarget(billing_c.agent_id),
            ),
        ],
        default_target=StayTarget(),
        max_turns=3,
    )

    channel = await user_c.open(
        type=WORKFLOW_TYPE,
        target=[triage_c.agent_id, eng_c.agent_id, billing_c.agent_id],
        knobs={"graph": graph.to_dict()},
        intent="triage routes to eng or billing",
    )
    await channel.send("I want a refund on a charge from yesterday. Please route me to the right team.")

    # Wait for triage's routing packet to land.
    deadline = asyncio.get_event_loop().time() + 60.0
    handoffs: list = []
    while asyncio.get_event_loop().time() < deadline:
        wal = await hub.read_wal(channel.channel_id)
        handoffs = [
            e
            for e in wal
            if e.event_type == EV_PACKET and (e.event_data.get("routing", {}) or {}).get("kind") == "handoff"
        ]
        if handoffs:
            break
        await asyncio.sleep(0.2)

    assert handoffs, "triage did not call any handoff tool"
    chosen_tool = (handoffs[0].event_data.get("routing", {}) or {}).get("tool")
    assert chosen_tool == "transfer_to_billing", f"expected billing routing, got tool={chosen_tool!r}"

    await hub.close()
