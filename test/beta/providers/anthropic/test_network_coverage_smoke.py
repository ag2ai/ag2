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
* ``sessions(action="close")`` — LLM closes the session it owns.
* ``context(action="search")`` — LLM finds an earlier turn by substring.
* Multi-tool workflow graph — graph with two ``ToolCalled`` transitions
  materialises two distinct handoff tools and the LLM picks the right one.

Uses ``claude-haiku-4-5`` for cost.
"""

import asyncio
import os
from pathlib import Path

import pytest

from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    EV_HANDOFF,
    EV_TEXT,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)
from autogen.beta.network.adapters.conversation import CONVERSATION_TYPE
from autogen.beta.network.adapters.workflow import WORKFLOW_TYPE
from autogen.beta.network.client.plugin import NetworkPlugin
from autogen.beta.network.client.session import Session
from autogen.beta.network.policies import (
    AGENT_CLIENT_DEP,
    HUB_DEP,
    SESSION_DEP,
)
from autogen.beta.network.session import SessionState
from autogen.beta.network.transitions import (
    AgentTarget,
    StayTarget,
    TerminateTarget,
    ToolCalled,
    Transition,
    TransitionGraph,
)

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
    session_id: str,
    expected: int,
    *,
    timeout: float = 60.0,
) -> int:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        wal = await hub.read_wal(session_id)
        count = sum(1 for e in wal if e.event_type == EV_TEXT)
        if count >= expected:
            return count
        await asyncio.sleep(0.2)
    return sum(
        1 for e in (await hub.read_wal(session_id)) if e.event_type == EV_TEXT
    )


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_conversation_adapter_bidirectional_two_turns(
    anthropic_config: AnthropicConfig,
) -> None:
    """The ``ConversationAdapter`` runs 1+1 indefinitely until close.

    alice opens a conversation with bob, sends a question, bob's notify
    handler engages bob's LLM and replies. The default adapter does NOT
    auto-close (unlike consulting), so the session stays ACTIVE. We
    verify both turns landed and the session is still active.
    """
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    link = LocalLink(hub)

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

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice_c = await alice_hc.register(alice, Passport(name="alice"), Resume())
    bob_c = await bob_hc.register(bob, Passport(name="bob"), Resume())

    session = await alice_c.open(type=CONVERSATION_TYPE, target=bob_c.agent_id)

    await session.send("Hi bob. What's a good Python web framework for tiny APIs?")

    # Wait for bob's reply to land.
    count = await _wait_for_text_count(
        hub, session.session_id, expected=2, timeout=60.0
    )
    assert count >= 2, f"expected 2 turns (alice + bob), got {count}"

    # Conversation is still active — no auto-close.
    metadata = await hub.get_session(session.session_id)
    assert metadata.state == SessionState.ACTIVE

    await session.close()
    metadata = await hub.get_session(session.session_id)
    assert metadata.state == SessionState.CLOSED

    await alice_hc.close()
    await bob_hc.close()
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
    link = LocalLink(hub)

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

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice_c = await alice_hc.register(alice, Passport(name="alice"), Resume())
    await bob_hc.register(
        bob,
        Passport(name="bob"),
        Resume(
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

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_sessions_close_invoked_by_llm(
    anthropic_config: AnthropicConfig,
) -> None:
    """The LLM uses ``sessions(action='close')`` to terminate a session.

    alice opens a conversation, sends a final message, then is told to
    close the session via the tool. We verify the session ends in
    ``CLOSED`` state.
    """
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    link = LocalLink(hub)

    alice = Agent(
        name="alice",
        prompt=(
            "You are a coordinator. When the user tells you to close the "
            "current session, call sessions(action='close') and confirm "
            "you closed it in one short sentence."
        ),
        config=anthropic_config,
    )
    bob = Agent(name="bob", prompt="You are bob.", config=anthropic_config)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice_c = await alice_hc.register(alice, Passport(name="alice"), Resume())
    bob_c = await bob_hc.register(bob, Passport(name="bob"), Resume())

    session = await alice_c.open(type=CONVERSATION_TYPE, target=bob_c.agent_id)

    deps = {
        SESSION_DEP: Session(metadata=session.metadata, client=alice_c),
        AGENT_CLIENT_DEP: alice_c,
        HUB_DEP: hub,
    }
    await alice_c.agent.ask(
        "We're done with this conversation. Please close the current session.",
        dependencies=deps,
    )

    # Allow the close to propagate through the dispatch loop.
    deadline = asyncio.get_event_loop().time() + 30.0
    while asyncio.get_event_loop().time() < deadline:
        metadata = await hub.get_session(session.session_id)
        if metadata.state == SessionState.CLOSED:
            break
        await asyncio.sleep(0.2)

    metadata = await hub.get_session(session.session_id)
    assert metadata.state == SessionState.CLOSED, (
        f"expected session to be CLOSED, got {metadata.state}"
    )

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_context_search_finds_earlier_turn(
    anthropic_config: AnthropicConfig,
) -> None:
    """The LLM uses ``context(action="search")`` to locate an earlier turn.

    alice opens a conversation, manually sends two distinct messages,
    then asks alice's LLM (within the same session context) to search
    for the password phrase and report it.
    """
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    link = LocalLink(hub)

    alice = Agent(
        name="alice",
        prompt=(
            "You are an assistant on a multi-agent network. When asked to "
            "find something earlier in the session, call "
            "context(action='search', query=<term>, scope='session') and "
            "report the matching text in one short sentence."
        ),
        config=anthropic_config,
    )
    bob = Agent(name="bob", prompt="You are bob.", config=anthropic_config)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice_c = await alice_hc.register(alice, Passport(name="alice"), Resume())
    bob_c = await bob_hc.register(bob, Passport(name="bob"), Resume())

    session = await alice_c.open(type=CONVERSATION_TYPE, target=bob_c.agent_id)

    # Seed the WAL with a unique fact alice can later search for.
    await session.send("FYI: the project codename is QUARTZSTONE-2026.")
    # Wait for bob's reply (1+1 conversation).
    await _wait_for_text_count(hub, session.session_id, expected=2, timeout=30.0)

    deps = {
        SESSION_DEP: Session(metadata=session.metadata, client=alice_c),
        AGENT_CLIENT_DEP: alice_c,
        HUB_DEP: hub,
    }
    reply = await alice_c.agent.ask(
        "Earlier in this session I mentioned a project codename. "
        "Search the session for the word 'codename' and tell me the value.",
        dependencies=deps,
    )

    assert reply.body is not None
    assert "QUARTZSTONE-2026" in reply.body or "quartzstone-2026" in reply.body.lower(), (
        f"expected reply to include 'QUARTZSTONE-2026', got: {reply.body!r}"
    )

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_workflow_graph_with_two_handoff_tools(
    anthropic_config: AnthropicConfig,
) -> None:
    """A workflow graph with two ``ToolCalled`` transitions materialises
    two distinct handoff tools and the LLM picks the right one.

    triage routes engineering questions to ``eng`` via ``transfer_to_eng``
    and billing questions to ``billing`` via ``transfer_to_billing``.
    The LLM is given a billing-flavoured prompt; we assert the EV_HANDOFF
    in the WAL has tool=transfer_to_billing.
    """
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    link = LocalLink(hub)

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
    eng = Agent(
        name="eng", prompt="You are engineering.", config=anthropic_config
    )
    billing = Agent(
        name="billing", prompt="You are billing.", config=anthropic_config
    )

    triage_hc = HubClient(link, hub=hub)
    eng_hc = HubClient(link, hub=hub)
    billing_hc = HubClient(link, hub=hub)

    triage_c = await triage_hc.register(triage, Passport(name="triage"), Resume())
    eng_c = await eng_hc.register(eng, Passport(name="eng"), Resume())
    billing_c = await billing_hc.register(billing, Passport(name="billing"), Resume())

    graph = TransitionGraph(
        initial_speaker=triage_c.agent_id,
        transitions=[
            Transition(
                when=ToolCalled("transfer_to_eng"),
                then=AgentTarget(eng_c.agent_id),
            ),
            Transition(
                when=ToolCalled("transfer_to_billing"),
                then=AgentTarget(billing_c.agent_id),
            ),
        ],
        # StayTarget keeps the workflow alive after the handoff so the
        # exit assertion only checks the routing tool, not the rest.
        default_target=StayTarget(),
        max_turns=2,
    )

    plugin = NetworkPlugin(triage_c)
    plugin.register_workflow(graph)
    triage_tool_names = {t.name for t in triage_c.agent.tools}
    assert "transfer_to_eng" in triage_tool_names
    assert "transfer_to_billing" in triage_tool_names

    session = await triage_c.open(
        type=WORKFLOW_TYPE,
        target=[eng_c.agent_id, billing_c.agent_id],
        knobs={"graph": graph.to_dict()},
        intent="triage routes to eng or billing",
    )
    deps = {
        SESSION_DEP: Session(metadata=session.metadata, client=triage_c),
        AGENT_CLIENT_DEP: triage_c,
        HUB_DEP: hub,
    }
    await triage_c.agent.ask(
        "I want a refund on a charge from yesterday. Please route me to the right team.",
        dependencies=deps,
    )

    wal = await hub.read_wal(session.session_id)
    handoffs = [e for e in wal if e.event_type == EV_HANDOFF]
    assert handoffs, "triage did not call any handoff tool"
    chosen_tool = handoffs[0].event_data.get("tool")
    assert chosen_tool == "transfer_to_billing", (
        f"expected billing routing, got tool={chosen_tool!r}"
    )

    await triage_hc.close()
    await eng_hc.close()
    await billing_hc.close()
    await hub.close()
