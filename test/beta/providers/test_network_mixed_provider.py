# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Mixed-provider network smoke tests.

The network module is meant to be provider-neutral: a single hub
should be able to host agents whose ``Agent.config`` is Anthropic,
OpenAI, or Gemini, and the choreography (consulting handshake,
round-robin discussion, workflow handoffs) should not care.

These tests register agents on different providers in the same hub
channel and verify the protocol still flows. They are skipped if any
of the three keys are missing.

Marked ``@pytest.mark.anthropic`` so they run alongside the existing
multi-provider smoke set; the test body itself enforces all three
keys via fixture skip.
"""

import asyncio
import os
from pathlib import Path

import pytest

# These tests host Anthropic, OpenAI, and Gemini agents in the same hub
pytest.importorskip("anthropic")
pytest.importorskip("openai")
pytest.importorskip("google.genai")

from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig, GeminiConfig, OpenAIConfig
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
from autogen.beta.network.adapters.discussion import (
    DISCUSSION_TYPE,
    ORDERING_ROUND_ROBIN,
)
from autogen.beta.network.adapters.workflow import WORKFLOW_TYPE
from autogen.beta.network.client.channel import Channel
from autogen.beta.network.policies import (
    AGENT_CLIENT_DEP,
    CHANNEL_DEP,
    HUB_DEP,
)
from autogen.beta.network.transitions import (
    AgentTarget,
    FromSpeaker,
    RevertToInitiatorTarget,
    TerminateTarget,
    ToolCalled,
    Transition,
    TransitionGraph,
)

try:
    from dotenv import load_dotenv

    _REPO_ROOT = Path(__file__).resolve().parents[3]
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass


def _require_all_keys() -> tuple[str, str, str]:
    anthropic = os.getenv("ANTHROPIC_API_KEY")
    openai = os.getenv("OPENAI_API_KEY")
    gemini = os.getenv("GEMINI_API_KEY")
    if not (anthropic and openai and gemini):
        pytest.skip("mixed-provider tests require ANTHROPIC_API_KEY, OPENAI_API_KEY, and GEMINI_API_KEY")
    return anthropic, openai, gemini


async def _wait_for_text_count(
    hub: Hub,
    channel_id: str,
    expected: int,
    *,
    timeout: float = 90.0,
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
@pytest.mark.openai
@pytest.mark.asyncio()
async def test_consulting_anthropic_initiator_openai_specialist() -> None:
    """alice (Anthropic) discovers + delegates to bob (OpenAI).

    Proves that ``peers`` / ``delegate`` work across providers — the
    initiator's tool-calling format and the respondent's tool-calling
    format are independent. The hub never inspects either format.
    """
    anth_key, oai_key, _ = _require_all_keys()
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    link = LocalLink(hub)

    alice = Agent(
        name="alice",
        prompt=(
            "You are a coordinator on a multi-agent network. When given a "
            "question, FIRST call peers(action='find', capability=<topic>) "
            "to discover a specialist. Then call delegate(target=<peer_name>, "
            "prompt=<question>) to ask them. Reply to the user with the "
            "specialist's answer verbatim."
        ),
        config=AnthropicConfig(model="claude-haiku-4-5", api_key=anth_key, temperature=0),
    )
    bob = Agent(
        name="bob",
        prompt=("You are a math specialist. Answer with just the numeric result, no explanation."),
        config=OpenAIConfig(model="gpt-5.4-nano", api_key=oai_key, temperature=0),
    )

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice_c = await alice_hc.register(alice, Passport(name="alice"), Resume(summary="multi-agent coordinator"))
    await bob_hc.register(bob, Passport(name="bob"), Resume(claimed_capabilities=["math"], summary="math"))

    reply = await alice_c.agent.ask("Find a math specialist on the network and ask them: what is 14 times 19?")

    assert reply.body is not None
    assert "266" in reply.body, f"expected 266 in alice's reply, got: {reply.body!r}"

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.anthropic
@pytest.mark.openai
@pytest.mark.gemini
@pytest.mark.asyncio()
async def test_3way_discussion_one_per_provider() -> None:
    """Three agents (Anthropic, OpenAI, Gemini) take round-robin turns.

    Each agent's ``say`` tool call is converted by *its* provider's
    mapper, but the WAL sees a uniform ``EV_TEXT`` envelope. The
    discussion adapter doesn't care which provider produced the turn.
    """
    anth_key, oai_key, gemini_key = _require_all_keys()
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    link = LocalLink(hub)

    configs = {
        "alice": AnthropicConfig(model="claude-haiku-4-5", api_key=anth_key, temperature=0),
        "bob": OpenAIConfig(model="gpt-5.4-nano", api_key=oai_key, temperature=0),
        "carol": GeminiConfig(model="gemini-3-flash-preview", api_key=gemini_key, temperature=0),
    }

    clients = []
    for name, cfg in configs.items():
        agent = Agent(
            name=name,
            prompt=(
                f"You are {name}, a participant in a 3-way discussion on a "
                "topic. When it is your turn, contribute exactly one short "
                "opinion (one sentence) by calling say(content=<your "
                "sentence>). Do not ask questions. Do not call any other tool."
            ),
            config=cfg,
        )
        hc = HubClient(link, hub=hub)
        client = await hc.register(agent, Passport(name=name), Resume())
        clients.append(client)

    alice = clients[0]

    channel = await alice.open(
        type=DISCUSSION_TYPE,
        target=[c.agent_id for c in clients[1:]],
        knobs={"ordering": ORDERING_ROUND_ROBIN},
        intent="quick mixed-provider debate",
    )

    await channel.send("Quick debate: should new web apps default to server-side rendering?")

    count = await _wait_for_text_count(hub, channel.channel_id, expected=3, timeout=180.0)
    assert count >= 3, f"expected 3 turns, got {count}"

    wal = await hub.read_wal(channel.channel_id)
    speakers = [e.sender_id for e in wal if e.event_type == EV_TEXT][:3]
    expected_order = [c.agent_id for c in clients]
    assert speakers == expected_order, (
        f"round-robin order broken across providers; expected {expected_order}, got {speakers}"
    )

    contributions = [e.event_data.get("text", "") for e in wal if e.event_type == EV_TEXT][:3]
    for i, text in enumerate(contributions):
        assert len(text) > 5, f"turn {i} from {list(configs)[i]} was empty/trivial: {text!r}"

    for c in clients:
        await c._hub_client.close()
    await hub.close()


@pytest.mark.anthropic
@pytest.mark.openai
@pytest.mark.asyncio()
async def test_workflow_handoff_anthropic_to_openai() -> None:
    """Workflow handoff across providers.

    triage (Anthropic) calls ``transfer_to_eng`` → eng (OpenAI) replies →
    ``RevertToInitiatorTarget`` rotates back to triage. Verifies the
    routing ``EV_PACKET`` is provider-neutral and that the receiving
    agent's notify handler engages regardless of provider.
    """
    anth_key, oai_key, _ = _require_all_keys()
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    link = LocalLink(hub)

    triage_agent = Agent(
        name="triage",
        prompt=(
            "You are the triage coordinator. When the user asks an "
            "engineering question, call the transfer_to_eng tool with a "
            "one-line reason. Do not try to answer it yourself."
        ),
        config=AnthropicConfig(model="claude-haiku-4-5", api_key=anth_key, temperature=0),
    )
    eng_agent = Agent(
        name="eng",
        prompt="You are a senior engineer. Answer concisely in one sentence.",
        config=OpenAIConfig(model="gpt-5.4-nano", api_key=oai_key, temperature=0),
    )

    triage_hc = HubClient(link, hub=hub)
    eng_hc = HubClient(link, hub=hub)
    triage = await triage_hc.register(triage_agent, Passport(name="triage"), Resume(claimed_capabilities=["triage"]))
    eng = await eng_hc.register(eng_agent, Passport(name="eng"), Resume(claimed_capabilities=["engineering"]))

    graph = TransitionGraph(
        initial_speaker=triage.agent_id,
        transitions=[
            Transition(when=ToolCalled("transfer_to_eng"), then=AgentTarget(eng.agent_id)),
            Transition(when=FromSpeaker(eng.agent_id), then=RevertToInitiatorTarget()),
        ],
        default_target=TerminateTarget(reason="triage_done"),
        max_turns=6,
    )

    # Attach a hand-written handoff tool on triage. Returns a typed
    # ``Handoff(target=eng.agent_id)`` so the workflow adapter folds
    # the next speaker to eng.
    eng_agent_id = eng.agent_id

    @triage_agent.tool
    async def transfer_to_eng(reason: str = "") -> Handoff:
        """Transfer the conversation to the engineering specialist."""
        return Handoff(target=eng_agent_id, reason=reason)

    channel = await triage.open(
        type=WORKFLOW_TYPE,
        target=[eng.agent_id],
        knobs={"graph": graph.to_dict()},
        intent="triage routes deep questions to engineering",
    )

    triage_deps = {
        CHANNEL_DEP: Channel(metadata=channel.metadata, client=triage),
        AGENT_CLIENT_DEP: triage,
        HUB_DEP: hub,
    }
    await triage.agent.ask(
        "What's the practical difference between processes and threads in Python?",
        dependencies=triage_deps,
    )

    # Confirm the handoff happened.
    wal = await hub.read_wal(channel.channel_id)
    handoff_envelopes = [
        e for e in wal if e.event_type == EV_PACKET and (e.event_data.get("routing", {}) or {}).get("kind") == "handoff"
    ]
    assert handoff_envelopes, "triage did not call transfer_to_eng"

    # Wait up to 60s for eng (OpenAI) to engage and reply.
    deadline = asyncio.get_event_loop().time() + 60.0
    eng_replied = False
    while asyncio.get_event_loop().time() < deadline:
        state = hub._adapter_states.get(channel.channel_id)
        if state is not None and state.last_speaker_id == eng.agent_id:
            eng_replied = True
            break
        await asyncio.sleep(0.2)
    assert eng_replied, "eng (OpenAI) did not reply to triage's (Anthropic) handoff within 60s"

    # After RevertToInitiator, expected_next_speaker should be triage again.
    state = hub._adapter_states[channel.channel_id]
    assert state.expected_next_speaker == triage.agent_id

    await triage_hc.close()
    await eng_hc.close()
    await hub.close()
