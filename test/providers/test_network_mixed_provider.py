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

from ag2 import Agent
from ag2.config import AnthropicConfig, GeminiConfig, OpenAIConfig
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_PACKET,
    EV_TEXT,
    Handoff,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)
from ag2.network.adapters.discussion import (
    DISCUSSION_TYPE,
    ORDERING_ROUND_ROBIN,
)
from ag2.network.adapters.workflow import WORKFLOW_TYPE
from ag2.network.transitions import (
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

    alice_c = await hub.register(alice, resume=Resume(summary="multi-agent coordinator"))
    await hub.register(bob, resume=Resume(claimed_capabilities=["math"], summary="math"))

    reply = await alice_c.agent.ask("Find a math specialist on the network and ask them: what is 14 times 19?")

    assert reply.body is not None
    assert "266" in reply.body, f"expected 266 in alice's reply, got: {reply.body!r}"

    await hub.close()


@pytest.mark.anthropic
@pytest.mark.openai
@pytest.mark.gemini
@pytest.mark.asyncio()
async def test_3way_discussion_one_per_provider() -> None:
    """Three agents (Anthropic, OpenAI, Gemini) take round-robin turns.

    Each agent contributes a plain-text reply produced by *its* provider,
    but the WAL sees a uniform ``EV_TEXT`` envelope. The discussion adapter
    doesn't care which provider produced the turn. Since #2886 the adapter
    offers no ``say`` tool — each participant's reply is posted as the
    round-end ``EV_TEXT``.
    """
    anth_key, oai_key, gemini_key = _require_all_keys()
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )

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
                "topic. When it is your turn, reply with exactly one short "
                "opinion (one sentence) as plain text. Do not ask questions "
                "and do not call any tools — just state your opinion."
            ),
            config=cfg,
        )
        client = await hub.register(agent)
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

    await hub.close()


async def _wait_for_handoff(hub: Hub, channel_id: str, *, timeout: float = 60.0) -> list:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        wal = await hub.read_wal(channel_id)
        handoffs = [
            e
            for e in wal
            if e.event_type == EV_PACKET and (e.event_data.get("routing", {}) or {}).get("kind") == "handoff"
        ]
        if handoffs:
            return handoffs
        await asyncio.sleep(0.5)
    return []


async def _wait_for_adapter_state(hub: Hub, channel_id: str, pred, *, timeout: float = 60.0) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if pred(hub.adapter_state(channel_id)):
            return True
        await asyncio.sleep(0.2)
    return False


@pytest.mark.anthropic
@pytest.mark.openai
@pytest.mark.asyncio()
async def test_workflow_handoff_anthropic_to_openai() -> None:
    """Workflow handoff across providers.

    A human seeds the workflow; the graph routes the human's turn to
    triage (Anthropic), which calls ``transfer_to_eng`` → eng (OpenAI)
    replies (via its notify handler) → ``RevertToInitiatorTarget``
    rotates back to triage. Verifies the routing ``EV_PACKET`` and the
    receiving agent's notify handler are provider-neutral.
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

    user_hc = HubClient(link, hub=hub)
    triage = await hub.register(triage_agent, resume=Resume(claimed_capabilities=["triage"]))
    eng = await hub.register(eng_agent, resume=Resume(claimed_capabilities=["engineering"]))
    user = await user_hc.register_human(Passport(name="user"))

    graph = TransitionGraph(
        initial_speaker=user.agent_id,
        transitions=[
            Transition(when=FromSpeaker(user.agent_id), then=AgentTarget(triage.agent_id)),
            Transition(when=ToolCalled("transfer_to_eng"), then=AgentTarget(eng.agent_id)),
            Transition(when=FromSpeaker(eng.agent_id), then=RevertToInitiatorTarget()),
        ],
        default_target=TerminateTarget(reason="triage_done"),
        max_turns=8,
    )

    # Attach a hand-written handoff tool on triage. Returns a typed
    # ``Handoff(target=eng.agent_id)`` so the workflow adapter folds
    # the next speaker to eng.
    eng_agent_id = eng.agent_id

    @triage.agent.tool
    async def transfer_to_eng(reason: str = "") -> Handoff:
        """Transfer the conversation to the engineering specialist."""
        return Handoff(target=eng_agent_id, reason=reason)

    # Triage opens the workflow (so it's the channel creator —
    # RevertToInitiatorTarget routes back to triage). The user is a
    # participant + initial_speaker; eng is the handoff target.
    channel = await triage.open(
        type=WORKFLOW_TYPE,
        target=[user.agent_id, eng.agent_id],
        knobs={"graph": graph.to_dict()},
        intent="triage routes deep questions to engineering",
    )

    # User seeds the workflow. FromSpeaker(user)→AgentTarget(triage)
    # advances expected_next_speaker to triage, whose notify handler
    # then engages triage's LLM with the user's text — no direct
    # ``agent.ask`` call.
    await user.send(
        channel.channel_id,
        "What's the practical difference between processes and threads in Python?",
    )

    handoff_envelopes = await _wait_for_handoff(hub, channel.channel_id, timeout=60.0)
    assert handoff_envelopes, "triage did not call transfer_to_eng within 60s"

    # Wait for eng (OpenAI) to engage and reply via its notify handler.
    eng_replied = await _wait_for_adapter_state(
        hub,
        channel.channel_id,
        lambda s: s is not None and s.last_speaker_id == eng.agent_id,
        timeout=60.0,
    )
    assert eng_replied, "eng (OpenAI) did not reply to triage's (Anthropic) handoff within 60s"

    # After RevertToInitiator, expected_next_speaker should be triage again.
    state = hub.adapter_state(channel.channel_id)
    assert state.expected_next_speaker == triage.agent_id

    await user_hc.close()
    await hub.close()
