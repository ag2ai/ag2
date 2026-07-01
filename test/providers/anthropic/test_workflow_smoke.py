# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow smoke test against a real LLM.

A swarm runs end-to-end via tool-driven handoffs. A human participant
seeds the workflow by sending the initial question; the workflow graph
routes the human's turn to the triage agent. Triage calls
``transfer_to_eng(reason)`` → eng agent replies →
``RevertToInitiatorTarget`` brings control back to triage (the channel
creator) → the test closes the channel deterministically. Workflow
state survives ``Hub.hydrate()`` mid-flow.

Uses ``claude-haiku-4-5`` for cost; loads ``.env`` from the repo
root; skips if ``ANTHROPIC_API_KEY`` is unset; marked
``@pytest.mark.anthropic`` so the default unit run skips it.
"""

import asyncio
import os
from pathlib import Path

import pytest

from ag2 import Agent
from ag2.config import AnthropicConfig
from ag2.knowledge import DiskKnowledgeStore
from ag2.network import (
    EV_PACKET,
    Handoff,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)
from ag2.network.adapters.workflow import WORKFLOW_TYPE, WorkflowState
from ag2.network.channel import ChannelState
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


async def _wait_for_state(
    hub: Hub,
    channel_id: str,
    *,
    pred,
    timeout: float = 60.0,
) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if pred(hub.adapter_state(channel_id)):
            return True
        await asyncio.sleep(0.2)
    return False


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_workflow_swarm_handoff_revert_close(
    anthropic_config: AnthropicConfig,
    tmp_path,
) -> None:
    """Human seeds → triage routes to eng via transfer_to_eng tool →
    FromSpeaker(eng) reverts to triage (channel creator) → test closes
    the workflow. State survives a mid-flow Hub.hydrate()."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    triage_agent = Agent(
        name="triage",
        prompt=(
            "You are the triage coordinator. When the user asks a question "
            "that requires engineering expertise, call the transfer_to_eng "
            "tool (with a one-line reason) — do not try to answer it "
            "yourself. After eng replies and control returns to you, "
            "summarise the answer in one short sentence and stop."
        ),
        config=anthropic_config,
    )
    eng_agent = Agent(
        name="eng",
        prompt=("You are a senior engineer. Answer the question concisely in one or two sentences."),
        config=anthropic_config,
    )

    user_hc = HubClient(link, hub=hub)
    triage = await hub.register(triage_agent, resume=Resume(claimed_capabilities=["triage"]))
    eng = await hub.register(eng_agent, resume=Resume(claimed_capabilities=["engineering"]))
    user = await user_hc.register_human(Passport(name="user"))

    graph = TransitionGraph(
        initial_speaker=user.agent_id,
        transitions=[
            Transition(
                when=FromSpeaker(user.agent_id),
                then=AgentTarget(triage.agent_id),
            ),
            Transition(
                when=ToolCalled("transfer_to_eng"),
                then=AgentTarget(eng.agent_id),
            ),
            Transition(
                when=FromSpeaker(eng.agent_id),
                then=RevertToInitiatorTarget(),
            ),
        ],
        default_target=TerminateTarget(reason="triage_done"),
        max_turns=8,
    )

    eng_agent_id = eng.agent_id

    @triage.agent.tool
    async def transfer_to_eng(reason: str = "") -> Handoff:
        """Transfer the conversation to the engineering specialist."""
        return Handoff(target=eng_agent_id, reason=reason)

    # Triage opens the workflow (so triage is the channel creator —
    # RevertToInitiatorTarget routes back to triage). The user is a
    # participant + initial_speaker; eng is the handoff target.
    channel = await triage.open(
        type=WORKFLOW_TYPE,
        target=[user.agent_id, eng.agent_id],
        knobs={"graph": graph.to_dict()},
        intent="triage routes deep questions to engineering",
    )

    # User seeds the workflow with the prompt. The graph's
    # FromSpeaker(user)→AgentTarget(triage) advances expected_next_speaker
    # to triage; triage's notify handler then engages triage's LLM with
    # the user's text as input — no direct agent.ask hack.
    await user.send(
        channel.channel_id,
        "Why is async file I/O typically slower than sync for small reads?",
    )

    # Wait for triage's handoff tool to land as an EV_PACKET in the WAL.
    handoff_envelopes: list = []

    async def _wait_for_handoff(timeout: float) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            wal = await hub.read_wal(channel.channel_id)
            handoff_envelopes.clear()
            handoff_envelopes.extend(
                e
                for e in wal
                if e.event_type == EV_PACKET and (e.event_data.get("routing", {}) or {}).get("kind") == "handoff"
            )
            if handoff_envelopes:
                return
            await asyncio.sleep(0.5)

    await _wait_for_handoff(timeout=60.0)
    assert handoff_envelopes, "triage did not call transfer_to_eng within 60s"

    # After the handoff, expected_next_speaker is eng. Wait for eng's
    # notify handler (default) to engage and reply.
    settled = await _wait_for_state(
        hub,
        channel.channel_id,
        pred=lambda s: s is not None and s.last_speaker_id == eng.agent_id,
        timeout=60.0,
    )
    assert settled, "eng never replied within 60s"

    state = hub.adapter_state(channel.channel_id)
    # FromSpeaker(eng) → RevertToInitiator → next is triage (channel creator).
    assert state.expected_next_speaker == triage.agent_id

    # ── Hub.hydrate() mid-flow: tear down + re-open against the same store.
    await user_hc.close()
    await hub.close()

    store2 = DiskKnowledgeStore(str(tmp_path))
    hub2 = await Hub.open(store2, ttl_sweep_interval=0, expectation_sweep_interval=0)
    rebuilt = hub2._adapter_states[channel.channel_id]
    assert isinstance(rebuilt, WorkflowState)
    assert rebuilt.expected_next_speaker == triage.agent_id
    assert rebuilt.last_speaker_id == eng.agent_id
    assert rebuilt.creator_id == triage.agent_id

    # Triage closes via TerminateTarget — equivalent to calling
    # channels(action="close"). For the deterministic verification we
    # use the hub directly.
    closed = await hub2.close_channel(channel.channel_id, reason="triage_done")
    assert closed.state == ChannelState.CLOSED
    assert closed.close_reason == "triage_done"

    await hub2.close()
