# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow smoke test against a real LLM.

A 3-agent swarm runs end-to-end via tool-driven handoffs. Triage
agent calls ``transfer_to_eng(reason)`` → eng agent replies →
``RevertToInitiatorTarget`` brings control back to triage → triage
closes via ``TerminateTarget``. Workflow state survives
``Hub.hydrate()`` mid-flow.

Uses ``claude-haiku-4-5`` for cost; loads ``.env`` from the repo
root; skips if ``ANTHROPIC_API_KEY`` is unset; marked
``@pytest.mark.anthropic`` so the default unit run skips it.
"""

import asyncio
import os
from pathlib import Path

import pytest

from autogen.beta import Agent
from autogen.beta.config import AnthropicConfig
from autogen.beta.knowledge import DiskKnowledgeStore
from autogen.beta.network import (
    EV_PACKET,
    Handoff,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)
from autogen.beta.network.adapters.workflow import WORKFLOW_TYPE, WorkflowState
from autogen.beta.network.session import SessionState
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
    session_id: str,
    *,
    pred,
    timeout: float = 60.0,
) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if pred(hub._adapter_states.get(session_id)):
            return True
        await asyncio.sleep(0.2)
    return False


@pytest.mark.anthropic
@pytest.mark.asyncio()
async def test_workflow_swarm_handoff_revert_close(
    anthropic_config: AnthropicConfig,
    tmp_path,
) -> None:
    """3-agent swarm: triage hands off to eng via the transfer_to_eng
    tool; FromSpeaker(eng) reverts control to triage; triage closes
    the workflow. Workflow state survives a mid-flow Hub.hydrate()."""
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

    triage_hc = HubClient(link, hub=hub)
    eng_hc = HubClient(link, hub=hub)
    triage = await triage_hc.register(triage_agent, Passport(name="triage"), Resume(claimed_capabilities=["triage"]))
    eng = await eng_hc.register(eng_agent, Passport(name="eng"), Resume(claimed_capabilities=["engineering"]))

    graph = TransitionGraph(
        initial_speaker=triage.agent_id,
        transitions=[
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
        max_turns=6,
    )

    # Attach a hand-written handoff tool on triage. Returns a typed
    # ``Handoff(target=eng.agent_id)`` so the workflow adapter folds
    # the next speaker to eng without needing a matching graph rule.
    eng_agent_id = eng.agent_id

    @triage.agent.tool
    async def transfer_to_eng(reason: str = "") -> Handoff:
        """Transfer the conversation to the engineering specialist."""
        return Handoff(target=eng_agent_id, reason=reason)

    session = await triage.open(
        type=WORKFLOW_TYPE,
        target=[eng.agent_id],
        knobs={"graph": graph.to_dict()},
        intent="triage routes deep questions to engineering",
    )

    # Drive the first turn directly via triage.agent.ask so we can
    # observe the handoff tool call. The session is in the LLM's
    # context via the plugin's NetworkContextPolicy.
    from autogen.beta.network.client.session import Session
    from autogen.beta.network.policies import (
        AGENT_CLIENT_DEP,
        HUB_DEP,
        SESSION_DEP,
    )

    session_handle = Session(metadata=session.metadata, client=triage)
    triage_dependencies = {
        SESSION_DEP: session_handle,
        AGENT_CLIENT_DEP: triage,
        HUB_DEP: hub,
    }
    reply = await triage.agent.ask(
        "Why is async file I/O typically slower than sync for small reads?",
        dependencies=triage_dependencies,
    )
    # Triage either calls transfer_to_eng (preferred) or answers directly;
    # either way the session should have at least one substantive event.
    wal = await hub.read_wal(session.session_id)
    handoff_envelopes = [
        e for e in wal if e.event_type == EV_PACKET and (e.event_data.get("routing", {}) or {}).get("kind") == "handoff"
    ]
    assert handoff_envelopes, f"triage did not call transfer_to_eng; reply={reply.body!r}"

    # After the handoff, expected_next_speaker is eng. Wait for eng's
    # notify handler (auto-attached default) to engage and reply.
    settled = await _wait_for_state(
        hub,
        session.session_id,
        pred=lambda s: s is not None and s.last_speaker_id == eng.agent_id,
        timeout=60.0,
    )
    assert settled, "eng never replied within 60s"

    state = hub._adapter_states[session.session_id]
    # FromSpeaker(eng) → RevertToInitiator → next is triage.
    assert state.expected_next_speaker == triage.agent_id

    # ── Hub.hydrate() mid-flow: tear down + re-open against the same store.
    await triage_hc.close()
    await eng_hc.close()
    await hub.close()

    store2 = DiskKnowledgeStore(str(tmp_path))
    hub2 = await Hub.open(store2, ttl_sweep_interval=0, expectation_sweep_interval=0)
    rebuilt = hub2._adapter_states[session.session_id]
    assert isinstance(rebuilt, WorkflowState)
    assert rebuilt.expected_next_speaker == triage.agent_id
    assert rebuilt.last_speaker_id == eng.agent_id
    assert rebuilt.creator_id == triage.agent_id

    # Triage closes via TerminateTarget — equivalent to calling
    # sessions(action="close"). For the deterministic verification we
    # use the hub directly.
    closed = await hub2.close_session(session.session_id, reason="triage_done")
    assert closed.state == SessionState.CLOSED
    assert closed.close_reason == "triage_done"

    await hub2.close()
