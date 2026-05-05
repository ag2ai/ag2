# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``Hub.hydrate()`` correctness at scale.

Verifies:

* All session metadata round-trips through ``DiskKnowledgeStore``.
* Every active session's WAL is re-folded through its adapter so the
  in-memory ``_adapter_states`` cache matches the on-disk truth.
* The capability index rebuilds from loaded resumes.
* Round-trip is deterministic: hydrating twice yields the same state.

Correctness benchmark, not a perf benchmark. Default: 100 sessions ×
100 envelopes (~10k total) so the test runs in <2s. Bump
``ENVELOPES_PER_SESSION`` locally to exercise larger sweeps.
"""

import asyncio

import pytest

from autogen.beta import Agent
from autogen.beta.knowledge import DiskKnowledgeStore
from autogen.beta.network import (
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)
from autogen.beta.network.adapters.conversation import (
    CONVERSATION_TYPE,
    ConversationState,
)
from autogen.beta.network.adapters.discussion import (
    DISCUSSION_TYPE,
    ORDERING_ROUND_ROBIN,
    DiscussionState,
)
from autogen.beta.network.envelope import EV_TEXT
from autogen.beta.network.session import SessionState
from autogen.beta.testing import TestConfig

# Scale chosen for unit-run speed: 100 sessions × 100 envelopes ≈
# 10k envelopes, ~1s populate, instant hydrate. Bump
# ``ENVELOPES_PER_SESSION`` locally for larger sweeps. Hydrate cost
# scales linearly with envelope count and is dwarfed by populate
# (write throughput) in any realistic setup.
SESSIONS = 100
ENVELOPES_PER_SESSION = 100


def _agent(name: str) -> Agent:
    return Agent(name=name, config=TestConfig())


@pytest.mark.asyncio
async def test_hydrate_round_trips_many_sessions(tmp_path) -> None:
    """Populate disk via a live hub, tear down, re-open, verify state matches."""
    n_sessions = SESSIONS
    n_envelopes = ENVELOPES_PER_SESSION

    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link1 = LocalLink(hub1)

    alice_hc = HubClient(link1, hub=hub1)
    bob_hc = HubClient(link1, hub=hub1)
    alice = await alice_hc.register(
        _agent("alice"),
        Passport(name="alice"),
        Resume(claimed_capabilities=["analysis"]),
    )
    bob = await bob_hc.register(
        _agent("bob"),
        Passport(name="bob"),
        Resume(claimed_capabilities=["coding"]),
    )

    # Build conversation sessions in parallel batches; each session
    # fills its WAL with N alternating EV_TEXT envelopes.
    expected_states: dict[str, dict] = {}
    sessions = []
    for _ in range(n_sessions):
        session = await alice.open(type=CONVERSATION_TYPE, target=bob.agent_id)
        sessions.append(session)

    # Fill each session's WAL.
    for session in sessions:
        for i in range(n_envelopes):
            sender = alice if (i % 2 == 0) else bob
            envelope = Envelope(
                session_id=session.session_id,
                sender_id=sender.agent_id,
                audience=None,
                event_type=EV_TEXT,
                event_data={"text": f"msg-{i}"},
            )
            await hub1.post_envelope(envelope)
        # Snapshot expected state.
        cached = hub1._adapter_states[session.session_id]
        expected_states[session.session_id] = {
            "turn_count": cached.turn_count,
            "last_speaker_id": cached.last_speaker_id,
        }

    await alice_hc.close()
    await bob_hc.close()
    await hub1.close()

    # Round 1: re-open and verify everything matches.
    store2 = DiskKnowledgeStore(str(tmp_path))
    hub2 = await Hub.open(store2, ttl_sweep_interval=0, expectation_sweep_interval=0)

    # Identities preserved.
    assert alice.agent_id in hub2._passports
    assert bob.agent_id in hub2._passports

    # Capability index rebuilt from resumes.
    assert hub2.agents_with_capability("analysis") == [alice.agent_id]
    assert hub2.agents_with_capability("coding") == [bob.agent_id]

    # Every session round-tripped.
    for session in sessions:
        meta = await hub2.get_session(session.session_id)
        assert meta.manifest.type == CONVERSATION_TYPE
        assert meta.state == SessionState.ACTIVE

        cached = hub2._adapter_states[session.session_id]
        assert isinstance(cached, ConversationState)
        assert cached.turn_count == expected_states[session.session_id]["turn_count"]
        assert cached.last_speaker_id == expected_states[session.session_id]["last_speaker_id"]

    # Round 2: hydrating twice from the same store is idempotent.
    await hub2.hydrate()
    for session in sessions:
        cached = hub2._adapter_states[session.session_id]
        assert cached.turn_count == expected_states[session.session_id]["turn_count"]

    await hub2.close()


@pytest.mark.asyncio
async def test_hydrate_refolds_discussion_round_robin_state(tmp_path) -> None:
    """Multi-party discussion rotation survives hub restart deterministically."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link1 = LocalLink(hub1)

    names = ["alice", "bob", "carol", "dave", "erin"]
    clients = []
    for name in names:
        hc = HubClient(link1, hub=hub1)
        client = await hc.register(_agent(name), Passport(name=name), Resume())
        clients.append(client)
    alice = clients[0]

    session = await alice.open(
        type=DISCUSSION_TYPE,
        target=[c.agent_id for c in clients[1:]],
        knobs={"ordering": ORDERING_ROUND_ROBIN},
    )

    # 50 turns through the rotation (10 full cycles of 5 speakers).
    for i in range(50):
        sender = clients[i % 5]
        envelope = Envelope(
            session_id=session.session_id,
            sender_id=sender.agent_id,
            audience=None,
            event_type=EV_TEXT,
            event_data={"text": f"turn-{i}"},
        )
        await hub1.post_envelope(envelope)

    expected = hub1._adapter_states[session.session_id]
    expected_turn_count = expected.turn_count
    expected_next = expected.expected_next_speaker
    expected_last = expected.last_speaker_id

    for hc in [c._hub_client for c in clients]:
        await hc.close()
    await hub1.close()

    store2 = DiskKnowledgeStore(str(tmp_path))
    hub2 = await Hub.open(store2, ttl_sweep_interval=0, expectation_sweep_interval=0)

    rebuilt = hub2._adapter_states[session.session_id]
    assert isinstance(rebuilt, DiscussionState)
    assert rebuilt.turn_count == expected_turn_count
    assert rebuilt.expected_next_speaker == expected_next
    assert rebuilt.last_speaker_id == expected_last
    assert rebuilt.participant_order == [c.agent_id for c in clients]

    await hub2.close()
