# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Channel-scoped context variables on ``WorkflowState``.

Pins the contract that ``EV_CONTEXT_SET`` envelopes:

* mutate ``WorkflowState.context_vars`` via ``fold`` (set + delete),
* survive ``Hub.hydrate()`` because they live on the WAL,
* don't advance ``turn_count`` (non-substantive),
* may be sent by any participant regardless of turn order (loose semantics),
* drive transitions via the new ``ContextEquals`` condition.
"""

import pytest

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_CLOSED,
    EV_CONTEXT_SET,
    WORKFLOW_TYPE,
    AgentTarget,
    ContextEquals,
    Envelope,
    FromSpeaker,
    Hub,
    TerminateTarget,
    Transition,
    TransitionGraph,
)
from ag2.testing import TestConfig


def _agent(name: str, *replies: str) -> Agent:
    return Agent(name=name, config=TestConfig(*replies))


async def _post_context_set(hub: Hub, *, sender: str, channel_id: str, **event_data) -> None:
    """Helper: post an EV_CONTEXT_SET envelope through the hub."""
    envelope = Envelope(
        channel_id=channel_id,
        sender_id=sender,
        audience=None,
        event_type=EV_CONTEXT_SET,
        event_data=event_data,
    )
    await hub.post_envelope(envelope)


@pytest.mark.asyncio
class TestContextVars:
    """``ag2.context.set`` envelopes mutate ``WorkflowState.context_vars``."""

    async def test_set_persists_across_fold(self) -> None:
        """One EV_CONTEXT_SET envelope merges into context_vars."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)

        alice = await hub.register(_agent("alice"))
        bob = await hub.register(_agent("bob"))

        graph = TransitionGraph(
            initial_speaker=alice.agent_id,
            transitions=[
                Transition(when=FromSpeaker(alice.agent_id), then=AgentTarget(bob.agent_id)),
            ],
            default_target=TerminateTarget(reason="done"),
            max_turns=10,
        )
        channel = await alice.open(
            type=WORKFLOW_TYPE,
            target=[bob.agent_id],
            knobs={"graph": graph.to_dict()},
        )

        await _post_context_set(
            hub,
            sender=alice.agent_id,
            channel_id=channel.channel_id,
            set={"priority": "high", "ticket_id": "T-481"},
        )

        state = hub._adapter_states[channel.channel_id]
        assert state.context_vars == {"priority": "high", "ticket_id": "T-481"}

        await hub.close()

    async def test_set_then_delete(self) -> None:
        """Subsequent envelopes can both set and remove keys."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)

        alice = await hub.register(_agent("alice"))
        bob = await hub.register(_agent("bob"))

        graph = TransitionGraph(
            initial_speaker=alice.agent_id,
            transitions=[
                Transition(when=FromSpeaker(alice.agent_id), then=AgentTarget(bob.agent_id)),
            ],
            default_target=TerminateTarget(reason="done"),
            max_turns=10,
        )
        channel = await alice.open(
            type=WORKFLOW_TYPE,
            target=[bob.agent_id],
            knobs={"graph": graph.to_dict()},
        )

        await _post_context_set(
            hub,
            sender=alice.agent_id,
            channel_id=channel.channel_id,
            set={"priority": "high", "temporary": "1"},
        )
        await _post_context_set(
            hub,
            sender=alice.agent_id,
            channel_id=channel.channel_id,
            delete=["temporary"],
            set={"resolved": True},
        )

        state = hub._adapter_states[channel.channel_id]
        assert state.context_vars == {"priority": "high", "resolved": True}

        await hub.close()

    async def test_loose_semantics_any_participant(self) -> None:
        """Non-current speaker can still post EV_CONTEXT_SET."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)

        alice = await hub.register(_agent("alice"))
        bob = await hub.register(_agent("bob"))

        # alice is initial speaker — bob is NOT expected_next_speaker.
        graph = TransitionGraph(
            initial_speaker=alice.agent_id,
            transitions=[
                Transition(when=FromSpeaker(alice.agent_id), then=AgentTarget(bob.agent_id)),
            ],
            default_target=TerminateTarget(reason="done"),
            max_turns=10,
        )
        channel = await alice.open(
            type=WORKFLOW_TYPE,
            target=[bob.agent_id],
            knobs={"graph": graph.to_dict()},
        )

        # bob writes context out-of-turn — should succeed.
        await _post_context_set(
            hub,
            sender=bob.agent_id,
            channel_id=channel.channel_id,
            set={"observer_flag": True},
        )

        state = hub._adapter_states[channel.channel_id]
        assert state.context_vars == {"observer_flag": True}

        await hub.close()

    async def test_context_equals_drives_transition(self) -> None:
        """ContextEquals condition routes to a different agent based on
        a value set via EV_CONTEXT_SET."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)
        triage = await hub.register(_agent("triage", "triaged"))
        security = await hub.register(_agent("security", "handled"))
        legal = await hub.register(_agent("legal", "legal-reviewed"))

        # Terminate transitions are listed first so the post-handoff
        # speaker hits them before the ContextEquals rule (which would
        # otherwise re-fire and loop). Then the ContextEquals override
        # for triage's turn, then the default triage→legal route.
        graph = TransitionGraph(
            initial_speaker=triage.agent_id,
            transitions=[
                Transition(
                    when=FromSpeaker(security.agent_id),
                    then=TerminateTarget(reason="security_done"),
                ),
                Transition(
                    when=FromSpeaker(legal.agent_id),
                    then=TerminateTarget(reason="legal_done"),
                ),
                Transition(
                    when=ContextEquals(key="route", value="security"),
                    then=AgentTarget(security.agent_id),
                ),
                Transition(
                    when=FromSpeaker(triage.agent_id),
                    then=AgentTarget(legal.agent_id),
                ),
            ],
            default_target=TerminateTarget(reason="fall_through"),
            max_turns=10,
        )

        channel = await triage.open(
            type=WORKFLOW_TYPE,
            target=[security.agent_id, legal.agent_id],
            knobs={"graph": graph.to_dict()},
        )

        # Triage writes route=security BEFORE its own substantive turn
        # so the condition fires when triage's text is folded.
        await _post_context_set(
            hub,
            sender=triage.agent_id,
            channel_id=channel.channel_id,
            set={"route": "security"},
        )
        await channel.send("kickoff")

        close_env = await triage.wait_for_channel_event(
            channel_id=channel.channel_id,
            predicate=lambda e: e.event_type == EV_CHANNEL_CLOSED,
            timeout=5.0,
        )
        # Routed to security, not legal.
        assert close_env.event_data.get("reason") == "security_done"

        await hub.close()

    async def test_hydrate_replays_context_vars(self) -> None:
        """Closing and re-opening a hub against the same store
        reconstructs ``context_vars`` via WAL replay."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)
        alice = await hub.register(_agent("alice"))
        bob = await hub.register(_agent("bob"))

        graph = TransitionGraph(
            initial_speaker=alice.agent_id,
            transitions=[
                Transition(when=FromSpeaker(alice.agent_id), then=AgentTarget(bob.agent_id)),
            ],
            default_target=TerminateTarget(reason="done"),
            max_turns=10,
        )
        channel = await alice.open(
            type=WORKFLOW_TYPE,
            target=[bob.agent_id],
            knobs={"graph": graph.to_dict()},
        )
        channel_id = channel.channel_id

        await _post_context_set(
            hub,
            sender=alice.agent_id,
            channel_id=channel_id,
            set={"k1": "v1", "k2": "v2"},
        )
        await _post_context_set(
            hub,
            sender=alice.agent_id,
            channel_id=channel_id,
            delete=["k1"],
            set={"k3": "v3"},
        )

        # Snapshot before close.
        before = dict(hub._adapter_states[channel_id].context_vars)

        await hub.close()

        # Re-open against the same store and re-fold the WAL.
        hub2 = await Hub.open(store, ttl_sweep_interval=0)
        try:
            after = dict(hub2._adapter_states[channel_id].context_vars)
            assert after == before
            assert after == {"k2": "v2", "k3": "v3"}
        finally:
            await hub2.close()

    async def test_context_set_does_not_advance_turn(self) -> None:
        """EV_CONTEXT_SET must not increment turn_count or rotate the speaker."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)
        alice = await hub.register(_agent("alice"))
        bob = await hub.register(_agent("bob"))

        graph = TransitionGraph(
            initial_speaker=alice.agent_id,
            transitions=[
                Transition(when=FromSpeaker(alice.agent_id), then=AgentTarget(bob.agent_id)),
            ],
            default_target=TerminateTarget(reason="done"),
            max_turns=10,
        )
        channel = await alice.open(
            type=WORKFLOW_TYPE,
            target=[bob.agent_id],
            knobs={"graph": graph.to_dict()},
        )

        before_turn = hub._adapter_states[channel.channel_id].turn_count
        before_speaker = hub._adapter_states[channel.channel_id].expected_next_speaker

        await _post_context_set(
            hub,
            sender=alice.agent_id,
            channel_id=channel.channel_id,
            set={"k": "v"},
        )

        after = hub._adapter_states[channel.channel_id]
        assert after.turn_count == before_turn
        assert after.expected_next_speaker == before_speaker

        await hub.close()
