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
from pydantic import JsonValue

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_CLOSED,
    EV_CONTEXT_SET,
    EV_PACKET,
    WORKFLOW_TYPE,
    AgentTarget,
    ContextEquals,
    Envelope,
    FromSpeaker,
    Hub,
    ProtocolError,
    TerminateTarget,
    Transition,
    TransitionGraph,
    WorkflowState,
)
from ag2.network.hub import wal_path
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


async def _open_workflow(hub: Hub) -> tuple[str, str, str]:
    """Open an alice → bob workflow; return ``(channel_id, alice_id, bob_id)``."""
    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))
    graph = TransitionGraph(
        initial_speaker=alice.agent_id,
        transitions=[Transition(when=FromSpeaker(alice.agent_id), then=AgentTarget(bob.agent_id))],
        default_target=TerminateTarget(reason="done"),
        max_turns=10,
    )
    channel = await alice.open(
        type=WORKFLOW_TYPE,
        target=[bob.agent_id],
        knobs={"graph": graph.to_dict()},
    )
    return channel.channel_id, alice.agent_id, bob.agent_id


@pytest.mark.asyncio
class TestWorkflowEnvelopeShape:
    """The shape of ``EV_CONTEXT_SET`` / ``EV_PACKET`` data on a workflow channel."""

    @pytest.mark.parametrize(
        ("event_type", "event_data"),
        [
            pytest.param(EV_CONTEXT_SET, {"set": "oops"}, id="context-set-not-an-object"),
            pytest.param(EV_CONTEXT_SET, {"delete": "k1"}, id="context-delete-not-a-list"),
            pytest.param(EV_CONTEXT_SET, {"delete": [1]}, id="context-delete-not-strings"),
            pytest.param(EV_CONTEXT_SET, {"set": []}, id="context-set-empty-list"),
            pytest.param(EV_CONTEXT_SET, {"set": 0}, id="context-set-zero"),
            pytest.param(EV_CONTEXT_SET, {"delete": ""}, id="context-delete-empty-string"),
            pytest.param(EV_PACKET, {"body": "hi", "context_updates": {"set": ["k"]}}, id="packet-set-not-an-object"),
            pytest.param(EV_PACKET, {"body": "hi", "context_updates": {"delete": "k1"}}, id="packet-delete-not-a-list"),
            pytest.param(EV_PACKET, {"body": "hi", "context_updates": "oops"}, id="packet-updates-not-an-object"),
            pytest.param(EV_PACKET, {"body": "hi", "routing": "handoff"}, id="packet-routing-not-an-object"),
        ],
    )
    async def test_malformed_envelope_is_rejected_before_the_wal(
        self, event_type: str, event_data: dict[str, JsonValue]
    ) -> None:
        """A malformed context mutation or routing is a protocol error: it never
        reaches the WAL, so the channel state is untouched and the hub reopens."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)
        channel_id, alice_id, _ = await _open_workflow(hub)

        with pytest.raises(ProtocolError):
            await hub.post_envelope(
                Envelope(
                    channel_id=channel_id,
                    sender_id=alice_id,
                    audience=None,
                    event_type=event_type,
                    event_data=event_data,
                )
            )
        await hub.close()

        reopened = await Hub.open(store, ttl_sweep_interval=0)
        state = reopened.adapter_state(channel_id, WorkflowState)
        assert state is not None
        assert (state.context_vars, state.turn_count, state.expected_next_speaker) == ({}, 0, alice_id)
        await reopened.close()

    async def test_null_routing_and_context_updates_mean_no_change(self) -> None:
        """An explicit ``null`` is read like an absent field, for both packet fields."""
        hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0)
        channel_id, alice_id, bob_id = await _open_workflow(hub)

        await hub.post_envelope(
            Envelope(
                channel_id=channel_id,
                sender_id=alice_id,
                audience=None,
                event_type=EV_PACKET,
                event_data={"body": "hi", "routing": None, "context_updates": None},
            )
        )

        state = hub.adapter_state(channel_id, WorkflowState)
        assert state is not None
        assert (state.context_vars, state.turn_count, state.expected_next_speaker) == ({}, 1, bob_id)
        await hub.close()

    async def test_malformed_context_set_already_in_the_wal_is_skipped_on_replay(self) -> None:
        """A WAL written before ``validate_send`` checked the shape may hold a
        malformed ``EV_CONTEXT_SET``. Replaying it ignores that entry and keeps
        folding the rest, so the hub still opens."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)
        channel_id, alice_id, _ = await _open_workflow(hub)
        await _post_context_set(hub, sender=alice_id, channel_id=channel_id, set={"stage": "draft"})
        await hub.close()

        malformed = Envelope(
            channel_id=channel_id,
            sender_id=alice_id,
            audience=None,
            event_type=EV_CONTEXT_SET,
            event_data={"set": "oops"},
        )
        await store.append(wal_path(channel_id), malformed.to_json() + "\n")

        reopened = await Hub.open(store, ttl_sweep_interval=0)
        await _post_context_set(reopened, sender=alice_id, channel_id=channel_id, set={"reviewed": True})

        state = reopened.adapter_state(channel_id, WorkflowState)
        assert state is not None
        assert state.context_vars == {"stage": "draft", "reviewed": True}
        await reopened.close()
