# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``WorkflowAdapter`` close-reason precedence.

Pins the rule that when both ``expected_next_speaker is None`` (a
``TerminateTarget`` or ``default_target`` resolved into the fold) and
``turn_count >= max_turns`` are true on the same envelope, the more
specific graph-emitted reason wins over the generic ``"max_turns"``
fallback.

Without this precedence ``TransitionGraph.sequence([a, b, c])`` would
close with ``"max_turns"`` instead of the ``"sequence_complete"``
reason its ``default_target`` declares — because the factory sets
``max_turns=len(steps)`` so both fire on the last step.
"""

import pytest

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_CLOSED,
    WORKFLOW_TYPE,
    AgentTarget,
    FromSpeaker,
    Hub,
    TerminateTarget,
    Transition,
    TransitionGraph,
)
from ag2.testing import TestConfig


def _agent(name: str, *replies: str) -> Agent:
    """TestConfig agent that emits ``replies`` in order on successive ``Agent.ask`` calls."""
    return Agent(name=name, config=TestConfig(*replies))


@pytest.mark.asyncio
class TestWorkflowTerminationReason:
    """Close-reason precedence between graph termination and max_turns."""

    async def test_sequence_closes_with_sequence_complete(self) -> None:
        """``sequence([a, b, c])`` sets ``max_turns=3`` and a
        ``TerminateTarget("sequence_complete")`` default. After step c
        both fire on the same envelope; the specific reason must win."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)

        # alice's first turn is the kickoff via channel.send (no Agent.ask).
        # bob and carol each speak exactly once via the default handler.
        alice = await hub.register(_agent("alice"))
        bob = await hub.register(_agent("bob", "bob-reply"))
        carol = await hub.register(_agent("carol", "carol-reply"))

        graph = TransitionGraph.sequence([alice.agent_id, bob.agent_id, carol.agent_id])

        channel = await alice.open(
            type=WORKFLOW_TYPE,
            target=[bob.agent_id, carol.agent_id],
            knobs={"graph": graph.to_dict()},
        )
        await channel.send("kickoff")

        close_env = await alice.wait_for_channel_event(
            channel_id=channel.channel_id,
            predicate=lambda e: e.event_type == EV_CHANNEL_CLOSED,
            timeout=5.0,
        )
        assert close_env.event_data.get("reason") == "sequence_complete"

        metadata = await hub.get_channel(channel.channel_id)
        assert metadata.close_reason == "sequence_complete"

        await hub.close()

    async def test_round_robin_closes_with_max_turns(self) -> None:
        """``round_robin`` uses ``Always() -> RoundRobinTarget()`` so
        ``expected_next_speaker`` is always set after each turn —
        ``max_turns`` is the only path to close."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)

        # 6 turns total: kickoff (alice) + 5 Agent.ask calls.
        # alice speaks turn 4 (1 reply), bob 2/5 (2 replies), carol 3/6 (2 replies).
        alice = await hub.register(_agent("alice", "alice-2"))
        bob = await hub.register(_agent("bob", "bob-1", "bob-2"))
        carol = await hub.register(_agent("carol", "carol-1", "carol-2"))

        graph = TransitionGraph.round_robin(
            participants=[alice.agent_id, bob.agent_id, carol.agent_id],
            max_turns=6,
        )

        channel = await alice.open(
            type=WORKFLOW_TYPE,
            target=[bob.agent_id, carol.agent_id],
            knobs={"graph": graph.to_dict()},
        )
        await channel.send("kickoff")

        close_env = await alice.wait_for_channel_event(
            channel_id=channel.channel_id,
            predicate=lambda e: e.event_type == EV_CHANNEL_CLOSED,
            timeout=5.0,
        )
        assert close_env.event_data.get("reason") == "max_turns"

        await hub.close()

    async def test_explicit_terminate_target_wins_over_max_turns(self) -> None:
        """A custom graph with both an explicit ``TerminateTarget`` and
        ``max_turns`` — the terminate's reason must surface, not
        ``"max_turns"``."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)

        alice = await hub.register(_agent("alice"))
        bob = await hub.register(_agent("bob", "bob-reply"))
        carol = await hub.register(_agent("carol", "carol-reply"))

        # Pipeline alice → bob → carol → terminate("custom_done");
        # max_turns=3 so the cap also fires on carol's turn. Must prefer
        # the explicit terminate reason.
        graph = TransitionGraph(
            initial_speaker=alice.agent_id,
            transitions=[
                Transition(when=FromSpeaker(alice.agent_id), then=AgentTarget(bob.agent_id)),
                Transition(when=FromSpeaker(bob.agent_id), then=AgentTarget(carol.agent_id)),
                Transition(
                    when=FromSpeaker(carol.agent_id),
                    then=TerminateTarget(reason="custom_done"),
                ),
            ],
            default_target=TerminateTarget(reason="fall_through"),
            max_turns=3,
        )

        channel = await alice.open(
            type=WORKFLOW_TYPE,
            target=[bob.agent_id, carol.agent_id],
            knobs={"graph": graph.to_dict()},
        )
        await channel.send("kickoff")

        close_env = await alice.wait_for_channel_event(
            channel_id=channel.channel_id,
            predicate=lambda e: e.event_type == EV_CHANNEL_CLOSED,
            timeout=5.0,
        )
        assert close_env.event_data.get("reason") == "custom_done"

        await hub.close()

    async def test_max_turns_wins_when_no_terminate_fires(self) -> None:
        """A custom graph that never resolves ``TerminateTarget`` —
        ``max_turns`` is the only thing left to fire."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)

        # 4 turns: kickoff (alice) + 3 Agent.ask. alice and bob alternate.
        alice = await hub.register(_agent("alice", "alice-2"))
        bob = await hub.register(_agent("bob", "bob-1", "bob-2"))

        # Two-step ping-pong, no TerminateTarget anywhere; default_target
        # could only fire if a turn produced no FromSpeaker match, which
        # the alternation prevents within max_turns.
        graph = TransitionGraph(
            initial_speaker=alice.agent_id,
            transitions=[
                Transition(when=FromSpeaker(alice.agent_id), then=AgentTarget(bob.agent_id)),
                Transition(when=FromSpeaker(bob.agent_id), then=AgentTarget(alice.agent_id)),
            ],
            default_target=AgentTarget(alice.agent_id),
            max_turns=4,
        )

        channel = await alice.open(
            type=WORKFLOW_TYPE,
            target=[bob.agent_id],
            knobs={"graph": graph.to_dict()},
        )
        await channel.send("kickoff")

        close_env = await alice.wait_for_channel_event(
            channel_id=channel.channel_id,
            predicate=lambda e: e.event_type == EV_CHANNEL_CLOSED,
            timeout=5.0,
        )
        assert close_env.event_data.get("reason") == "max_turns"

        await hub.close()
