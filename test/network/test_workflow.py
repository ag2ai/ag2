# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow adapter + transition vocabulary tests.

Two layers:

* **Transitions vocabulary** (unit) — every target / condition resolves
  correctly; ``TransitionGraph.to_dict()`` + ``loads()`` round-trip
  through the named registry.
* **WorkflowAdapter** (integration) — four orchestration patterns:
    1. Round-robin via ``WorkflowAdapter`` (cycles correctly).
    2. Sequential pipeline (each step transitions to the next).
    3. Swarm with tool-driven handoffs + revert-to-initiator.
    4. Manager-as-initiator (auto-pattern equivalent).
  Each test verifies ``Hub.hydrate()`` re-folds the WAL through the
  adapter and recovers ``expected_next_speaker`` deterministically.
"""

import pytest

from ag2 import Agent
from ag2.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from ag2.network import (
    EV_PACKET,
    EV_TEXT,
    Envelope,
    Hub,
)
from ag2.network.adapters.workflow import (
    WORKFLOW_TYPE,
    WorkflowAdapter,
    WorkflowState,
)
from ag2.network.channel import (
    ChannelState,
)
from ag2.network.errors import ProtocolError
from ag2.network.transitions import (
    AgentTarget,
    Always,
    FromSpeaker,
    RevertToInitiatorTarget,
    RoundRobinTarget,
    StayTarget,
    TerminateTarget,
    ToolCalled,
    Transition,
    TransitionDecision,
    TransitionGraph,
    WorkflowGraphError,
    register_target,
)
from ag2.testing import TestConfig


def _agent(name: str, *events: object) -> Agent:
    return Agent(name=name, config=TestConfig(*events))


def _state(
    *,
    order: list[str],
    last: str | None = None,
    creator: str = "alice",
    turn_count: int = 0,
) -> WorkflowState:
    return WorkflowState(
        participant_order=order,
        last_speaker_id=last,
        creator_id=creator,
        turn_count=turn_count,
    )


def _routing_packet(tool: str, *, reason: str = "") -> dict:
    """Construct an ``EV_PACKET`` payload that simulates a tool-driven
    handoff (the framework normally builds this from the agent's local
    ``ToolCallEvent``s at round-end)."""
    return {
        "routing": {"kind": "handoff", "tool": tool, "reason": reason},
        "context_updates": {"set": {}, "delete": []},
        "body": "",
    }


def _envelope(sender: str, *, event_type: str = EV_TEXT, tool: str = "") -> Envelope:
    if event_type == EV_TEXT:
        data: dict = {"text": "x"}
    elif event_type == EV_PACKET:
        data = {
            "routing": {"kind": "handoff", "tool": tool, "reason": ""},
            "context_updates": {"set": {}, "delete": []},
            "body": "",
        }
    else:
        data = {"tool": tool}
    return Envelope(
        envelope_id=f"env-{sender}",
        channel_id="s1",
        sender_id=sender,
        audience=None,
        event_type=event_type,
        event_data=data,
    )


# ── TransitionTarget unit tests ─────────────────────────────────────────────


class TestBuiltInTargets:
    def test_agent_target_resolves_to_named_agent(self) -> None:
        decision = AgentTarget("bob").resolve(_state(order=["alice", "bob", "carol"]), _envelope("alice"))
        assert decision == TransitionDecision(next_speaker="bob")

    def test_round_robin_advances_through_order(self) -> None:
        order = ["alice", "bob", "carol"]
        target = RoundRobinTarget()
        # alice just spoke → bob next
        d = target.resolve(_state(order=order, last="alice"), _envelope("alice"))
        assert d.next_speaker == "bob"
        # carol just spoke → alice next (wrap)
        d = target.resolve(_state(order=order, last="carol"), _envelope("carol"))
        assert d.next_speaker == "alice"

    def test_round_robin_with_no_participants_terminates(self) -> None:
        d = RoundRobinTarget().resolve(_state(order=[]), _envelope("alice"))
        assert d.next_speaker is None
        assert d.close_reason == "no_participants"

    def test_stay_target_keeps_current_speaker(self) -> None:
        d = StayTarget().resolve(_state(order=["alice", "bob"], last="bob"), _envelope("bob"))
        assert d.next_speaker == "bob"

    def test_revert_to_initiator(self) -> None:
        d = RevertToInitiatorTarget().resolve(
            _state(order=["alice", "bob"], creator="alice", last="bob"),
            _envelope("bob"),
        )
        assert d.next_speaker == "alice"

    def test_terminate_carries_reason(self) -> None:
        d = TerminateTarget("done").resolve(_state(order=["alice", "bob"]), _envelope("alice"))
        assert d.next_speaker is None
        assert d.close_reason == "done"


class TestBuiltInConditions:
    def test_always_fires(self) -> None:
        assert Always().evaluate(_state(order=["alice"]), _envelope("alice")) is True

    def test_from_speaker_matches_sender(self) -> None:
        assert FromSpeaker("bob").evaluate(_state(order=["alice", "bob"]), _envelope("bob")) is True
        assert FromSpeaker("bob").evaluate(_state(order=["alice", "bob"]), _envelope("alice")) is False

    def test_tool_called_matches_routing_tool_in_packet(self) -> None:
        env = _envelope("alice", event_type=EV_PACKET, tool="transfer_to_eng")
        assert ToolCalled("transfer_to_eng").evaluate(_state(order=["alice"]), env) is True
        assert ToolCalled("escalate").evaluate(_state(order=["alice"]), env) is False

    def test_tool_called_ignores_non_packet_envelopes(self) -> None:
        text_env = _envelope("alice")
        assert ToolCalled("transfer_to_eng").evaluate(_state(order=["alice"]), text_env) is False


# ── TransitionGraph serialization ───────────────────────────────────────────


class TestTransitionGraphSerialization:
    def test_round_trip_via_to_dict(self) -> None:
        graph = TransitionGraph(
            initial_speaker="alice",
            transitions=[
                Transition(when=Always(), then=RoundRobinTarget(), priority=1),
                Transition(when=ToolCalled("escalate"), then=AgentTarget("bob"), priority=0),
                Transition(
                    when=FromSpeaker("bob"),
                    then=RevertToInitiatorTarget(),
                ),
            ],
            default_target=TerminateTarget("done"),
            max_turns=10,
        )
        restored = TransitionGraph.loads(graph.to_dict())
        assert restored.initial_speaker == "alice"
        assert restored.max_turns == 10
        assert restored.default_target == TerminateTarget("done")
        assert len(restored.transitions) == 3
        # Priorities preserved.
        assert restored.transitions[0].priority == 1
        assert restored.transitions[1].priority == 0
        # Tool name preserved on the ToolCalled condition.
        assert restored.transitions[1].when == ToolCalled("escalate")

    def test_round_trip_via_dumps_string(self) -> None:
        graph = TransitionGraph.round_robin(["a", "b", "c"], max_turns=5)
        restored = TransitionGraph.loads(graph.dumps())
        assert restored.initial_speaker == "a"
        assert restored.max_turns == 5

    def test_unknown_target_name_raises(self) -> None:
        bad = {
            "initial_speaker": "alice",
            "transitions": [],
            "default_target": {"name": "unknown_target", "args": {}},
            "max_turns": None,
        }
        with pytest.raises(WorkflowGraphError, match="no transition target"):
            TransitionGraph.loads(bad)

    def test_unknown_condition_name_raises(self) -> None:
        bad = {
            "initial_speaker": "alice",
            "transitions": [
                {
                    "when": {"name": "unknown_when", "args": {}},
                    "then": {"name": "agent", "args": {"agent_id": "bob"}},
                    "priority": 0,
                }
            ],
            "default_target": {"name": "terminate", "args": {"reason": "x"}},
            "max_turns": None,
        }
        with pytest.raises(WorkflowGraphError, match="no transition condition"):
            TransitionGraph.loads(bad)


class TestRegistry:
    def test_register_custom_target_extends_serialization(self) -> None:
        from dataclasses import dataclass
        from typing import ClassVar

        @dataclass(slots=True)
        class WhenIdle:
            seconds: int
            name: ClassVar[str] = "when_idle_target_test"

            def resolve(self, state, envelope):
                return TransitionDecision(next_speaker=None, close_reason="idle")

        register_target(WhenIdle)
        graph = TransitionGraph(
            initial_speaker="alice",
            transitions=[],
            default_target=WhenIdle(seconds=42),
        )
        restored = TransitionGraph.loads(graph.to_dict())
        assert restored.default_target == WhenIdle(seconds=42)


class TestGraphFactories:
    def test_round_robin_factory(self) -> None:
        graph = TransitionGraph.round_robin(["a", "b", "c"], max_turns=6)
        assert graph.initial_speaker == "a"
        assert graph.max_turns == 6
        assert graph.transitions == [Transition(when=Always(), then=RoundRobinTarget())]

    def test_sequence_factory(self) -> None:
        graph = TransitionGraph.sequence(["a", "b", "c"])
        assert graph.initial_speaker == "a"
        # 2 transitions: a → b, b → c. c is terminal via max_turns.
        assert len(graph.transitions) == 2
        assert graph.transitions[0].when == FromSpeaker("a")
        assert graph.transitions[0].then == AgentTarget("b")
        assert graph.transitions[1].when == FromSpeaker("b")
        assert graph.transitions[1].then == AgentTarget("c")
        assert graph.max_turns == 3


# ── WorkflowAdapter integration tests ───────────────────────────────────────


@pytest.mark.asyncio
async def test_default_workflow_adapter_registered_on_open() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    assert isinstance(hub._adapters.get((WORKFLOW_TYPE, 1)), WorkflowAdapter)
    await hub.close()


@pytest.mark.asyncio
async def test_workflow_round_robin_advances_through_participants() -> None:
    """3-agent round_robin via WorkflowAdapter: alice → bob → carol → alice."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))
    carol = await hub.register(_agent("carol"))

    graph = TransitionGraph.round_robin([alice.agent_id, bob.agent_id, carol.agent_id])
    channel = await alice.open(
        type=WORKFLOW_TYPE,
        target=[bob.agent_id, carol.agent_id],
        knobs={"graph": graph.to_dict()},
    )
    assert channel.state == ChannelState.ACTIVE

    state = hub._adapter_states[channel.channel_id]
    assert isinstance(state, WorkflowState)
    assert state.expected_next_speaker == alice.agent_id

    # Manual sends in turn order.
    for sender_client in [alice, bob, carol]:
        env = Envelope(
            channel_id=channel.channel_id,
            sender_id=sender_client.agent_id,
            audience=None,
            event_type=EV_TEXT,
            event_data={"text": f"turn from {sender_client.agent_id}"},
        )
        await hub.post_envelope(env)

    state = hub._adapter_states[channel.channel_id]
    assert state.expected_next_speaker == alice.agent_id  # cycled back
    assert state.turn_count == 3

    # Out-of-turn send is rejected.
    bad = Envelope(
        channel_id=channel.channel_id,
        sender_id=bob.agent_id,  # alice is next
        audience=None,
        event_type=EV_TEXT,
        event_data={"text": "out of turn"},
    )
    with pytest.raises(ProtocolError, match="expects"):
        await hub.post_envelope(bad)

    await hub.close()


@pytest.mark.asyncio
async def test_workflow_sequence_pipeline_terminates_after_last_step() -> None:
    """Sequential pipeline a → b → c via TransitionGraph.sequence."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))
    carol = await hub.register(_agent("carol"))

    graph = TransitionGraph.sequence([alice.agent_id, bob.agent_id, carol.agent_id])
    channel = await alice.open(
        type=WORKFLOW_TYPE,
        target=[bob.agent_id, carol.agent_id],
        knobs={"graph": graph.to_dict()},
    )

    # alice, bob, carol each post once. After carol's post, the
    # sequence factory's TerminateTarget("sequence_complete") fires
    # and closes the channel.
    for sender_client in [alice, bob, carol]:
        env = Envelope(
            channel_id=channel.channel_id,
            sender_id=sender_client.agent_id,
            audience=None,
            event_type=EV_TEXT,
            event_data={"text": f"turn from {sender_client.agent_id}"},
        )
        await hub.post_envelope(env)

    # Wait briefly for the close envelope to be dispatched.
    final = await hub.get_channel(channel.channel_id)
    assert final.state == ChannelState.CLOSED
    assert final.close_reason == "sequence_complete"

    await hub.close()


@pytest.mark.asyncio
async def test_workflow_swarm_with_tool_handoff_and_revert() -> None:
    """Swarm: triage hands off to eng via ToolCalled, eng replies, control
    reverts to triage via FromSpeaker(eng) → RevertToInitiatorTarget."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    triage = await hub.register(_agent("triage"))
    eng = await hub.register(_agent("eng"))

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
        max_turns=4,
    )
    channel = await triage.open(
        type=WORKFLOW_TYPE,
        target=[eng.agent_id],
        knobs={"graph": graph.to_dict()},
    )

    # 1. triage emits the routing packet (simulating LLM tool call →
    # framework-built EV_PACKET with routing.tool set).
    handoff_env = Envelope(
        channel_id=channel.channel_id,
        sender_id=triage.agent_id,
        audience=None,
        event_type=EV_PACKET,
        event_data={
            "routing": {
                "kind": "handoff",
                "tool": "transfer_to_eng",
                "reason": "needs eng review",
            },
            "context_updates": {"set": {}, "delete": []},
            "body": "",
        },
    )
    await hub.post_envelope(handoff_env)
    state = hub._adapter_states[channel.channel_id]
    assert state.expected_next_speaker == eng.agent_id

    # 2. eng replies with text.
    eng_env = Envelope(
        channel_id=channel.channel_id,
        sender_id=eng.agent_id,
        audience=None,
        event_type=EV_TEXT,
        event_data={"text": "eng analysis: looks good"},
    )
    await hub.post_envelope(eng_env)
    state = hub._adapter_states[channel.channel_id]
    # FromSpeaker(eng) fires → revert to initiator (triage).
    assert state.expected_next_speaker == triage.agent_id

    # 3. triage closes via the explicit close channel API (mirrors the
    # exit criterion's TerminateTarget, but we use close_channel for the
    # deterministic test). Alternatively triage could send a "done"
    # handoff envelope routed to TerminateTarget in a richer graph.
    closed = await hub.close_channel(channel.channel_id, reason="triage_done")
    assert closed.state == ChannelState.CLOSED
    assert closed.close_reason == "triage_done"

    await hub.close()


@pytest.mark.asyncio
async def test_workflow_finish_routing_closes_channel() -> None:
    """A ``Finish`` typed return surfaces on the packet as
    ``routing.kind == "finish"``; ``fold`` then sets the next speaker
    to ``None`` and ``on_accepted`` closes the channel using the
    finish's ``reason`` as the close reason."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    graph = TransitionGraph(
        initial_speaker=alice.agent_id,
        transitions=[
            Transition(when=FromSpeaker(alice.agent_id), then=AgentTarget(bob.agent_id)),
            Transition(when=FromSpeaker(bob.agent_id), then=AgentTarget(alice.agent_id)),
        ],
        default_target=TerminateTarget(reason="default"),
        max_turns=10,
    )
    channel = await alice.open(
        type=WORKFLOW_TYPE,
        target=[bob.agent_id],
        knobs={"graph": graph.to_dict()},
    )

    # Alice emits a finish-routed packet — the typed-return payload as
    # the framework would construct it from a ``Finish`` instance.
    finish_env = Envelope(
        channel_id=channel.channel_id,
        sender_id=alice.agent_id,
        audience=None,
        event_type=EV_PACKET,
        event_data={
            "routing": {
                "kind": "finish",
                "tool": "finish",
                "reason": "all_done",
                "summary": "covered the agenda",
            },
            "context_updates": {"set": {}, "delete": []},
            "body": "",
        },
    )
    await hub.post_envelope(finish_env)

    # State should reflect termination intent: no next speaker, reason
    # propagated from Finish.
    state = hub._adapter_states[channel.channel_id]
    assert state.expected_next_speaker is None
    assert state.pending_close_reason == "all_done"

    # Channel itself should be closed with the finish reason.
    refreshed = await hub.get_channel(channel.channel_id)
    assert refreshed.state == ChannelState.CLOSED
    assert refreshed.close_reason == "all_done"

    await hub.close()


@pytest.mark.asyncio
async def test_workflow_finish_default_reason_uses_finished() -> None:
    """``Finish()`` with no args (or empty reason) folds to the
    ``"finished"`` default close reason."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    graph = TransitionGraph(
        initial_speaker=alice.agent_id,
        transitions=[Transition(when=Always(), then=AgentTarget(bob.agent_id))],
        default_target=TerminateTarget(reason="default"),
        max_turns=5,
    )
    channel = await alice.open(
        type=WORKFLOW_TYPE,
        target=[bob.agent_id],
        knobs={"graph": graph.to_dict()},
    )

    # routing.reason missing entirely — fold should fall back to "finished".
    finish_env = Envelope(
        channel_id=channel.channel_id,
        sender_id=alice.agent_id,
        audience=None,
        event_type=EV_PACKET,
        event_data={
            "routing": {"kind": "finish", "tool": "finish"},
            "context_updates": {"set": {}, "delete": []},
            "body": "",
        },
    )
    await hub.post_envelope(finish_env)

    refreshed = await hub.get_channel(channel.channel_id)
    assert refreshed.state == ChannelState.CLOSED
    assert refreshed.close_reason == "finished"

    await hub.close()


@pytest.mark.asyncio
async def test_workflow_manager_as_initiator_auto_pattern() -> None:
    """AutoPattern equivalent: manager is initiator + RevertToInitiator default.

    Manager directs by emitting handoff envelopes; respondents always
    revert to the manager."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    mgr = await hub.register(_agent("mgr"))
    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    graph = TransitionGraph(
        initial_speaker=mgr.agent_id,
        transitions=[
            Transition(when=ToolCalled("ask_alice"), then=AgentTarget(alice.agent_id)),
            Transition(when=ToolCalled("ask_bob"), then=AgentTarget(bob.agent_id)),
        ],
        default_target=RevertToInitiatorTarget(),
        max_turns=20,
    )
    channel = await mgr.open(
        type=WORKFLOW_TYPE,
        target=[alice.agent_id, bob.agent_id],
        knobs={"graph": graph.to_dict()},
    )

    # mgr asks alice → alice → revert to mgr → mgr asks bob → bob → revert to mgr
    sequence = [
        (mgr, EV_PACKET, _routing_packet("ask_alice")),
        (alice, EV_TEXT, {"text": "alice answers"}),
        (mgr, EV_PACKET, _routing_packet("ask_bob")),
        (bob, EV_TEXT, {"text": "bob answers"}),
    ]
    expected_next = [alice.agent_id, mgr.agent_id, bob.agent_id, mgr.agent_id]
    for (sender, et, ed), exp in zip(sequence, expected_next):
        env = Envelope(
            channel_id=channel.channel_id,
            sender_id=sender.agent_id,
            audience=None,
            event_type=et,
            event_data=ed,
        )
        await hub.post_envelope(env)
        state = hub._adapter_states[channel.channel_id]
        assert state.expected_next_speaker == exp, (
            f"after {sender.agent_id} sent {et}, expected_next was {state.expected_next_speaker}, expected {exp}"
        )

    await hub.close()


@pytest.mark.asyncio
async def test_workflow_hydrate_recovers_expected_next_speaker(tmp_path) -> None:
    """Hub restart mid-workflow recovers expected_next_speaker by re-folding
    the WAL through WorkflowAdapter.fold."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    triage = await hub1.register(_agent("triage"))
    eng = await hub1.register(_agent("eng"))

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
        default_target=TerminateTarget(),
        max_turns=10,
    )
    channel = await triage.open(
        type=WORKFLOW_TYPE,
        target=[eng.agent_id],
        knobs={"graph": graph.to_dict()},
    )
    handoff_env = Envelope(
        channel_id=channel.channel_id,
        sender_id=triage.agent_id,
        audience=None,
        event_type=EV_PACKET,
        event_data={
            "routing": {"kind": "handoff", "tool": "transfer_to_eng", "reason": ""},
            "context_updates": {"set": {}, "delete": []},
            "body": "",
        },
    )
    await hub1.post_envelope(handoff_env)
    pre_state = hub1._adapter_states[channel.channel_id]
    assert pre_state.expected_next_speaker == eng.agent_id

    await hub1.close()

    # Reopen a fresh hub against the same store.
    store2 = DiskKnowledgeStore(str(tmp_path))
    hub2 = await Hub.open(store2, ttl_sweep_interval=0, expectation_sweep_interval=0)

    refreshed = await hub2.get_channel(channel.channel_id)
    assert refreshed.manifest.type == WORKFLOW_TYPE
    assert refreshed.state == ChannelState.ACTIVE

    rebuilt = hub2._adapter_states[channel.channel_id]
    assert isinstance(rebuilt, WorkflowState)
    assert rebuilt.expected_next_speaker == eng.agent_id
    assert rebuilt.last_speaker_id == triage.agent_id
    assert rebuilt.turn_count == 1
    assert rebuilt.creator_id == triage.agent_id
    assert rebuilt.participant_order == [triage.agent_id, eng.agent_id]

    await hub2.close()


@pytest.mark.asyncio
async def test_workflow_validate_create_rejects_missing_graph() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    with pytest.raises(ProtocolError, match="graph"):
        await hub.create_channel(
            creator_id=alice.agent_id,
            manifest_type=WORKFLOW_TYPE,
            participants=[bob.agent_id],
            knobs={},
        )

    await hub.close()
