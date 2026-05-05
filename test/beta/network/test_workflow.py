# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow adapter + transition vocabulary tests.

Three layers:

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
* **NetworkPlugin.register_workflow** — handoff tools materialised
  per ``ToolCalled`` transition.
"""

import json

import pytest

from autogen.beta import Agent
from autogen.beta.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from autogen.beta.network import (
    EV_HANDOFF,
    EV_TEXT,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)
from autogen.beta.network.adapters.workflow import (
    WORKFLOW_TYPE,
    WorkflowAdapter,
    WorkflowState,
)
from autogen.beta.network.client.tools.handoff import (
    make_handoff_tool,
)
from autogen.beta.network.errors import ProtocolError
from autogen.beta.network.session import (
    SessionState,
)
from autogen.beta.network.transitions import (
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
from autogen.beta.testing import TestConfig


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


def _envelope(sender: str, *, event_type: str = EV_TEXT, tool: str = "") -> Envelope:
    data = {"text": "x"} if event_type == EV_TEXT else {"tool": tool}
    return Envelope(
        envelope_id=f"env-{sender}",
        session_id="s1",
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

    def test_tool_called_matches_handoff_tool(self) -> None:
        env = _envelope("alice", event_type=EV_HANDOFF, tool="transfer_to_eng")
        assert ToolCalled("transfer_to_eng").evaluate(_state(order=["alice"]), env) is True
        assert ToolCalled("escalate").evaluate(_state(order=["alice"]), env) is False

    def test_tool_called_ignores_non_handoff_envelopes(self) -> None:
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
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    carol_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())
    carol = await carol_hc.register(_agent("carol"), Passport(name="carol"), Resume())

    graph = TransitionGraph.round_robin([alice.agent_id, bob.agent_id, carol.agent_id])
    session = await alice.open(
        type=WORKFLOW_TYPE,
        target=[bob.agent_id, carol.agent_id],
        knobs={"graph": graph.to_dict()},
    )
    assert session.state == SessionState.ACTIVE

    state = hub._adapter_states[session.session_id]
    assert isinstance(state, WorkflowState)
    assert state.expected_next_speaker == alice.agent_id

    # Manual sends in turn order.
    for sender_client in [alice, bob, carol]:
        env = Envelope(
            session_id=session.session_id,
            sender_id=sender_client.agent_id,
            audience=None,
            event_type=EV_TEXT,
            event_data={"text": f"turn from {sender_client.agent_id}"},
        )
        await hub.post_envelope(env)

    state = hub._adapter_states[session.session_id]
    assert state.expected_next_speaker == alice.agent_id  # cycled back
    assert state.turn_count == 3

    # Out-of-turn send is rejected.
    bad = Envelope(
        session_id=session.session_id,
        sender_id=bob.agent_id,  # alice is next
        audience=None,
        event_type=EV_TEXT,
        event_data={"text": "out of turn"},
    )
    with pytest.raises(ProtocolError, match="expects"):
        await hub.post_envelope(bad)

    await alice_hc.close()
    await bob_hc.close()
    await carol_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_workflow_sequence_pipeline_terminates_after_last_step() -> None:
    """Sequential pipeline a → b → c via TransitionGraph.sequence."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    a_hc = HubClient(link, hub=hub)
    b_hc = HubClient(link, hub=hub)
    c_hc = HubClient(link, hub=hub)
    alice = await a_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await b_hc.register(_agent("bob"), Passport(name="bob"), Resume())
    carol = await c_hc.register(_agent("carol"), Passport(name="carol"), Resume())

    graph = TransitionGraph.sequence([alice.agent_id, bob.agent_id, carol.agent_id])
    session = await alice.open(
        type=WORKFLOW_TYPE,
        target=[bob.agent_id, carol.agent_id],
        knobs={"graph": graph.to_dict()},
    )

    # alice, bob, carol each post once. After carol's post, max_turns=3
    # is reached and the session closes.
    for sender_client in [alice, bob, carol]:
        env = Envelope(
            session_id=session.session_id,
            sender_id=sender_client.agent_id,
            audience=None,
            event_type=EV_TEXT,
            event_data={"text": f"turn from {sender_client.agent_id}"},
        )
        await hub.post_envelope(env)

    # Wait briefly for the close envelope to be dispatched.
    final = await hub.get_session(session.session_id)
    assert final.state == SessionState.CLOSED
    assert final.close_reason == "max_turns"

    await a_hc.close()
    await b_hc.close()
    await c_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_workflow_swarm_with_tool_handoff_and_revert() -> None:
    """Swarm: triage hands off to eng via ToolCalled, eng replies, control
    reverts to triage via FromSpeaker(eng) → RevertToInitiatorTarget."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    triage_hc = HubClient(link, hub=hub)
    eng_hc = HubClient(link, hub=hub)
    triage = await triage_hc.register(_agent("triage"), Passport(name="triage"), Resume())
    eng = await eng_hc.register(_agent("eng"), Passport(name="eng"), Resume())

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
    session = await triage.open(
        type=WORKFLOW_TYPE,
        target=[eng.agent_id],
        knobs={"graph": graph.to_dict()},
    )

    # 1. triage emits the handoff envelope (simulating LLM tool call).
    handoff_env = Envelope(
        session_id=session.session_id,
        sender_id=triage.agent_id,
        audience=None,
        event_type=EV_HANDOFF,
        event_data={"tool": "transfer_to_eng", "reason": "needs eng review"},
    )
    await hub.post_envelope(handoff_env)
    state = hub._adapter_states[session.session_id]
    assert state.expected_next_speaker == eng.agent_id

    # 2. eng replies with text.
    eng_env = Envelope(
        session_id=session.session_id,
        sender_id=eng.agent_id,
        audience=None,
        event_type=EV_TEXT,
        event_data={"text": "eng analysis: looks good"},
    )
    await hub.post_envelope(eng_env)
    state = hub._adapter_states[session.session_id]
    # FromSpeaker(eng) fires → revert to initiator (triage).
    assert state.expected_next_speaker == triage.agent_id

    # 3. triage closes via the explicit close session API (mirrors the
    # exit criterion's TerminateTarget, but we use close_session for the
    # deterministic test). Alternatively triage could send a "done"
    # handoff envelope routed to TerminateTarget in a richer graph.
    closed = await hub.close_session(session.session_id, reason="triage_done")
    assert closed.state == SessionState.CLOSED
    assert closed.close_reason == "triage_done"

    await triage_hc.close()
    await eng_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_workflow_manager_as_initiator_auto_pattern() -> None:
    """AutoPattern equivalent: manager is initiator + RevertToInitiator default.

    Manager directs by emitting handoff envelopes; respondents always
    revert to the manager."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    mgr_hc = HubClient(link, hub=hub)
    a_hc = HubClient(link, hub=hub)
    b_hc = HubClient(link, hub=hub)
    mgr = await mgr_hc.register(_agent("mgr"), Passport(name="mgr"), Resume())
    alice = await a_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await b_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    graph = TransitionGraph(
        initial_speaker=mgr.agent_id,
        transitions=[
            Transition(when=ToolCalled("ask_alice"), then=AgentTarget(alice.agent_id)),
            Transition(when=ToolCalled("ask_bob"), then=AgentTarget(bob.agent_id)),
        ],
        default_target=RevertToInitiatorTarget(),
        max_turns=20,
    )
    session = await mgr.open(
        type=WORKFLOW_TYPE,
        target=[alice.agent_id, bob.agent_id],
        knobs={"graph": graph.to_dict()},
    )

    # mgr asks alice → alice → revert to mgr → mgr asks bob → bob → revert to mgr
    sequence = [
        (mgr, EV_HANDOFF, {"tool": "ask_alice"}),
        (alice, EV_TEXT, {"text": "alice answers"}),
        (mgr, EV_HANDOFF, {"tool": "ask_bob"}),
        (bob, EV_TEXT, {"text": "bob answers"}),
    ]
    expected_next = [alice.agent_id, mgr.agent_id, bob.agent_id, mgr.agent_id]
    for (sender, et, ed), exp in zip(sequence, expected_next):
        env = Envelope(
            session_id=session.session_id,
            sender_id=sender.agent_id,
            audience=None,
            event_type=et,
            event_data=ed,
        )
        await hub.post_envelope(env)
        state = hub._adapter_states[session.session_id]
        assert state.expected_next_speaker == exp, (
            f"after {sender.agent_id} sent {et}, expected_next was {state.expected_next_speaker}, expected {exp}"
        )

    await mgr_hc.close()
    await a_hc.close()
    await b_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_workflow_hydrate_recovers_expected_next_speaker(tmp_path) -> None:
    """Hub restart mid-workflow recovers expected_next_speaker by re-folding
    the WAL through WorkflowAdapter.fold."""
    store = DiskKnowledgeStore(str(tmp_path))
    hub1 = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link1 = LocalLink(hub1)

    triage_hc = HubClient(link1, hub=hub1)
    eng_hc = HubClient(link1, hub=hub1)
    triage = await triage_hc.register(_agent("triage"), Passport(name="triage"), Resume())
    eng = await eng_hc.register(_agent("eng"), Passport(name="eng"), Resume())

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
    session = await triage.open(
        type=WORKFLOW_TYPE,
        target=[eng.agent_id],
        knobs={"graph": graph.to_dict()},
    )
    handoff_env = Envelope(
        session_id=session.session_id,
        sender_id=triage.agent_id,
        audience=None,
        event_type=EV_HANDOFF,
        event_data={"tool": "transfer_to_eng"},
    )
    await hub1.post_envelope(handoff_env)
    pre_state = hub1._adapter_states[session.session_id]
    assert pre_state.expected_next_speaker == eng.agent_id

    await triage_hc.close()
    await eng_hc.close()
    await hub1.close()

    # Reopen a fresh hub against the same store.
    store2 = DiskKnowledgeStore(str(tmp_path))
    hub2 = await Hub.open(store2, ttl_sweep_interval=0, expectation_sweep_interval=0)

    refreshed = await hub2.get_session(session.session_id)
    assert refreshed.manifest.type == WORKFLOW_TYPE
    assert refreshed.state == SessionState.ACTIVE

    rebuilt = hub2._adapter_states[session.session_id]
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
    link = LocalLink(hub)
    a_hc = HubClient(link, hub=hub)
    b_hc = HubClient(link, hub=hub)
    alice = await a_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await b_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    with pytest.raises(ProtocolError, match="graph"):
        await hub.create_session(
            creator_id=alice.agent_id,
            manifest_type=WORKFLOW_TYPE,
            participants=[bob.agent_id],
            knobs={},
        )

    await a_hc.close()
    await b_hc.close()
    await hub.close()


# ── NetworkPlugin.register_workflow ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_workflow_attaches_handoff_tools_per_tool_called() -> None:
    """NetworkPlugin.register_workflow materialises one tool per ToolCalled
    transition, deduping by tool_name."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())

    pre_count = len(alice.agent.tools)

    graph = TransitionGraph(
        initial_speaker="alice",
        transitions=[
            Transition(when=ToolCalled("transfer_to_eng"), then=AgentTarget("eng")),
            Transition(when=ToolCalled("escalate"), then=AgentTarget("legal")),
            # Same tool name in another transition should be deduped.
            Transition(
                when=ToolCalled("escalate"),
                then=AgentTarget("legal"),
                priority=1,
            ),
            # FromSpeaker doesn't materialise a tool.
            Transition(when=FromSpeaker("eng"), then=RevertToInitiatorTarget()),
        ],
    )
    plugin = alice.agent._plugins[-1] if hasattr(alice.agent, "_plugins") else None
    # Locate the NetworkPlugin to call register_workflow.
    from autogen.beta.network.client.plugin import NetworkPlugin

    network_plugin = NetworkPlugin(alice)
    attached = network_plugin.register_workflow(graph)

    new_names = {t.name for t in attached}
    assert new_names == {"transfer_to_eng", "escalate"}
    assert len(alice.agent.tools) == pre_count + 2

    # Re-register the same graph: idempotent (no duplicate tools).
    again = network_plugin.register_workflow(graph)
    assert again == []

    await alice_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_handoff_tool_posts_ev_handoff_envelope() -> None:
    """The materialised tool body posts an EV_HANDOFF envelope tagged with
    tool_name + reason into the active session."""
    from autogen.beta import Context
    from autogen.beta.events import ToolCallEvent
    from autogen.beta.network.policies import SESSION_DEP
    from autogen.beta.stream import MemoryStream

    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    triage_hc = HubClient(link, hub=hub)
    eng_hc = HubClient(link, hub=hub)
    triage = await triage_hc.register(_agent("triage"), Passport(name="triage"), Resume())
    eng = await eng_hc.register(_agent("eng"), Passport(name="eng"), Resume())

    graph = TransitionGraph(
        initial_speaker=triage.agent_id,
        transitions=[
            Transition(
                when=ToolCalled("transfer_to_eng"),
                then=AgentTarget(eng.agent_id),
            ),
        ],
        default_target=TerminateTarget(),
        max_turns=4,
    )
    session = await triage.open(
        type=WORKFLOW_TYPE,
        target=[eng.agent_id],
        knobs={"graph": graph.to_dict()},
    )

    handoff = make_handoff_tool(triage, "transfer_to_eng")
    event = ToolCallEvent(
        name="transfer_to_eng",
        arguments=json.dumps({"reason": "needs domain expert"}),
    )
    ctx = Context(
        stream=MemoryStream(),
        dependencies={SESSION_DEP: session},
    )
    result = await handoff(event, ctx)
    # Either ToolResultEvent or ToolErrorEvent — assert the success path.
    assert hasattr(result, "result")
    parts = result.result.parts
    assert "handoff posted" in parts[0].content

    # Adapter should have advanced to eng.
    state = hub._adapter_states[session.session_id]
    assert state.expected_next_speaker == eng.agent_id

    await triage_hc.close()
    await eng_hc.close()
    await hub.close()
