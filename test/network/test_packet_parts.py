# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A workflow packet reaches the next speaker as separate parts.

The handoff signal and the body stay their own ``TextInput`` on both paths:
the turn input handed to ``Agent.ask`` and the view projection of history.
"""

import asyncio

import pytest

from ag2 import Agent
from ag2.events import ModelMessage, ModelRequest, TextInput
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_PACKET,
    WORKFLOW_TYPE,
    AgentTarget,
    Envelope,
    Hub,
    TerminateTarget,
    ToolCalled,
    Transition,
    TransitionGraph,
)
from ag2.network.adapters.workflow import WorkflowAdapter
from ag2.network.views.builtin import FullTranscript, NamedTranscript, NamedWindowedSummary
from ag2.testing import TestConfig, TrackingConfig

from ._helpers import two_party_metadata


def _packet(*, sender: str = "alice", routing: dict | None = None, body: str = "") -> Envelope:
    return Envelope(
        channel_id="ch-test",
        sender_id=sender,
        audience=None,
        event_type=EV_PACKET,
        event_data={"routing": routing or {"kind": "text"}, "body": body},
    )


@pytest.fixture
def handoff() -> dict[str, str]:
    return {"kind": "handoff", "tool": "to_reviewer", "reason": "needs a check"}


def test_turn_input_keeps_handoff_signal_and_body_apart(handoff: dict[str, str]) -> None:
    parts = WorkflowAdapter().extract_turn_input(_packet(routing=handoff, body="draft"))

    assert parts == [TextInput("[Handed off via to_reviewer] needs a check"), TextInput("draft")]


def test_turn_input_without_reason_has_a_bare_signal() -> None:
    routing = {"kind": "handoff", "tool": "to_reviewer"}

    parts = WorkflowAdapter().extract_turn_input(_packet(routing=routing))

    assert parts == [TextInput("[Handed off via to_reviewer]")]


def test_turn_input_for_a_bridge_built_handoff_without_a_tool_names_it_generically() -> None:
    routing = {"kind": "handoff", "target": "bob", "reason": "needs expert"}

    parts = WorkflowAdapter().extract_turn_input(_packet(routing=routing))

    assert parts == [TextInput("[Handed off via handoff] needs expert")]


def test_turn_input_for_a_plain_text_packet_is_the_body() -> None:
    assert WorkflowAdapter().extract_turn_input(_packet(body="draft")) == [TextInput("draft")]


def test_an_empty_packet_yields_nothing_on_both_paths() -> None:
    adapter = WorkflowAdapter()
    envelope = _packet()

    assert adapter.extract_turn_input(envelope) is None
    assert adapter.render_envelope(envelope) is None


def test_render_and_turn_input_agree(handoff: dict[str, str]) -> None:
    adapter = WorkflowAdapter()
    envelope = _packet(routing=handoff, body="draft")

    assert adapter.render_envelope(envelope) == adapter.extract_turn_input(envelope)


@pytest.mark.asyncio
async def test_an_inbound_packet_projects_to_one_request_with_both_parts(handoff: dict[str, str]) -> None:
    projection = await FullTranscript().project(
        [_packet(routing=handoff, body="draft")],
        participant_id="bob",
        channel=two_party_metadata("alice", "bob"),
        render_envelope=WorkflowAdapter().render_envelope,
    )

    assert projection == [
        ModelRequest([TextInput("[Handed off via to_reviewer] needs a check"), TextInput("draft")]),
    ]


@pytest.mark.asyncio
async def test_a_named_view_labels_only_the_leading_part(handoff: dict[str, str]) -> None:
    projection = await NamedTranscript().project(
        [_packet(routing=handoff, body="draft")],
        participant_id="bob",
        channel=two_party_metadata("alice", "bob"),
        render_envelope=WorkflowAdapter().render_envelope,
        name_for=lambda agent_id: agent_id.title(),
    )

    assert projection == [
        ModelRequest([TextInput("[Alice]: [Handed off via to_reviewer] needs a check"), TextInput("draft")]),
    ]


@pytest.mark.asyncio
async def test_an_own_packet_projects_to_one_message_per_part(handoff: dict[str, str]) -> None:
    projection = await NamedWindowedSummary(recent_n=5).project(
        [_packet(sender="bob", routing=handoff, body="draft")],
        participant_id="bob",
        channel=two_party_metadata("alice", "bob"),
        render_envelope=WorkflowAdapter().render_envelope,
    )

    assert projection == [ModelMessage("[Handed off via to_reviewer] needs a check"), ModelMessage("draft")]


@pytest.mark.asyncio
async def test_the_next_speaker_is_handed_the_signal_and_the_body_as_two_parts() -> None:
    hub = await Hub.open(MemoryKnowledgeStore(), ttl_sweep_interval=0, expectation_sweep_interval=0)
    bob_config = TrackingConfig(TestConfig("reviewed"))
    alice = await hub.register(Agent("alice", config=TestConfig("")))
    bob = await hub.register(Agent("bob", config=bob_config))
    graph = TransitionGraph(
        initial_speaker=alice.agent_id,
        transitions=[Transition(when=ToolCalled("to_reviewer"), then=AgentTarget(bob.agent_id))],
        default_target=TerminateTarget(reason="done"),
    )
    channel = await alice.open(type=WORKFLOW_TYPE, target=[bob.agent_id], knobs={"graph": graph.to_dict()})

    await hub.post_envelope(
        Envelope(
            channel_id=channel.channel_id,
            sender_id=alice.agent_id,
            audience=None,
            event_type=EV_PACKET,
            event_data={
                "routing": {"kind": "handoff", "tool": "to_reviewer", "reason": "needs a check"},
                "context_updates": {"set": {}, "delete": []},
                "body": "draft",
            },
        )
    )
    for _ in range(250):
        if bob_config.mock.called:
            break
        await asyncio.sleep(0.02)
    await hub.close()

    bob_config.mock.assert_called_once_with(
        ModelRequest([TextInput("[alice]: [Handed off via to_reviewer] needs a check"), TextInput("draft")])
    )
