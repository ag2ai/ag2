# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Plain-Python bridge drives a workflow turn end-to-end.

Headline non-AG2 surface for the adapter Layer-2 helpers: a non-AG2
client constructs a workflow ``EV_PACKET`` envelope via
``WorkflowAdapter.build_packet_envelope`` and posts it directly through
``Hub.post_envelope`` — no ``@tool`` decorator, no ``Agent``, no
``NetworkPlugin``. The workflow's transition graph advances the
expected next speaker based purely on the bridge-supplied envelopes.

This test is the load-bearing evidence that the three-layer design
(capabilities → envelope helpers → LLM-tool wrappers) actually delivers
its promised non-AG2 surface, not just adapter-gated AG2-only tools.
"""

import pytest

from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_PACKET,
    AgentTarget,
    FromSpeaker,
    Handoff,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
    RevertToInitiatorTarget,
    TerminateTarget,
    Transition,
    TransitionGraph,
    WorkflowState,
    default_build_packet_envelope,
)
from ag2.network.adapters.workflow import WORKFLOW_TYPE
from ag2.network.channel import ChannelState


@pytest.mark.asyncio
async def test_bridge_drives_workflow_via_layer2_envelope_helpers() -> None:
    """A bridge with no AG2 plumbing posts EV_PACKET envelopes to
    advance the workflow through alice → bob → alice (revert).

    Demonstrates the full non-AG2 surface:
    1. ``adapter.build_packet_envelope(...)`` constructs the round.
    2. ``hub.post_envelope(envelope)`` commits it.
    3. ``hub.adapter_state(channel_id)`` reflects the advance.

    No ``Agent``, no ``@tool``, no ``NetworkPlugin``.
    """
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    # Register the two participants without any LLM — they are pure
    # identities. ``register_human`` is the cleanest way to get a
    # passport stamped without attaching an LLM agent.
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register_human(Passport(name="alice"), resume=Resume())
    bob = await bob_hc.register_human(Passport(name="bob"), resume=Resume())

    graph = TransitionGraph(
        initial_speaker=alice.agent_id,
        transitions=[
            Transition(
                when=FromSpeaker(alice.agent_id),
                then=AgentTarget(bob.agent_id),
            ),
            Transition(
                when=FromSpeaker(bob.agent_id),
                then=RevertToInitiatorTarget(),
            ),
        ],
        default_target=TerminateTarget(reason="bridge_done"),
        max_turns=8,
    )

    # Alice opens the workflow channel as the creator.
    channel = await alice.open(
        type=WORKFLOW_TYPE,
        target=[bob.agent_id],
        knobs={"graph": graph.to_dict()},
    )
    assert channel.state == ChannelState.ACTIVE

    # Round 1: bridge constructs a workflow packet "as alice" using the
    # adapter's Layer-2 helper. No tool wrapper involved.
    adapter = hub.adapter_for(channel.channel_id)
    env_alice = adapter.build_packet_envelope(
        channel_id=channel.channel_id,
        sender_id=alice.agent_id,
        body="alice opens the discussion",
    )
    assert env_alice.event_type == EV_PACKET

    await hub.post_envelope(env_alice)

    # Workflow state advanced to expect bob next.
    state = hub.adapter_state(channel.channel_id)
    assert isinstance(state, WorkflowState)
    assert state.expected_next_speaker == bob.agent_id
    assert state.last_speaker_id == alice.agent_id

    # Round 2: bridge constructs bob's reply via the helper, including
    # a handoff back to alice.
    env_bob = adapter.build_packet_envelope(
        channel_id=channel.channel_id,
        sender_id=bob.agent_id,
        body="bob replies",
        handoff=Handoff(target=alice.agent_id, reason="back to you"),
    )
    await hub.post_envelope(env_bob)

    # FromSpeaker(bob) → RevertToInitiatorTarget → expected next is alice.
    state = hub.adapter_state(channel.channel_id)
    assert state.expected_next_speaker == alice.agent_id
    assert state.last_speaker_id == bob.agent_id

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_bridge_uses_module_level_default_helper_directly() -> None:
    """A bridge that doesn't even resolve the adapter can use the
    module-level ``default_build_packet_envelope`` directly.

    Use case: a Python harness that pre-constructs envelopes offline
    and posts them in batch through ``hub.post_envelope`` without
    any per-channel adapter resolution.
    """
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register_human(Passport(name="alice"))
    bob = await bob_hc.register_human(Passport(name="bob"))

    graph = TransitionGraph(
        initial_speaker=alice.agent_id,
        transitions=[
            Transition(
                when=FromSpeaker(alice.agent_id),
                then=AgentTarget(bob.agent_id),
            ),
        ],
        default_target=TerminateTarget(reason="bridge_done"),
        max_turns=4,
    )

    channel = await alice.open(
        type=WORKFLOW_TYPE,
        target=[bob.agent_id],
        knobs={"graph": graph.to_dict()},
    )

    # Construct via the module-level helper without ever touching the
    # adapter instance.
    envelope = default_build_packet_envelope(
        channel_id=channel.channel_id,
        sender_id=alice.agent_id,
        body="a packet built without adapter resolution",
    )
    assert envelope.event_type == EV_PACKET
    await hub.post_envelope(envelope)

    state = hub.adapter_state(channel.channel_id)
    assert state.expected_next_speaker == bob.agent_id

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()
