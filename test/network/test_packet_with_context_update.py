# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``EV_PACKET`` carrying routing + ``context_updates`` in event_data.

Pins the packet-execution-model contract that a single ``EV_PACKET``
envelope can atomically:

* Update ``WorkflowState.context_vars`` (via ``set`` / ``delete`` in
  ``event_data["context_updates"]``), and
* Drive routing — either via a pre-resolved ``routing.target``
  (dynamic ``Handoff`` return) or via a ``routing.tool`` that matches
  a ``ToolCalled`` graph rule.

``WorkflowAdapter.fold`` applies ``context_updates`` to
``context_vars`` before running ``select_next``, so a
``ContextEquals`` rule on the same packet matches the just-set value.
"""

import pytest

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_PACKET,
    WORKFLOW_TYPE,
    AgentTarget,
    ContextEquals,
    Envelope,
    FromSpeaker,
    Hub,
    TerminateTarget,
    ToolCalled,
    Transition,
    TransitionGraph,
)
from ag2.testing import TestConfig


def _agent(name: str, *replies: str) -> Agent:
    return Agent(name=name, config=TestConfig(*replies))


def _packet(
    *,
    channel_id: str,
    sender_id: str,
    tool: str | None = None,
    reason: str = "",
    target: str | None = None,
    set_vars: dict | None = None,
    delete_vars: list | None = None,
    body: str = "",
) -> Envelope:
    """Helper: build an EV_PACKET envelope with the given routing
    intent and context_updates."""
    routing: dict = {"kind": "handoff" if tool or target else "text"}
    if tool is not None:
        routing["tool"] = tool
    if reason:
        routing["reason"] = reason
    if target is not None:
        routing["target"] = target
    return Envelope(
        channel_id=channel_id,
        sender_id=sender_id,
        audience=None,
        event_type=EV_PACKET,
        event_data={
            "routing": routing,
            "context_updates": {
                "set": set_vars or {},
                "delete": delete_vars or [],
            },
            "body": body,
        },
    )


@pytest.mark.asyncio
class TestPacketWithContextUpdate:
    """``EV_PACKET`` ``event_data["context_updates"]`` is applied
    before ``select_next`` runs on the same fold."""

    async def test_set_applied_before_select_next(self) -> None:
        """``ContextEquals`` matches the value the packet's
        ``context_updates.set`` writes in the same fold."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)

        router = await hub.register(_agent("router"))
        billing = await hub.register(_agent("billing"))
        technical = await hub.register(_agent("technical"))

        graph = TransitionGraph(
            initial_speaker=router.agent_id,
            transitions=[
                Transition(when=ContextEquals("category", "billing"), then=AgentTarget(billing.agent_id)),
                Transition(when=ContextEquals("category", "technical"), then=AgentTarget(technical.agent_id)),
            ],
            default_target=TerminateTarget(reason="unrouted"),
            max_turns=10,
        )
        channel = await router.open(
            type=WORKFLOW_TYPE,
            target=[billing.agent_id, technical.agent_id],
            knobs={"graph": graph.to_dict()},
        )

        # Single EV_PACKET carrying routing + context_updates. Fold
        # applies context_updates BEFORE running select_next.
        envelope = _packet(
            channel_id=channel.channel_id,
            sender_id=router.agent_id,
            tool="classify_as_technical",
            reason="API error",
            set_vars={"category": "technical"},
        )
        await hub.post_envelope(envelope)

        state = hub._adapter_states[channel.channel_id]
        # Context vars carry the new key.
        assert state.context_vars == {"category": "technical"}
        # ContextEquals matched on post-update state — speaker is the
        # technical specialist, NOT routed to billing or terminated.
        assert state.expected_next_speaker == technical.agent_id

        await hub.close()

    async def test_delete_applied_before_select_next(self) -> None:
        """``context_updates.delete`` removes keys atomically with the packet."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)

        alice = await hub.register(_agent("alice"))
        bob = await hub.register(_agent("bob"))

        graph = TransitionGraph(
            initial_speaker=alice.agent_id,
            transitions=[
                # If `route` is unset, terminate. Otherwise route to bob.
                Transition(when=ContextEquals("route", None), then=TerminateTarget("cleared")),
                Transition(when=FromSpeaker(alice.agent_id), then=AgentTarget(bob.agent_id)),
            ],
            default_target=TerminateTarget(reason="default"),
            max_turns=10,
        )
        channel = await alice.open(
            type=WORKFLOW_TYPE,
            target=[bob.agent_id],
            knobs={
                "graph": graph.to_dict(),
                "context_vars": {"route": "stale"},
            },
        )

        # Verify the seeded value is in place.
        assert hub._adapter_states[channel.channel_id].context_vars == {"route": "stale"}

        # Packet that clears the route via context_updates.delete.
        # Fold applies it before select_next, so ContextEquals(route, None)
        # terminate rule fires.
        envelope = _packet(
            channel_id=channel.channel_id,
            sender_id=alice.agent_id,
            tool="clear_route",
            reason="done routing",
            delete_vars=["route"],
        )
        await hub.post_envelope(envelope)

        state = hub._adapter_states[channel.channel_id]
        assert state.context_vars == {}
        # Terminate rule fired — no next speaker, close reason set.
        assert state.expected_next_speaker is None
        assert state.pending_close_reason == "cleared"

        await hub.close()

    async def test_packet_without_context_updates_unchanged(self) -> None:
        """``EV_PACKET`` with empty ``context_updates`` behaves as the
        normal substantive envelope it is — routing fires, context
        untouched."""
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
            knobs={
                "graph": graph.to_dict(),
                "context_vars": {"existing": "kept"},
            },
        )

        envelope = _packet(
            channel_id=channel.channel_id,
            sender_id=alice.agent_id,
            tool="advance",
            reason="go",
        )
        await hub.post_envelope(envelope)

        state = hub._adapter_states[channel.channel_id]
        # Existing context untouched; routing follows FromSpeaker rule.
        assert state.context_vars == {"existing": "kept"}
        assert state.expected_next_speaker == bob.agent_id

        await hub.close()

    async def test_dynamic_handoff_target_supersedes_select_next(self) -> None:
        """A packet with a pre-resolved ``routing.target`` (dynamic
        ``Handoff`` return) routes there directly, bypassing graph rules."""
        store = MemoryKnowledgeStore()
        hub = await Hub.open(store, ttl_sweep_interval=0)

        router = await hub.register(_agent("router"))
        a = await hub.register(_agent("a"))
        b = await hub.register(_agent("b"))

        # Graph would route smart_route to A via ToolCalled rule, but
        # the packet's pre-resolved routing.target=b should win.
        graph = TransitionGraph(
            initial_speaker=router.agent_id,
            transitions=[
                Transition(when=ToolCalled("smart_route"), then=AgentTarget(a.agent_id)),
            ],
            default_target=TerminateTarget(reason="unrouted"),
            max_turns=10,
        )
        channel = await router.open(
            type=WORKFLOW_TYPE,
            target=[a.agent_id, b.agent_id],
            knobs={"graph": graph.to_dict()},
        )

        envelope = _packet(
            channel_id=channel.channel_id,
            sender_id=router.agent_id,
            tool="smart_route",
            reason="dynamic",
            target=b.agent_id,  # pre-resolved Handoff target — wins over graph rule
        )
        await hub.post_envelope(envelope)

        state = hub._adapter_states[channel.channel_id]
        assert state.expected_next_speaker == b.agent_id

        await hub.close()
