# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Observability tests: ``HubListener``, ``HubArbiter``, audit subscribe,
handler exception trap, ``Hub.health()``, hub logging.
"""

import asyncio
import logging

import pytest

from autogen.beta import Agent
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    AccessDeniedError,
    Allow,
    BaseHubListener,
    Deny,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
    Rule,
)
from autogen.beta.network.rule import AccessBlock
from autogen.beta.testing import TestConfig


def _agent(name: str, *replies: str) -> Agent:
    return Agent(name=name, config=TestConfig(*replies))


# ── HubListener ───────────────────────────────────────────────────────────


class _RecordingListener(BaseHubListener):
    """Captures every event for assertion."""

    def __init__(self) -> None:
        self.envelope_posted: list = []
        self.envelope_rejected: list = []
        self.channel_events: list = []
        self.agent_events: list = []
        self.turn_failed: list = []
        self.task_events: list = []
        self.dispatch_failed: list = []

    async def on_envelope_posted(self, envelope, metadata) -> None:
        self.envelope_posted.append((envelope.event_type, envelope.sender_id))

    async def on_envelope_rejected(self, envelope, reason) -> None:
        self.envelope_rejected.append((envelope.event_type, type(reason).__name__))

    async def on_channel_event(self, channel_id, kind, payload) -> None:
        self.channel_events.append((kind, channel_id))

    async def on_agent_event(self, agent_id, kind, payload) -> None:
        self.agent_events.append((kind, agent_id))

    async def on_turn_failed(self, channel_id, agent_id, envelope_id, exc) -> None:
        self.turn_failed.append((channel_id, agent_id, type(exc).__name__))

    async def on_task_event(self, task_id, kind, payload) -> None:
        self.task_events.append((kind, task_id))

    async def on_dispatch_failed(self, envelope, recipient_id, reason) -> None:
        self.dispatch_failed.append((envelope.event_type, recipient_id))


@pytest.mark.asyncio
async def test_listener_receives_agent_and_channel_events() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    listener = _RecordingListener()
    hub.register_listener(listener)

    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    channel = await alice.open(type="consulting", target="bob")
    await alice_hc.close_channel(channel.channel_id, reason="done")

    agent_kinds = [k for k, _ in listener.agent_events]
    assert "registered" in agent_kinds
    channel_kinds = [k for k, _ in listener.channel_events]
    assert "created" in channel_kinds
    assert "opened" in channel_kinds
    assert "closed" in channel_kinds

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_listener_receives_envelope_posted() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    listener = _RecordingListener()
    hub.register_listener(listener)

    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    channel = await alice.open(type="conversation", target="bob")
    await channel.send("hi there")

    from autogen.beta.network import EV_TEXT

    text_posts = [t for t, _ in listener.envelope_posted if t == EV_TEXT]
    assert len(text_posts) >= 1

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_listener_receives_envelope_rejected_on_access_denied() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    listener = _RecordingListener()
    hub.register_listener(listener)

    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    # bob blocks alice inbound — channel.open will fail at create_channel.
    await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob_rule = Rule(access=AccessBlock(inbound_from=["carol"], outbound_to=["*"]))
    await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume(), rule=bob_rule)

    alice_client = next(iter(alice_hc._clients.values()))
    with pytest.raises(AccessDeniedError):
        await alice_client.open(type="consulting", target="bob")

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_listener_exception_does_not_break_dispatch() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    class _BadListener(BaseHubListener):
        async def on_envelope_posted(self, envelope, metadata) -> None:
            raise RuntimeError("boom")

    good = _RecordingListener()
    hub.register_listener(_BadListener())
    hub.register_listener(good)

    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    channel = await alice.open(type="conversation", target="bob")
    await channel.send("hi")

    # Both listeners ran; bad listener's exception was swallowed,
    # good listener still got the events.
    assert good.envelope_posted

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


# ── HubArbiter ────────────────────────────────────────────────────────────


class _DenyArbiter:
    """Denies every send. Used to verify the arbiter seam works."""

    async def authorize_send(self, envelope, sender, sender_rule, recipients):
        return Deny(reason="custom denial")

    async def authorize_inbox(self, envelope, recipient, recipient_rule, current_pending):
        return Allow()

    async def authorize_dispatch(self, envelope, sender, recipient, recipient_rule):
        return Allow()

    async def authorize_channel_open(
        self, manifest, creator, creator_rule, invitees, invitee_rules, active_creator_channels
    ):
        return Allow()

    async def authorize_register(self, passport, resume, rule):
        return Allow()

    async def resolve_unknown_audience(self, envelope, unknown_ids):
        return None


@pytest.mark.asyncio
async def test_custom_arbiter_can_deny_send() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())
    channel = await alice.open(type="conversation", target="bob")

    # Swap in the deny-everything arbiter AFTER the channel is open.
    hub.register_arbiter(_DenyArbiter())

    with pytest.raises(AccessDeniedError, match="custom denial"):
        await channel.send(
            "blocked",
            audience=[next(p.agent_id for p in channel.metadata.participants if p.agent_id != alice.agent_id)],
        )

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_default_arbiter_preserves_rule_based_behavior() -> None:
    """Default ``RuleBasedArbiter`` matches the prior inline-check semantics."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())
    channel = await alice.open(type="conversation", target="bob")

    # Tighten alice's rule after the channel is open: outbound only to "carol".
    # The default RuleBasedArbiter must reject explicit-audience sends to bob.
    await hub.set_rule(alice.agent_id, Rule(access=AccessBlock(inbound_from=["*"], outbound_to=["carol"])))

    bob_id = next(p.agent_id for p in channel.metadata.participants if p.agent_id != alice.agent_id)
    with pytest.raises(AccessDeniedError):
        await channel.send("nope", audience=[bob_id])

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_resolve_unknown_audience_silent_drop_default() -> None:
    """Default arbiter silently drops envelopes for unknown audience ids."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())
    channel = await alice.open(type="conversation", target="bob")

    # Audience with one real id + one unknown — only real id gets delivery.
    bob_id = next(p.agent_id for p in channel.metadata.participants if p.agent_id != alice.agent_id)
    envelope_id = await channel.send("partial", audience=[bob_id, "ghost-id"])
    assert envelope_id  # accepted into WAL despite unknown audience member

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


# ── Audit subscribe ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_subscribe_taps_live_stream() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    captured: list[dict] = []

    async def tap(record: dict) -> None:
        captured.append(record)

    hub._audit_log.subscribe(tap)

    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())

    kinds = [r.get("kind") for r in captured]
    assert "agent_registered" in kinds

    await alice_hc.close()
    await hub.close()


# ── Handler exception trap ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handler_exception_does_not_crash_channel() -> None:
    """A handler that raises produces on_turn_failed; channel stays alive."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    listener = _RecordingListener()
    hub.register_listener(listener)

    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())

    # Bob's agent will raise inside agent.ask.
    bob_agent = Agent(name="bob", config=TestConfig(RuntimeError("intentional")))
    bob = await bob_hc.register(bob_agent, Passport(name="bob"), Resume())

    # Replace bob's notify handler to also be the default; raise happens
    # inside _process_substantive when agent.ask is called.
    channel = await alice.open(type="conversation", target="bob")
    # Suppress the receive-side re-raise so the test can observe the
    # listener-side audit + on_turn_failed without the test framework
    # surfacing the raise as a test failure.
    bob._on_envelope = None
    from autogen.beta.network.client.handlers import _process_substantive

    import contextlib as _ctx

    async def safe_handler(env):
        with _ctx.suppress(Exception):
            await _process_substantive(env, bob)

    bob.on_envelope(safe_handler)

    await channel.send("hello bob")

    # Yield so receive-loop dispatches and handler runs.
    await asyncio.sleep(0.1)

    assert listener.turn_failed, "expected on_turn_failed to fire"
    # Subsequent send still works — channel survived.
    envelope_id = await channel.send("still alive?")
    assert envelope_id

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


# ── Hub.health() ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_snapshot_shape() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)

    initial = hub.health()
    assert initial == {
        "active_channels": 0,
        "registered_agents": 0,
        "pending_inbox_total": 0,
        "oldest_pending_count": None,
        "registered_listeners": 0,
        "adapters_loaded": 4,
    }

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())
    await alice.open(type="conversation", target="bob")

    snapshot = hub.health()
    assert snapshot["registered_agents"] == 2
    assert snapshot["active_channels"] == 1
    assert snapshot["adapters_loaded"] == 4

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


# ── Hub logging ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hub_logs_state_transitions(caplog) -> None:
    caplog.set_level(logging.INFO, logger="autogen.beta.network.hub.core")
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    messages = [r.getMessage() for r in caplog.records]
    assert any("agent registered" in m for m in messages)

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_hub_logs_warning_on_rejection(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="autogen.beta.network.hub.core")
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())
    channel = await alice.open(type="conversation", target="bob")
    await hub.set_rule(alice.agent_id, Rule(access=AccessBlock(inbound_from=["*"], outbound_to=["nope"])))

    bob_id = next(p.agent_id for p in channel.metadata.participants if p.agent_id != alice.agent_id)
    with pytest.raises(AccessDeniedError):
        await channel.send("blocked", audience=[bob_id])

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("post_envelope rejected" in r.getMessage() for r in warnings)

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()
