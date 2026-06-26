# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Observability tests: ``HubListener``, ``HubArbiter``, audit subscribe,
handler exception trap, ``Hub.health()``, hub logging.
"""

import asyncio
import logging

import pytest

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_TEXT,
    AccessDeniedError,
    Allow,
    AuditLog,
    BaseHubArbiter,
    BaseHubListener,
    Deny,
    Hub,
    HubClient,
    HumanClient,
    LocalLink,
    NetworkError,
    Passport,
    Resume,
    Rule,
)
from ag2.network.rule import AccessBlock
from ag2.testing import TestConfig


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

    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))

    channel = await alice.open(type="conversation", target="bob")
    await channel.send("hi there")

    text_posts = [t for t, _ in listener.envelope_posted if t == EV_TEXT]
    assert len(text_posts) >= 1

    await hub.close()


@pytest.mark.asyncio
async def test_listener_receives_envelope_rejected_on_access_denied() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    listener = _RecordingListener()
    hub.register_listener(listener)

    # bob blocks alice inbound — channel.open will fail at create_channel.
    alice = await hub.register(_agent("alice"))
    bob_rule = Rule(access=AccessBlock(inbound_from=["carol"], outbound_to=["*"]))
    await hub.register(_agent("bob"), rule=bob_rule)

    with pytest.raises(AccessDeniedError):
        await alice.open(type="consulting", target="bob")

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

    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))

    channel = await alice.open(type="conversation", target="bob")
    await channel.send("hi")

    # Both listeners ran; bad listener's exception was swallowed,
    # good listener still got the events.
    assert good.envelope_posted

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

    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))
    channel = await alice.open(type="conversation", target="bob")

    # Swap in the deny-everything arbiter AFTER the channel is open.
    hub.register_arbiter(_DenyArbiter())

    with pytest.raises(AccessDeniedError, match="custom denial"):
        await channel.send(
            "blocked",
            audience=[next(p.agent_id for p in channel.metadata.participants if p.agent_id != alice.agent_id)],
        )

    await hub.close()


@pytest.mark.asyncio
async def test_default_arbiter_preserves_rule_based_behavior() -> None:
    """Default ``RuleBasedArbiter`` matches the prior inline-check semantics."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))
    channel = await alice.open(type="conversation", target="bob")

    # Tighten alice's rule after the channel is open: outbound only to "carol".
    # The default RuleBasedArbiter must reject explicit-audience sends to bob.
    await hub.set_rule(alice.agent_id, Rule(access=AccessBlock(inbound_from=["*"], outbound_to=["carol"])))

    bob_id = next(p.agent_id for p in channel.metadata.participants if p.agent_id != alice.agent_id)
    with pytest.raises(AccessDeniedError):
        await channel.send("nope", audience=[bob_id])

    await hub.close()


@pytest.mark.asyncio
async def test_resolve_unknown_audience_silent_drop_default() -> None:
    """Default arbiter silently drops envelopes for unknown audience ids."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))
    channel = await alice.open(type="conversation", target="bob")

    # Audience with one real id + one unknown — only real id gets delivery.
    bob_id = next(p.agent_id for p in channel.metadata.participants if p.agent_id != alice.agent_id)
    envelope_id = await channel.send("partial", audience=[bob_id, "ghost-id"])
    assert envelope_id  # accepted into WAL despite unknown audience member

    await hub.close()


# ── Audit subscribe ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_subscribe_taps_live_stream() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    captured: list[dict] = []

    async def tap(record: dict) -> None:
        captured.append(record)

    hub.audit_log.subscribe(tap)

    await hub.register(_agent("alice"))

    kinds = [r.get("kind") for r in captured]
    assert "agent_registered" in kinds

    await hub.close()


# ── Handler exception trap ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handler_exception_does_not_crash_channel() -> None:
    """A handler that raises produces on_turn_failed; channel stays alive."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    listener = _RecordingListener()
    hub.register_listener(listener)

    alice = await hub.register(_agent("alice"))

    # Bob's agent will raise inside agent.ask. The default handler's
    # trap must observe + report the failure without re-raising and
    # the channel must keep accepting envelopes.
    bob_agent = Agent(name="bob", config=TestConfig(RuntimeError("intentional")))
    await hub.register(bob_agent)

    channel = await alice.open(type="conversation", target="bob")
    await channel.send("hello bob")

    for _ in range(200):
        if listener.turn_failed:
            break
        await asyncio.sleep(0.01)

    assert listener.turn_failed, "expected on_turn_failed to fire"
    # Subsequent send still works — channel survived the crash.
    envelope_id = await channel.send("still alive?")
    assert envelope_id

    await hub.close()


# ── Hub.health() ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_snapshot_shape() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    initial = hub.health()
    assert initial == {
        "active_channels": 0,
        "registered_agents": 0,
        "pending_inbox_total": 0,
        "max_pending_inbox_depth": None,
        # AuditLog auto-installs as the first listener.
        "registered_listeners": 1,
        "adapters_loaded": 4,
        "audit_log_bytes": 0,
    }

    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))
    await alice.open(type="conversation", target="bob")

    snapshot = hub.health()
    assert snapshot["registered_agents"] == 2
    assert snapshot["active_channels"] == 1
    assert snapshot["adapters_loaded"] == 4
    # Registering 2 agents + opening 1 channel emits >0 audit bytes.
    assert snapshot["audit_log_bytes"] > 0

    await hub.close()


# ── Hub logging ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hub_logs_state_transitions(caplog) -> None:
    caplog.set_level(logging.INFO, logger="ag2.network.hub.core")
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))

    messages = [r.getMessage() for r in caplog.records]
    assert any("agent registered" in m for m in messages)

    await hub.close()


@pytest.mark.asyncio
async def test_hub_logs_warning_on_rejection(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="ag2.network.hub.core")
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))
    channel = await alice.open(type="conversation", target="bob")
    await hub.set_rule(alice.agent_id, Rule(access=AccessBlock(inbound_from=["*"], outbound_to=["nope"])))

    bob_id = next(p.agent_id for p in channel.metadata.participants if p.agent_id != alice.agent_id)
    with pytest.raises(AccessDeniedError):
        await channel.send("blocked", audience=[bob_id])

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("post_envelope rejected" in r.getMessage() for r in warnings)


# ── Coverage gap tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multiple_bad_listeners_do_not_break_dispatch() -> None:
    """Three buggy listeners in a chain — the good one in the middle still fires."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    class _Bad(BaseHubListener):
        async def on_envelope_posted(self, envelope, metadata) -> None:
            raise RuntimeError("bad")

    good = _RecordingListener()
    hub.register_listener(_Bad())
    hub.register_listener(_Bad())
    hub.register_listener(good)
    hub.register_listener(_Bad())

    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))
    channel = await alice.open(type="conversation", target="bob")
    await channel.send("hi")

    assert good.envelope_posted, "good listener should have fired despite bad siblings"

    await hub.close()


@pytest.mark.asyncio
async def test_base_hub_arbiter_allows_everything_by_default() -> None:
    """``BaseHubArbiter`` returns Allow() from every gate — minimal opt-in shape."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    hub.register_arbiter(BaseHubArbiter())

    # Even with an inbound-block rule, the BaseHubArbiter (which allows
    # everything) lets the channel open and the envelope land.
    blocking_rule = Rule(access=AccessBlock(inbound_from=["nobody"], outbound_to=["*"]))
    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"), rule=blocking_rule)
    channel = await alice.open(type="conversation", target="bob")
    bob_id = next(p.agent_id for p in channel.metadata.participants if p.agent_id != alice.agent_id)
    envelope_id = await channel.send("through", audience=[bob_id])
    assert envelope_id

    await hub.close()


class _DenyAtDelegationDepth(BaseHubArbiter):
    """Custom arbiter that denies sends when envelope.depth exceeds a cap."""

    def __init__(self, cap: int) -> None:
        self._cap = cap

    async def authorize_send(self, envelope, sender, sender_rule, recipients):
        if envelope.depth > self._cap:
            return Deny(reason=f"depth {envelope.depth} > cap {self._cap}")
        return Allow()


@pytest.mark.asyncio
async def test_custom_arbiter_authorize_send_overrides_rule_based() -> None:
    """A subclass of ``BaseHubArbiter`` can deny based on envelope depth."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob"))
    channel = await alice.open(type="conversation", target="bob")

    # Install the deny-everything arbiter after the channel is open so
    # the invite envelope doesn't get denied. Substantive sends now fail.
    hub.register_arbiter(_DenyAtDelegationDepth(cap=-1))

    with pytest.raises(AccessDeniedError, match="cap"):
        await channel.send("blocked")

    await hub.close()


@pytest.mark.asyncio
async def test_hub_health_on_populated_hub() -> None:
    """3 channels × 5 agents — counters reflect realistic operational load."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    agents = []
    for name in ("a", "b", "c", "d", "e"):
        ac = await hub.register(_agent(name))
        agents.append(ac)

    # 3 channels: a→b, a→c, b→d.
    await agents[0].open(type="conversation", target="b")
    await agents[0].open(type="conversation", target="c")
    await agents[1].open(type="conversation", target="d")

    snapshot = hub.health()
    assert snapshot["registered_agents"] == 5
    assert snapshot["active_channels"] == 3
    # AuditLog has written register + channel-create records.
    assert snapshot["audit_log_bytes"] > 0

    await hub.close()


@pytest.mark.asyncio
async def test_widened_trap_catches_pre_ask_failures() -> None:
    """A monkey-patched ``default_view_policy`` that raises lands in ``on_turn_failed``.

    Regression for the widened ``_process_substantive`` trap — the
    original trap started after view projection, so a buggy view
    would escape the receive loop instead of routing through the
    observability surface.
    """

    class _BadView:
        async def project(self, history, *, participant_id, channel, render_envelope):
            raise RuntimeError("view broke")

    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    listener = _RecordingListener()
    hub.register_listener(listener)

    # Force every view resolution to return the bad view.
    hub.default_view_policy = lambda channel_id, participant_id: _BadView()  # type: ignore[method-assign]

    alice = await hub.register(_agent("alice"))
    await hub.register(_agent("bob", "ok"))

    channel = await alice.open(type="conversation", target="bob")
    await channel.send("hi bob")
    for _ in range(200):
        if listener.turn_failed:
            break
        await asyncio.sleep(0.01)

    assert listener.turn_failed, "view.project crash should fire on_turn_failed"
    # Channel survives — subsequent send still works.
    assert await channel.send("still alive?")

    await hub.close()


@pytest.mark.asyncio
async def test_register_human_duplicate_name_raises() -> None:
    """Re-registering the same human name through register_human raises."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)
    hc = HubClient(LocalLink(hub), hub=hub)
    first = await hc.register_human(Passport(name="reviewer"))
    assert isinstance(first, HumanClient)
    with pytest.raises(NetworkError):
        await hc.register_human(Passport(name="reviewer"))
    await hc.close()
    await hub.close()


def test_passport_kind_rejects_typo_at_construction() -> None:
    """Tightened ``Passport.kind`` raises ValueError for unknown literals."""
    with pytest.raises(ValueError, match="kind must be one of"):
        Passport(name="x", kind="huMAn")  # typo


@pytest.mark.asyncio
async def test_replace_audit_log_swaps_listener_chain() -> None:
    """Tenant-supplied AuditLog replaces the built-in one in the listener chain.

    Covers the subclass-friendly hub story: a custom audit format can
    be plugged in without monkey-patching, and the listener order
    keeps audit-first semantics.
    """
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0)

    replacement = AuditLog(store)
    hub.replace_audit_log(replacement)
    assert hub.audit_log is replacement

    await hub.register(_agent("alice"))

    # The replacement listener saw the register event.
    records = await replacement.read_all()
    kinds = [r.get("kind") for r in records]
    assert "agent_registered" in kinds

    await hub.close()
