# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Hub subclass surface + sweeper extension + latent fixes.

* Subclass ``on_*`` hooks fire alongside externally-registered listeners.
* ``register_sweeper`` spawns a custom periodic worker; ``unregister_sweeper``
  stops it cleanly.
* Open audit-kind set: subclasses can append arbitrary records.
* ``on_inbox_pressure`` fires when the recipient crosses the high-water mark.
* ``TaskMirror`` mirror failures fire ``on_task_event(kind="mirror_failed")``.
"""

import asyncio

import pytest

from ag2 import Agent
from ag2.events import TaskStarted
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    BaseHubListener,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
    Rule,
)
from ag2.network.adapters.consulting import ConsultingAdapter
from ag2.network.adapters.conversation import ConversationAdapter
from ag2.network.rule import InboxBlock, LimitsBlock
from ag2.network.task_mirror import TaskMirror
from ag2.testing import TestConfig


def _agent(name: str, *replies: str) -> Agent:
    return Agent(name=name, config=TestConfig(*replies))


# ── Subclass on_* hooks ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_subclass_on_envelope_posted_fires_without_listener_registration() -> None:
    """A subclass that overrides ``on_envelope_posted`` sees every envelope
    without registering itself as a listener."""

    class _RecordingHub(Hub):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.captured: list = []

        async def on_envelope_posted(self, envelope, metadata) -> None:
            self.captured.append((envelope.event_type, envelope.sender_id))

    store = MemoryKnowledgeStore()
    hub = _RecordingHub(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    # Register adapters manually since we used __init__ directly.
    hub.register_adapter(ConsultingAdapter())
    hub.register_adapter(ConversationAdapter())
    await hub.hydrate()
    await hub.start()

    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())
    channel = await alice.open(type="conversation", target="bob")
    await channel.send("hi")

    assert hub.captured  # subclass override received envelopes

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_subclass_hook_runs_alongside_external_listener() -> None:
    """Both subclass override and externally-registered listener fire."""

    class _SubHub(Hub):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.sub_calls: list = []

        async def on_agent_event(self, agent_id, kind, payload) -> None:
            self.sub_calls.append((kind, agent_id))

    class _Listener(BaseHubListener):
        def __init__(self) -> None:
            self.calls: list = []

        async def on_agent_event(self, agent_id, kind, payload) -> None:
            self.calls.append((kind, agent_id))

    store = MemoryKnowledgeStore()
    hub = _SubHub(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    hub.register_adapter(ConsultingAdapter())
    await hub.hydrate()
    await hub.start()

    listener = _Listener()
    hub.register_listener(listener)

    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())

    # Both saw the registration event.
    assert any(kind == "registered" for kind, _ in hub.sub_calls)
    assert any(kind == "registered" for kind, _ in listener.calls)

    await alice_hc.close()
    await hub.close()


# ── Custom sweeper ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_register_sweeper_runs_periodically() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    ticks: list[float] = []

    async def my_tick() -> None:
        ticks.append(asyncio.get_event_loop().time())

    hub.register_sweeper("test-sweep", interval_seconds=0.05, fn=my_tick)
    await asyncio.sleep(0.16)
    assert len(ticks) >= 2

    await hub.unregister_sweeper("test-sweep")
    snapshot = len(ticks)
    await asyncio.sleep(0.12)
    # After unregister, no more ticks (allow one in-flight straggler).
    assert len(ticks) <= snapshot + 1

    await hub.close()


@pytest.mark.asyncio
async def test_register_sweeper_rejects_duplicate_name() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    async def noop() -> None:
        pass

    hub.register_sweeper("once", interval_seconds=10.0, fn=noop)
    with pytest.raises(ValueError, match="already registered"):
        hub.register_sweeper("once", interval_seconds=5.0, fn=noop)

    await hub.unregister_sweeper("once")
    await hub.close()


@pytest.mark.asyncio
async def test_register_sweeper_rejects_non_positive_interval() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    async def noop() -> None:
        pass

    with pytest.raises(ValueError, match="positive"):
        hub.register_sweeper("bad", interval_seconds=0, fn=noop)
    with pytest.raises(ValueError, match="positive"):
        hub.register_sweeper("bad", interval_seconds=-1.0, fn=noop)

    await hub.close()


# ── Audit kinds open set ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_log_accepts_custom_kinds() -> None:
    """Subclasses / tenants append records with their own ``kind`` values."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    await hub._audit_log.append({
        "at": "2026-01-01T00:00:00+00:00",
        "kind": "my_app.custom_event",
        "detail": "something happened",
    })

    records = await hub._audit_log.read_all()
    kinds = {r["kind"] for r in records}
    assert "my_app.custom_event" in kinds

    await hub.close()


# ── Inbox pressure ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inbox_pressure_fires_on_crossing_high_water() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    fires: list = []

    class _PressureListener(BaseHubListener):
        async def on_inbox_pressure(self, agent_id, pending, cap) -> None:
            fires.append((agent_id, pending, cap))

    hub.register_listener(_PressureListener())

    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)

    # Bob's inbox: cap 10, high_water 3.
    bob_rule = Rule(
        limits=LimitsBlock(
            inbox=InboxBlock(max_pending=10, overflow="reject", high_water=3),
        ),
    )
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume(), rule=bob_rule)
    channel = await alice.open(type="conversation", target="bob")

    # Send 4 substantive envelopes addressed to bob — crosses 3 at the third.
    for _ in range(4):
        await channel.send("ping", audience=[bob.agent_id])

    # Exactly one firing — at the crossing of prev<3 → new>=3.
    assert len(fires) == 1
    fired_agent_id, fired_pending, fired_cap = fires[0]
    assert fired_agent_id == bob.agent_id
    assert fired_pending == 3
    assert fired_cap == 10

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


# ── Task mirror error escalation ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_task_mirror_failure_fires_mirror_failed_event() -> None:
    """When the mirror cannot reach the hub, fire ``on_task_event(mirror_failed)``."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    events: list = []

    class _Listener(BaseHubListener):
        async def on_task_event(self, task_id, kind, payload) -> None:
            events.append((task_id, kind))

    hub.register_listener(_Listener())

    # Force observe_task to raise so the escalation path runs.
    async def _broken_observe(metadata) -> None:
        raise RuntimeError("simulated hub failure")

    hub.observe_task = _broken_observe  # type: ignore[assignment]

    mirror = TaskMirror(hub=hub, owner_id="ghost-agent", channel_id="ghost-channel")
    await mirror._on_started(TaskStarted(task_id="task-x", objective="test"))

    assert any(kind == "mirror_failed" for _, kind in events)

    await hub.close()
