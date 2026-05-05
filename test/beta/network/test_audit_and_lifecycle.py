# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Audit log + lifecycle invariants.

Covers:

* ``delegation_depth`` enforcement at hub.
* Hydrate when an adapter is unregistered no longer raises ``KeyError``
  on the next ``post_envelope`` (clear ``ProtocolError`` instead).
* Sweeper ``auto_close`` does not bleed across sessions in one tick.
* Session lifecycle audit records (``session_created`` / ``_closed``).
* Task lifecycle audit record (``task_terminated``).
"""

import asyncio

import pytest

from autogen.beta import Agent
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    EV_TEXT,
    AccessDeniedError,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    ProtocolError,
    Resume,
    Rule,
)
from autogen.beta.network.adapters.conversation import ConversationAdapter
from autogen.beta.network.hub.audit import (
    AUDIT_KIND_EXPECTATION_VIOLATED,
    AUDIT_KIND_SESSION_CLOSED,
    AUDIT_KIND_SESSION_CREATED,
    AUDIT_KIND_SESSION_EXPIRED,
    AUDIT_KIND_TASK_TERMINATED,
)
from autogen.beta.network.rule import LimitsBlock
from autogen.beta.network.session import (
    Expectation,
    ParticipantSchema,
    SessionManifest,
)
from autogen.beta.testing import TestConfig

from ._helpers import _MockClock


def _agent(name: str) -> Agent:
    return Agent(name=name, config=TestConfig())


# ── delegation_depth ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_post_envelope_rejects_envelope_above_delegation_depth() -> None:
    """Hub rejects an envelope whose ``depth`` exceeds the sender's
    ``Rule.limits.delegation_depth`` cap. ``0`` disables the check."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    capped_rule = Rule(limits=LimitsBlock(delegation_depth=2))
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(
        _agent("alice"), Passport(name="alice"), Resume(), rule=capped_rule
    )
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    session = await alice.open(type="conversation", target=bob.agent_id)

    # depth=3 exceeds cap of 2.
    too_deep = Envelope(
        session_id=session.session_id,
        sender_id=alice.agent_id,
        audience=[bob.agent_id],
        event_type=EV_TEXT,
        event_data={"text": "hi"},
        depth=3,
    )
    with pytest.raises(AccessDeniedError):
        await hub.post_envelope(too_deep)

    # depth=2 is at the cap → accepted.
    at_cap = Envelope(
        session_id=session.session_id,
        sender_id=alice.agent_id,
        audience=[bob.agent_id],
        event_type=EV_TEXT,
        event_data={"text": "hi"},
        depth=2,
    )
    await hub.post_envelope(at_cap)

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_delegation_depth_zero_disables_cap() -> None:
    """``delegation_depth=0`` means no cap; arbitrary depth is allowed."""
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    no_cap = Rule(limits=LimitsBlock(delegation_depth=0))
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(
        _agent("alice"), Passport(name="alice"), Resume(), rule=no_cap
    )
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    session = await alice.open(type="conversation", target=bob.agent_id)
    deep = Envelope(
        session_id=session.session_id,
        sender_id=alice.agent_id,
        audience=[bob.agent_id],
        event_type=EV_TEXT,
        event_data={"text": "hi"},
        depth=999,
    )
    await hub.post_envelope(deep)  # accepted

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


# ── Hydrate with missing adapter ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_post_envelope_after_hydrate_without_adapter_state_raises_protocol_error() -> None:
    """If a session is loaded by ``hydrate()`` while its adapter is
    missing, the session is kept dormant — its metadata is read-only,
    not added to ``_active_sessions``, and ``_adapter_states`` has no
    entry. If an adapter is later registered (so ``_adapter_for``
    succeeds) and a stale envelope arrives, the hub raises a clear
    ``ProtocolError`` instead of bare ``KeyError``."""
    store = MemoryKnowledgeStore()

    custom_manifest = SessionManifest(
        type="custom_recovery",
        version=1,
        participants=ParticipantSchema(min=2),
        expectations=[],
    )

    class _CustomAdapter(ConversationAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.manifest = custom_manifest

    # First boot: register custom adapter, open session, persist.
    hub1 = await Hub.open(
        store, ttl_sweep_interval=0, expectation_sweep_interval=0
    )
    hub1.register_adapter(_CustomAdapter())
    link1 = LocalLink(hub1)
    alice_hc = HubClient(link1, hub=hub1)
    bob_hc = HubClient(link1, hub=hub1)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())
    session = await alice.open(type="custom_recovery", target=bob.agent_id)
    session_id = session.session_id
    alice_id = alice.agent_id

    await alice_hc.close()
    await bob_hc.close()
    await hub1.close()

    # Second boot: hydrate WITHOUT the custom adapter. Session is
    # dormant: metadata loaded, but no adapter state, not active.
    hub2 = await Hub.open(
        store, ttl_sweep_interval=0, expectation_sweep_interval=0
    )
    assert session_id in hub2._sessions
    assert session_id not in hub2._active_sessions
    assert session_id not in hub2._adapter_states

    # Now register the adapter post-hydrate (e.g. a recovery script).
    # ``_adapter_for`` will find it; ``_adapter_states`` is still empty.
    hub2.register_adapter(_CustomAdapter())

    # The session metadata says it's ACTIVE (carried over from boot 1)
    # but no fold ran. ``post_envelope`` must surface a clear error.
    envelope = Envelope(
        session_id=session_id,
        sender_id=alice_id,
        audience=None,
        event_type=EV_TEXT,
        event_data={"text": "hi"},
    )
    with pytest.raises(ProtocolError, match="no adapter state"):
        await hub2.post_envelope(envelope)

    await hub2.close()


# ── Sweeper auto_close cross-session bleed ──────────────────────────────────


@pytest.mark.asyncio
async def test_expectation_tick_processes_all_sessions_when_one_auto_closes() -> None:
    """When session A's expectation fires ``auto_close``, session B's
    expectations on the same tick are still evaluated.

    Pre-fix: the sweeper ``return``ed out of the whole tick after the
    first ``auto_close``, so sessions later in the iteration order
    waited for the next 10s tick.
    """
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    store = MemoryKnowledgeStore()
    hub = await Hub.open(
        store, clock=clock, ttl_sweep_interval=0, expectation_sweep_interval=0
    )

    # Custom conversation manifest with an aggressive max_silence that
    # auto_closes — so two sessions both violate at the same tick.
    aggressive_manifest = SessionManifest(
        type="conversation_aggressive",
        version=1,
        participants=ParticipantSchema(min=2),
        expectations=[
            Expectation(
                name="max_silence",
                on_violation="auto_close",
                params={"seconds": 60},
            ),
        ],
    )

    class _AggressiveAdapter(ConversationAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.manifest = aggressive_manifest

    hub.register_adapter(_AggressiveAdapter())
    link = LocalLink(hub)

    a_hc = HubClient(link, hub=hub)
    b_hc = HubClient(link, hub=hub)
    c_hc = HubClient(link, hub=hub)
    d_hc = HubClient(link, hub=hub)
    alice = await a_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await b_hc.register(_agent("bob"), Passport(name="bob"), Resume())
    carol = await c_hc.register(_agent("carol"), Passport(name="carol"), Resume())
    dave = await d_hc.register(_agent("dave"), Passport(name="dave"), Resume())

    sess_ab = await alice.open(type="conversation_aggressive", target=bob.agent_id)
    sess_cd = await carol.open(type="conversation_aggressive", target=dave.agent_id)

    # Advance past max_silence — both should violate.
    clock.advance(120)
    await hub._expectation_tick()

    audit = await hub._audit_log.read_all()
    closed_session_ids = {
        r["session_id"]
        for r in audit
        if r["kind"] == AUDIT_KIND_SESSION_CLOSED
    }
    # Both sessions auto-closed in a single tick.
    assert sess_ab.session_id in closed_session_ids
    assert sess_cd.session_id in closed_session_ids

    # Both expectation violations were logged.
    violations = [r for r in audit if r["kind"] == AUDIT_KIND_EXPECTATION_VIOLATED]
    violation_session_ids = {r["session_id"] for r in violations}
    assert sess_ab.session_id in violation_session_ids
    assert sess_cd.session_id in violation_session_ids

    await a_hc.close()
    await b_hc.close()
    await c_hc.close()
    await d_hc.close()
    await hub.close()


# ── Session + task lifecycle audit ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_log_records_session_created_and_closed() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    a_hc = HubClient(link, hub=hub)
    b_hc = HubClient(link, hub=hub)
    alice = await a_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await b_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    pre_audit = len(await hub._audit_log.read_all())
    session = await alice.open(type="conversation", target=bob.agent_id)
    await hub.close_session(session.session_id, reason="explicit")

    audit = await hub._audit_log.read_all()
    new_records = audit[pre_audit:]
    kinds = [r["kind"] for r in new_records]
    assert AUDIT_KIND_SESSION_CREATED in kinds
    assert AUDIT_KIND_SESSION_CLOSED in kinds

    closed_record = next(r for r in new_records if r["kind"] == AUDIT_KIND_SESSION_CLOSED)
    assert closed_record["session_id"] == session.session_id
    assert closed_record["reason"] == "explicit"

    await a_hc.close()
    await b_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_audit_log_records_session_expired_on_ttl_sweep() -> None:
    """``EXPIRED`` transitions emit ``session_expired`` (not ``session_closed``)."""
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    store = MemoryKnowledgeStore()
    hub = await Hub.open(
        store, clock=clock, ttl_sweep_interval=0, expectation_sweep_interval=0
    )
    link = LocalLink(hub)

    a_hc = HubClient(link, hub=hub)
    b_hc = HubClient(link, hub=hub)
    alice = await a_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await b_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    session = await alice.open(
        type="conversation", target=bob.agent_id, ttl="60s"
    )
    pre_audit = len(await hub._audit_log.read_all())
    clock.advance(120)
    await hub.expire_due()

    audit = await hub._audit_log.read_all()
    new_records = audit[pre_audit:]
    expired = [
        r for r in new_records
        if r["kind"] == AUDIT_KIND_SESSION_EXPIRED
        and r["session_id"] == session.session_id
    ]
    assert len(expired) == 1
    assert expired[0]["reason"] == "ttl_expired"

    await a_hc.close()
    await b_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_audit_log_records_task_terminated_on_session_cascade() -> None:
    """Tasks under a closing session cascade to ``EXPIRED`` and emit
    ``task_terminated`` audit records carrying ``capability``."""
    from autogen.beta.task import TaskMetadata, TaskSpec, TaskState

    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    a_hc = HubClient(link, hub=hub)
    b_hc = HubClient(link, hub=hub)
    alice = await a_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await b_hc.register(_agent("bob"), Passport(name="bob"), Resume())
    session = await alice.open(type="conversation", target=bob.agent_id)

    # Plant a non-terminal capability-tagged task under the session.
    task_meta = TaskMetadata(
        task_id="task-cap-1",
        owner_id=bob.agent_id,
        spec=TaskSpec(title="analysing", capability="analysis"),
        state=TaskState.RUNNING,
        created_at="2026-01-01T00:00:00+00:00",
        session_id=session.session_id,
    )
    hub._tasks[task_meta.task_id] = task_meta
    hub._session_tasks.setdefault(session.session_id, set()).add(task_meta.task_id)

    pre_audit = len(await hub._audit_log.read_all())
    await hub.close_session(session.session_id, reason="explicit")

    audit = await hub._audit_log.read_all()
    new_records = audit[pre_audit:]
    task_records = [r for r in new_records if r["kind"] == AUDIT_KIND_TASK_TERMINATED]
    assert len(task_records) == 1
    rec = task_records[0]
    assert rec["task_id"] == task_meta.task_id
    assert rec["capability"] == "analysis"
    assert rec["outcome"] == TaskState.EXPIRED.value

    await a_hc.close()
    await b_hc.close()
    await hub.close()
