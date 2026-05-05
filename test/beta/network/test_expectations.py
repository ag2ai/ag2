# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Expectation evaluators, violation handlers, and audit log tests.

Three layers covered:

* **Evaluators** (unit) — ``AcksWithinEvaluator``,
  ``ReplyWithinEvaluator``, ``MaxSilenceEvaluator`` against synthesised
  ``ExpectationContext`` inputs.
* **Handlers** (integration) — ``AuditHandler``,
  ``NotifySessionHandler``, ``AutoCloseHandler`` driven by the hub's
  manual ``_expectation_tick()`` call with a controllable clock.
* **Audit log** — ``register`` / ``unregister`` / ``set_*`` write the
  expected records via ``AuditLog.read_all()``.

Time control: tests use a ``_MockClock`` so the threshold logic is
exercised deterministically without sleeping. The sweeper interval is
disabled (``expectation_sweep_interval=0``); tests call
``hub._expectation_tick()`` explicitly.
"""

from datetime import datetime

import pytest

from autogen.beta import Agent
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    EV_EXPECTATION_VIOLATED,
    EV_TEXT,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
    Rule,
)
from autogen.beta.network.adapters.consulting import CONSULTING_TYPE
from autogen.beta.network.adapters.conversation import (
    CONVERSATION_TYPE,
    ConversationAdapter,
)
from autogen.beta.network.hub import (
    AUDIT_KIND_AGENT_REGISTERED,
    AUDIT_KIND_AGENT_UNREGISTERED,
    AUDIT_KIND_EXPECTATION_VIOLATED,
    AUDIT_KIND_RESUME_SET,
    AUDIT_KIND_RULE_SET,
    AUDIT_KIND_SKILL_SET,
    AcksWithinEvaluator,
    AuditLog,
    ExpectationContext,
    MaxSilenceEvaluator,
    ReplyWithinEvaluator,
)
from autogen.beta.network.session import (
    Expectation,
    Participant,
    ParticipantRole,
    ParticipantSchema,
    SessionManifest,
    SessionMetadata,
    SessionState,
)
from autogen.beta.testing import TestConfig

from ._helpers import _MockClock


def _agent(name: str, *events: object) -> Agent:
    return Agent(name=name, config=TestConfig(*events))


async def _silent_handler(_envelope: Envelope) -> None:
    """No-op handler that ignores every envelope.

    Used to drive expectation paths that depend on a participant
    *not* responding (e.g. ``acks_within`` / ``reply_within`` tests).
    """


def _conv_metadata(
    *,
    state: SessionState = SessionState.ACTIVE,
    created_at: str = "2026-01-01T00:00:00+00:00",
    pending_acks: list[str] | None = None,
) -> SessionMetadata:
    """Build a 2-party conversation session metadata for evaluator tests."""
    manifest = ConversationAdapter().manifest
    return SessionMetadata(
        session_id="s1",
        manifest=manifest,
        creator_id="alice",
        participants=[
            Participant(agent_id="alice", role=ParticipantRole.INITIATOR, order=0),
            Participant(agent_id="bob", role=ParticipantRole.RESPONDENT, order=1),
        ],
        state=state,
        created_at=created_at,
        pending_acks=list(pending_acks or []),
    )


def _envelope(
    sender: str,
    text: str,
    created_at: str,
    *,
    audience: list[str] | None = None,
) -> Envelope:
    env = Envelope(
        envelope_id=f"env-{sender}-{text}",
        session_id="s1",
        sender_id=sender,
        audience=audience,
        event_type=EV_TEXT,
        event_data={"text": text},
        created_at=created_at,
    )
    return env


def _ctx(
    metadata: SessionMetadata,
    *,
    wal: list[Envelope] | None = None,
    now: str = "2026-01-01T00:01:00+00:00",
) -> ExpectationContext:
    now_dt = datetime.fromisoformat(now)
    return ExpectationContext(
        metadata=metadata,
        state=None,
        wal=wal or [],
        now_iso=now,
        now_seconds=now_dt.timestamp(),
    )


# ── Evaluator unit tests ────────────────────────────────────────────────────


class TestAcksWithinEvaluator:
    def test_fires_after_threshold_with_pending_acks(self) -> None:
        metadata = _conv_metadata(
            state=SessionState.PENDING,
            created_at="2026-01-01T00:00:00+00:00",
            pending_acks=["bob"],
        )
        evaluator = AcksWithinEvaluator()
        expectation = Expectation(
            name="acks_within",
            on_violation="auto_close",
            params={"seconds": 30},
        )
        violation = evaluator.evaluate(expectation, _ctx(metadata, now="2026-01-01T00:01:00+00:00"))
        assert violation is not None
        assert violation.violator_ids == ["bob"]
        assert violation.detail["threshold_seconds"] == 30

    def test_silent_within_threshold(self) -> None:
        metadata = _conv_metadata(
            state=SessionState.PENDING,
            pending_acks=["bob"],
        )
        evaluator = AcksWithinEvaluator()
        expectation = Expectation(
            name="acks_within",
            on_violation="auto_close",
            params={"seconds": 60},
        )
        # 30s elapsed, threshold 60s → no violation
        violation = evaluator.evaluate(expectation, _ctx(metadata, now="2026-01-01T00:00:30+00:00"))
        assert violation is None

    def test_silent_when_no_pending_acks(self) -> None:
        metadata = _conv_metadata(
            state=SessionState.PENDING,
            pending_acks=[],
        )
        evaluator = AcksWithinEvaluator()
        expectation = Expectation(name="acks_within", on_violation="auto_close", params={"seconds": 30})
        violation = evaluator.evaluate(expectation, _ctx(metadata, now="2026-01-01T00:10:00+00:00"))
        assert violation is None

    def test_silent_when_session_active(self) -> None:
        # Active session (acks already collected) — evaluator skips.
        metadata = _conv_metadata(state=SessionState.ACTIVE, pending_acks=[])
        evaluator = AcksWithinEvaluator()
        expectation = Expectation(name="acks_within", on_violation="auto_close", params={"seconds": 30})
        violation = evaluator.evaluate(expectation, _ctx(metadata, now="2026-01-01T00:10:00+00:00"))
        assert violation is None


class TestReplyWithinEvaluator:
    def test_fires_when_addressed_participant_silent(self) -> None:
        metadata = _conv_metadata()
        wal = [_envelope("alice", "hi bob", "2026-01-01T00:00:00+00:00", audience=["bob"])]
        evaluator = ReplyWithinEvaluator()
        expectation = Expectation(
            name="reply_within",
            on_violation="audit",
            params={"seconds": 60},
        )
        violation = evaluator.evaluate(expectation, _ctx(metadata, wal=wal, now="2026-01-01T00:02:00+00:00"))
        assert violation is not None
        assert violation.violator_ids == ["bob"]

    def test_silent_when_reply_within_threshold(self) -> None:
        # Tight timing: bob replied 15s ago; alice's window to follow up
        # hasn't expired yet either.
        metadata = _conv_metadata()
        wal = [
            _envelope("alice", "hi bob", "2026-01-01T00:00:00+00:00", audience=["bob"]),
            _envelope("bob", "hi back", "2026-01-01T00:00:30+00:00"),
        ]
        evaluator = ReplyWithinEvaluator()
        expectation = Expectation(name="reply_within", on_violation="audit", params={"seconds": 60})
        # Now = 00:00:45 → bob's reply is 15s old; alice's outstanding
        # incoming (bob's broadcast) is 15s old — both within 60s window.
        violation = evaluator.evaluate(expectation, _ctx(metadata, wal=wal, now="2026-01-01T00:00:45+00:00"))
        assert violation is None

    def test_originally_addressed_party_not_violator_after_reply(self) -> None:
        # Bob has answered — even past the threshold he is not flagged
        # because his outbound is newer than his most recent inbound.
        metadata = _conv_metadata()
        wal = [
            _envelope("alice", "hi bob", "2026-01-01T00:00:00+00:00", audience=["bob"]),
            _envelope("bob", "hi back", "2026-01-01T00:00:30+00:00", audience=["alice"]),
        ]
        evaluator = ReplyWithinEvaluator()
        expectation = Expectation(name="reply_within", on_violation="audit", params={"seconds": 60})
        # Long after — alice will be flagged but bob will not.
        violation = evaluator.evaluate(expectation, _ctx(metadata, wal=wal, now="2026-01-01T00:10:00+00:00"))
        assert violation is not None
        assert "bob" not in violation.violator_ids
        assert "alice" in violation.violator_ids


class TestMaxSilenceEvaluator:
    def test_fires_when_session_silent_past_threshold(self) -> None:
        metadata = _conv_metadata(created_at="2026-01-01T00:00:00+00:00")
        wal = [_envelope("alice", "hello", "2026-01-01T00:00:30+00:00")]
        evaluator = MaxSilenceEvaluator()
        expectation = Expectation(
            name="max_silence",
            on_violation="audit",
            params={"seconds": 60},
        )
        violation = evaluator.evaluate(expectation, _ctx(metadata, wal=wal, now="2026-01-01T00:02:00+00:00"))
        assert violation is not None
        assert violation.violator_ids == []  # session-wide

    def test_silent_when_recent_activity(self) -> None:
        metadata = _conv_metadata(created_at="2026-01-01T00:00:00+00:00")
        wal = [_envelope("alice", "hello", "2026-01-01T00:01:30+00:00")]
        evaluator = MaxSilenceEvaluator()
        expectation = Expectation(name="max_silence", on_violation="audit", params={"seconds": 60})
        violation = evaluator.evaluate(expectation, _ctx(metadata, wal=wal, now="2026-01-01T00:02:00+00:00"))
        assert violation is None


# ── Handler integration tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_close_handler_terminates_session_with_audit() -> None:
    """Consulting's ``acks_within(30s, auto_close)`` fires after 30s,
    transitions session to CLOSED, records to audit log."""
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    store = MemoryKnowledgeStore()
    hub = await Hub.open(
        store,
        clock=clock,
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
        invite_ack_timeout=300.0,  # long enough that the sweeper closes first
    )
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    # Bob registers without the auto-ack default handler — install a
    # silent handler explicitly so invites are never acked.
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume(), attach_plugin=False)
    bob.on_envelope(_silent_handler)

    # Open in background; the sweeper auto_closes via ProtocolError on the waiter.
    import asyncio as _asyncio

    open_task = _asyncio.create_task(alice.open(type=CONSULTING_TYPE, target=bob.agent_id))
    # Let the invite dispatch.
    await _asyncio.sleep(0.05)

    # Advance past the 30s acks_within threshold and tick.
    clock.advance(45)
    await hub._expectation_tick()

    # Open fails because the session was auto-closed.
    with pytest.raises(Exception):
        await open_task

    # Find the session id from cached state.
    sessions = list(hub._sessions.values())
    assert len(sessions) == 1
    session_id = sessions[0].session_id
    final = await hub.get_session(session_id)
    assert final.state == SessionState.CLOSED
    assert "expectation_violated:acks_within" in final.close_reason

    audit = await hub._audit_log.read_all()
    kinds = [r["kind"] for r in audit]
    assert AUDIT_KIND_EXPECTATION_VIOLATED in kinds

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_audit_handler_records_without_envelope_or_close() -> None:
    """``audit`` handler logs the violation and nothing else."""
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, clock=clock, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    session = await alice.open(type=CONVERSATION_TYPE, target=bob.agent_id)
    pre_audit = len(await hub._audit_log.read_all())

    # Conversation declares max_silence(3600s, audit). Advance 1h+.
    clock.advance(3700)
    await hub._expectation_tick()

    # Session still ACTIVE; no EV_EXPECTATION_VIOLATED in WAL.
    state = await hub.get_session(session.session_id)
    assert state.state == SessionState.ACTIVE
    wal = await hub.read_wal(session.session_id)
    assert not any(e.event_type == EV_EXPECTATION_VIOLATED for e in wal)

    # Audit log has one new violation entry.
    audit = await hub._audit_log.read_all()
    new_records = audit[pre_audit:]
    violation_records = [r for r in new_records if r["kind"] == AUDIT_KIND_EXPECTATION_VIOLATED]
    assert len(violation_records) == 1
    assert violation_records[0]["expectation"] == "max_silence"

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_notify_session_handler_broadcasts_envelope() -> None:
    """``notify_session`` handler audits + posts EV_EXPECTATION_VIOLATED."""
    # Use a custom session with a notify_session expectation we can drive.
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, clock=clock, ttl_sweep_interval=0, expectation_sweep_interval=0)

    # Register a custom adapter with notify_session on max_silence so we
    # don't have to wait for conversation's 1h default.
    from autogen.beta.network.adapters.conversation import ConversationAdapter
    from autogen.beta.network.views.builtin import WindowedSummary

    class _NotifyAdapter(ConversationAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.manifest = SessionManifest(
                type="conversation_notify",
                version=1,
                participants=ParticipantSchema(min=2, max=2, roles=["initiator", "respondent"]),
                knobs_schema={},
                default_view_policy=WindowedSummary.name,
                expectations=[
                    Expectation(
                        name="max_silence",
                        on_violation="notify_session",
                        params={"seconds": 60},
                    ),
                ],
            )

    hub.register_adapter(_NotifyAdapter())

    link = LocalLink(hub)
    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    session = await alice.open(type="conversation_notify", target=bob.agent_id)

    clock.advance(120)
    await hub._expectation_tick()

    wal = await hub.read_wal(session.session_id)
    violation_envelopes = [e for e in wal if e.event_type == EV_EXPECTATION_VIOLATED]
    assert len(violation_envelopes) == 1
    assert violation_envelopes[0].event_data["expectation"] == "max_silence"

    # Session still ACTIVE — notify_session does not close.
    state = await hub.get_session(session.session_id)
    assert state.state == SessionState.ACTIVE

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_violation_dedup_within_session_lifetime() -> None:
    """Two consecutive ticks fire the same violation only once."""
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, clock=clock, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    bob_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    bob = await bob_hc.register(_agent("bob"), Passport(name="bob"), Resume())

    await alice.open(type=CONVERSATION_TYPE, target=bob.agent_id)
    pre_audit = len(await hub._audit_log.read_all())

    clock.advance(3700)
    await hub._expectation_tick()
    await hub._expectation_tick()
    await hub._expectation_tick()

    audit = await hub._audit_log.read_all()
    violations = [r for r in audit[pre_audit:] if r["kind"] == AUDIT_KIND_EXPECTATION_VIOLATED]
    assert len(violations) == 1  # deduped across 3 ticks

    await alice_hc.close()
    await bob_hc.close()
    await hub.close()


# ── Audit log tests for identity changes ────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_log_records_register_and_unregister() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())
    await hub.unregister(alice.agent_id)

    audit = await hub._audit_log.read_all()
    kinds = [r["kind"] for r in audit]
    assert AUDIT_KIND_AGENT_REGISTERED in kinds
    assert AUDIT_KIND_AGENT_UNREGISTERED in kinds

    await alice_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_audit_log_records_set_resume_skill_rule() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    link = LocalLink(hub)

    alice_hc = HubClient(link, hub=hub)
    alice = await alice_hc.register(_agent("alice"), Passport(name="alice"), Resume())

    pre_audit = len(await hub._audit_log.read_all())

    await hub.set_resume(alice.agent_id, Resume(summary="updated"))
    await hub.set_skill(alice.agent_id, "# alice's skill")
    await hub.set_rule(alice.agent_id, Rule())

    audit = await hub._audit_log.read_all()
    new_kinds = [r["kind"] for r in audit[pre_audit:]]
    assert AUDIT_KIND_RESUME_SET in new_kinds
    assert AUDIT_KIND_SKILL_SET in new_kinds
    assert AUDIT_KIND_RULE_SET in new_kinds

    await alice_hc.close()
    await hub.close()


@pytest.mark.asyncio
async def test_audit_log_writer_round_trips() -> None:
    """``AuditLog`` writes JSON lines that ``read_all`` round-trips."""
    store = MemoryKnowledgeStore()
    log = AuditLog(store)
    await log.append({"at": "2026-01-01T00:00:00", "kind": "test", "n": 1})
    await log.append({"at": "2026-01-01T00:00:01", "kind": "test", "n": 2})
    records = await log.read_all()
    assert records == [
        {"at": "2026-01-01T00:00:00", "kind": "test", "n": 1},
        {"at": "2026-01-01T00:00:01", "kind": "test", "n": 2},
    ]
