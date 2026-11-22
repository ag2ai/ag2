# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Expectation evaluators, violation handlers, and audit log tests.

Three layers covered:

* **Evaluators** (unit) — ``AcksWithinEvaluator``,
  ``ReplyWithinEvaluator``, ``MaxSilenceEvaluator`` against synthesised
  ``ExpectationContext`` inputs.
* **Handlers** (integration) — ``AuditHandler``,
  ``NotifyChannelHandler``, ``AutoCloseHandler`` driven by the hub's
  manual ``_expectation_tick()`` call with a controllable clock.
* **Audit log** — ``register`` / ``unregister`` / ``set_*`` write the
  expected records via ``AuditLog.read_all()``.

Time control: tests use a ``_MockClock`` so the threshold logic is
exercised deterministically without sleeping. The sweeper interval is
disabled (``expectation_sweep_interval=0``); tests call
``hub._expectation_tick()`` explicitly.
"""

import asyncio
from datetime import datetime

import pytest

from ag2 import Agent
from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_EXPECTATION_VIOLATED,
    EV_TEXT,
    WORKFLOW_TYPE,
    AgentTarget,
    Envelope,
    ExpectedTurn,
    Hub,
    Resume,
    Rule,
    TransitionGraph,
)
from ag2.network.adapters.consulting import CONSULTING_TYPE
from ag2.network.adapters.conversation import (
    CONVERSATION_TYPE,
    ConversationAdapter,
)
from ag2.network.adapters.discussion import DISCUSSION_TYPE
from ag2.network.channel import (
    ChannelManifest,
    ChannelMetadata,
    ChannelState,
    Expectation,
    Participant,
    ParticipantRole,
    ParticipantSchema,
)
from ag2.network.hub import (
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
    TurnWithinEvaluator,
    Violation,
    WarnHandler,
)
from ag2.network.views.builtin import WindowedSummary
from ag2.testing import TestConfig

from ._helpers import _MockClock


def _agent(name: str, *events: object) -> Agent:
    return Agent(name=name, config=TestConfig(*events))


async def _silent_handler(_envelope: Envelope) -> None:
    """No-op handler that ignores every envelope.

    Used to drive expectation paths that depend on a participant
    *not* responding (e.g. ``acks_within`` / ``reply_within`` tests).
    """


class _CaptureHandler:
    def __init__(self) -> None:
        self.received: list[Envelope] = []

    async def __call__(self, envelope: Envelope) -> None:
        self.received.append(envelope)


def _conv_metadata(
    *,
    state: ChannelState = ChannelState.ACTIVE,
    created_at: str = "2026-01-01T00:00:00+00:00",
    pending_acks: list[str] | None = None,
) -> ChannelMetadata:
    """Build a 2-party conversation channel metadata for evaluator tests."""
    manifest = ConversationAdapter().manifest
    return ChannelMetadata(
        channel_id="s1",
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
        channel_id="s1",
        sender_id=sender,
        audience=audience,
        event_type=EV_TEXT,
        event_data={"text": text},
        created_at=created_at,
    )
    return env


def _ctx(
    metadata: ChannelMetadata,
    *,
    wal: list[Envelope] | None = None,
    now: str = "2026-01-01T00:01:00+00:00",
    expected_turn: ExpectedTurn | None = None,
) -> ExpectationContext:
    now_dt = datetime.fromisoformat(now)
    return ExpectationContext(
        metadata=metadata,
        state=None,
        wal=wal or [],
        now_iso=now,
        now_seconds=now_dt.timestamp(),
        expected_turn=expected_turn,
    )


# ── Evaluator unit tests ────────────────────────────────────────────────────


class TestAcksWithinEvaluator:
    def test_fires_after_threshold_with_pending_acks(self) -> None:
        metadata = _conv_metadata(
            state=ChannelState.PENDING,
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
            state=ChannelState.PENDING,
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
            state=ChannelState.PENDING,
            pending_acks=[],
        )
        evaluator = AcksWithinEvaluator()
        expectation = Expectation(name="acks_within", on_violation="auto_close", params={"seconds": 30})
        violation = evaluator.evaluate(expectation, _ctx(metadata, now="2026-01-01T00:10:00+00:00"))
        assert violation is None

    def test_silent_when_channel_active(self) -> None:
        # Active channel (acks already collected) — evaluator skips.
        metadata = _conv_metadata(state=ChannelState.ACTIVE, pending_acks=[])
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
    def test_fires_when_channel_silent_past_threshold(self) -> None:
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
        assert violation.violator_ids == []  # channel-wide

    def test_silent_when_recent_activity(self) -> None:
        metadata = _conv_metadata(created_at="2026-01-01T00:00:00+00:00")
        wal = [_envelope("alice", "hello", "2026-01-01T00:01:30+00:00")]
        evaluator = MaxSilenceEvaluator()
        expectation = Expectation(name="max_silence", on_violation="audit", params={"seconds": 60})
        violation = evaluator.evaluate(expectation, _ctx(metadata, wal=wal, now="2026-01-01T00:02:00+00:00"))
        assert violation is None


# ── Handler integration tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_close_handler_terminates_channel_with_audit() -> None:
    """Consulting's ``acks_within(30s, auto_close)`` fires after 30s,
    transitions channel to CLOSED, records to audit log."""
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    store = MemoryKnowledgeStore()
    hub = await Hub.open(
        store,
        clock=clock,
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
        invite_ack_timeout=300.0,  # long enough that the sweeper closes first
    )

    alice = await hub.register(_agent("alice"))
    # Bob registers without the auto-ack default handler — install a
    # silent handler explicitly so invites are never acked.
    bob = await hub.register(_agent("bob"), attach_plugin=False)
    bob.on_envelope(_silent_handler)

    # Open in background; the sweeper auto_closes via ProtocolError on the waiter.

    open_task = asyncio.create_task(alice.open(type=CONSULTING_TYPE, target=bob.agent_id))
    # Let the invite dispatch.
    await asyncio.sleep(0.05)

    # Advance past the 30s acks_within threshold and tick.
    clock.advance(45)
    await hub._expectation_tick()

    # Open fails because the channel was auto-closed.
    with pytest.raises(Exception):
        await open_task

    # Find the channel id from cached state.
    channels = list(hub._channels.values())
    assert len(channels) == 1
    channel_id = channels[0].channel_id
    final = await hub.get_channel(channel_id)
    assert final.state == ChannelState.CLOSED
    assert "expectation_violated:acks_within" in final.close_reason

    audit = await hub._audit_log.read_all()
    kinds = [r["kind"] for r in audit]
    assert AUDIT_KIND_EXPECTATION_VIOLATED in kinds

    await hub.close()


@pytest.mark.asyncio
async def test_audit_handler_records_without_envelope_or_close() -> None:
    """``audit`` handler logs the violation and nothing else."""
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, clock=clock, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    channel = await alice.open(type=CONVERSATION_TYPE, target=bob.agent_id)
    pre_audit = len(await hub._audit_log.read_all())

    # Conversation declares max_silence(3600s, audit). Advance 1h+.
    clock.advance(3700)
    await hub._expectation_tick()

    # Channel still ACTIVE; no EV_EXPECTATION_VIOLATED in WAL.
    state = await hub.get_channel(channel.channel_id)
    assert state.state == ChannelState.ACTIVE
    wal = await hub.read_wal(channel.channel_id)
    assert not any(e.event_type == EV_EXPECTATION_VIOLATED for e in wal)

    # Audit log has one new violation entry.
    audit = await hub._audit_log.read_all()
    new_records = audit[pre_audit:]
    violation_records = [r for r in new_records if r["kind"] == AUDIT_KIND_EXPECTATION_VIOLATED]
    assert len(violation_records) == 1
    assert violation_records[0]["expectation"] == "max_silence"

    await hub.close()


@pytest.mark.asyncio
async def test_notify_channel_handler_broadcasts_envelope() -> None:
    """``notify_channel`` handler audits + posts EV_EXPECTATION_VIOLATED."""
    # Use a custom channel with a notify_channel expectation we can drive.
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, clock=clock, ttl_sweep_interval=0, expectation_sweep_interval=0)

    # Register a custom adapter with notify_channel on max_silence so we
    # don't have to wait for conversation's 1h default.

    class _NotifyAdapter(ConversationAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.manifest = ChannelManifest(
                type="conversation_notify",
                version=1,
                participants=ParticipantSchema(min=2, max=2, roles=["initiator", "respondent"]),
                knobs_schema={},
                default_view_policy=WindowedSummary.name,
                expectations=[
                    Expectation(
                        name="max_silence",
                        on_violation="notify_channel",
                        params={"seconds": 60},
                    ),
                ],
            )

    hub.register_adapter(_NotifyAdapter())

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    channel = await alice.open(type="conversation_notify", target=bob.agent_id)

    clock.advance(120)
    await hub._expectation_tick()

    wal = await hub.read_wal(channel.channel_id)
    violation_envelopes = [e for e in wal if e.event_type == EV_EXPECTATION_VIOLATED]
    assert len(violation_envelopes) == 1
    assert violation_envelopes[0].event_data["expectation"] == "max_silence"

    # Channel still ACTIVE — notify_channel does not close.
    state = await hub.get_channel(channel.channel_id)
    assert state.state == ChannelState.ACTIVE

    await hub.close()


@pytest.mark.asyncio
async def test_violation_dedup_within_channel_lifetime() -> None:
    """Two consecutive ticks fire the same violation only once."""
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, clock=clock, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    await alice.open(type=CONVERSATION_TYPE, target=bob.agent_id)
    pre_audit = len(await hub._audit_log.read_all())

    clock.advance(3700)
    await hub._expectation_tick()
    await hub._expectation_tick()
    await hub._expectation_tick()

    audit = await hub._audit_log.read_all()
    violations = [r for r in audit[pre_audit:] if r["kind"] == AUDIT_KIND_EXPECTATION_VIOLATED]
    assert len(violations) == 1  # deduped across 3 ticks

    await hub.close()


# ── Audit log tests for identity changes ────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_log_records_register_and_unregister() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))
    await hub.unregister(alice.agent_id)

    audit = await hub._audit_log.read_all()
    kinds = [r["kind"] for r in audit]
    assert AUDIT_KIND_AGENT_REGISTERED in kinds
    assert AUDIT_KIND_AGENT_UNREGISTERED in kinds

    await hub.close()


@pytest.mark.asyncio
async def test_audit_log_records_set_resume_skill_rule() -> None:
    store = MemoryKnowledgeStore()
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)

    alice = await hub.register(_agent("alice"))

    pre_audit = len(await hub._audit_log.read_all())

    await hub.set_resume(alice.agent_id, Resume(summary="updated"))
    await hub.set_skill(alice.agent_id, "# alice's skill")
    await hub.set_rule(alice.agent_id, Rule())

    audit = await hub._audit_log.read_all()
    new_kinds = [r["kind"] for r in audit[pre_audit:]]
    assert AUDIT_KIND_RESUME_SET in new_kinds
    assert AUDIT_KIND_SKILL_SET in new_kinds
    assert AUDIT_KIND_RULE_SET in new_kinds

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


class TestTurnWithinEvaluator:
    def test_fresh_channel_anchors_on_creation_and_fires_after_threshold(self) -> None:
        metadata = _conv_metadata(created_at="2026-01-01T00:00:00+00:00")
        expectation = Expectation(name="turn_within", on_violation="warn", params={"seconds": 60})
        expected = ExpectedTurn(agent_id="alice")

        violation = TurnWithinEvaluator().evaluate(
            expectation,
            _ctx(metadata, now="2026-01-01T00:01:01+00:00", expected_turn=expected),
        )

        assert violation is not None
        assert violation.violator_ids == ["alice"]
        assert violation.detail["threshold_seconds"] == 60
        assert violation.detail["elapsed_seconds"] == 61

    def test_triggering_envelope_resets_timer_for_new_expected_speaker(self) -> None:
        metadata = _conv_metadata(created_at="2026-01-01T00:00:00+00:00")
        first = _envelope("alice", "first", "2026-01-01T00:00:00+00:00")
        latest = _envelope("bob", "latest", "2026-01-01T00:00:50+00:00")
        expectation = Expectation(name="turn_within", on_violation="warn", params={"seconds": 60})
        expected = ExpectedTurn(agent_id="alice", triggering_envelope_id=latest.envelope_id)

        before_new_deadline = TurnWithinEvaluator().evaluate(
            expectation,
            _ctx(
                metadata,
                wal=[first, latest],
                now="2026-01-01T00:01:20+00:00",
                expected_turn=expected,
            ),
        )
        at_new_deadline = TurnWithinEvaluator().evaluate(
            expectation,
            _ctx(
                metadata,
                wal=[first, latest],
                now="2026-01-01T00:01:50+00:00",
                expected_turn=expected,
            ),
        )

        assert before_new_deadline is None
        assert at_new_deadline is not None
        assert at_new_deadline.violator_ids == ["alice"]

    def test_missing_triggering_envelope_does_not_fall_back_to_creation(self, caplog) -> None:
        metadata = _conv_metadata(created_at="2026-01-01T00:00:00+00:00")
        expectation = Expectation(name="turn_within", on_violation="warn", params={"seconds": 60})
        expected = ExpectedTurn(agent_id="nobody", triggering_envelope_id="missing-trigger")

        violation = TurnWithinEvaluator().evaluate(
            expectation,
            _ctx(metadata, now="2026-01-01T01:00:00+00:00", expected_turn=expected),
        )

        assert violation is None
        assert "s1" in caplog.text
        assert "missing-trigger" in caplog.text

    def test_exact_threshold_fires_and_just_under_is_silent(self) -> None:
        metadata = _conv_metadata(created_at="2026-01-01T00:00:00+00:00")
        expectation = Expectation(name="turn_within", on_violation="warn", params={"seconds": 60})
        expected = ExpectedTurn(agent_id="alice")

        exact = TurnWithinEvaluator().evaluate(
            expectation,
            _ctx(metadata, now="2026-01-01T00:01:00+00:00", expected_turn=expected),
        )
        under = TurnWithinEvaluator().evaluate(
            expectation,
            _ctx(metadata, now="2026-01-01T00:00:59.999000+00:00", expected_turn=expected),
        )

        assert exact is not None
        assert exact.violator_ids == ["alice"]
        assert under is None

    def test_silent_when_channel_not_active(self) -> None:
        metadata = _conv_metadata(state=ChannelState.PENDING)
        expectation = Expectation(name="turn_within", on_violation="warn", params={"seconds": 60})
        expected = ExpectedTurn(agent_id="alice")

        assert TurnWithinEvaluator().evaluate(expectation, _ctx(metadata, expected_turn=expected)) is None

    def test_silent_when_no_expected_turn(self) -> None:
        metadata = _conv_metadata()
        expectation = Expectation(name="turn_within", on_violation="warn", params={"seconds": 60})

        assert TurnWithinEvaluator().evaluate(expectation, _ctx(metadata, expected_turn=None)) is None


@pytest.mark.asyncio
async def test_warn_handler_delivers_violation_only_to_violator() -> None:
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        clock=clock,
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    assert isinstance(hub._expectation_evaluators["turn_within"], TurnWithinEvaluator)
    assert isinstance(hub._violation_handlers["warn"], WarnHandler)

    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))
    graph = TransitionGraph.round_robin([alice.agent_id, bob.agent_id])
    channel = await alice.open(
        type=WORKFLOW_TYPE,
        target=[bob.agent_id],
        knobs={"graph": graph.to_dict()},
    )

    bob_capture = _CaptureHandler()
    alice_capture = _CaptureHandler()
    bob.on_envelope(bob_capture)
    alice.on_envelope(alice_capture)
    await hub.post_envelope(
        Envelope(
            channel_id=channel.channel_id,
            sender_id=alice.agent_id,
            audience=None,
            event_type=EV_TEXT,
            event_data={"text": "hello"},
        )
    )
    clock.advance(121)
    await hub._expectation_tick()
    await asyncio.sleep(0.05)

    wal = await hub.read_wal(channel.channel_id)
    violations = [e for e in wal if e.event_type == EV_EXPECTATION_VIOLATED]
    assert len(violations) == 1
    assert violations[0].audience == [bob.agent_id]
    assert violations[0].event_data["violators"] == [bob.agent_id]
    assert any(e.envelope_id == violations[0].envelope_id for e in bob_capture.received)
    assert not any(e.event_type == EV_EXPECTATION_VIOLATED for e in alice_capture.received)

    await hub.close()


@pytest.mark.asyncio
async def test_warn_handler_noops_on_missing_or_terminal_channel() -> None:
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        clock=clock,
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    try:
        alice = await hub.register(_agent("alice"))
        bob = await hub.register(_agent("bob"))
        graph = TransitionGraph.round_robin([alice.agent_id, bob.agent_id])
        channel = await alice.open(
            type=WORKFLOW_TYPE,
            target=[bob.agent_id],
            knobs={"graph": graph.to_dict()},
        )
        await hub.close_channel(channel.channel_id)

        handler = WarnHandler()
        violation = Violation(
            expectation=Expectation(name="turn_within", on_violation="warn", params={"seconds": 120}),
            violator_ids=[bob.agent_id],
            detail={"elapsed_seconds": 121, "threshold_seconds": 120},
        )

        # Closed channel: metadata.is_terminal() is True -> early return, no post.
        await handler.handle(hub, channel.channel_id, violation)
        # Missing channel: metadata is None -> early return, no post.
        await handler.handle(hub, "missing-channel", violation)

        wal = await hub.read_wal(channel.channel_id)
        assert not any(e.event_type == EV_EXPECTATION_VIOLATED for e in wal)
    finally:
        await hub.close()


@pytest.mark.asyncio
async def test_workflow_turn_timeout_auto_closes_channel() -> None:
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        clock=clock,
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))
    graph = TransitionGraph.round_robin([alice.agent_id, bob.agent_id])
    channel = await alice.open(
        type=WORKFLOW_TYPE,
        target=[bob.agent_id],
        knobs={"graph": graph.to_dict()},
    )
    bob.on_envelope(_silent_handler)
    await hub.post_envelope(
        Envelope(
            channel_id=channel.channel_id,
            sender_id=alice.agent_id,
            audience=None,
            event_type=EV_TEXT,
            event_data={"text": "bob, your turn"},
        )
    )
    assert getattr(hub._adapter_states[channel.channel_id], "expected_next_speaker") == bob.agent_id

    clock.advance(601)
    await hub._expectation_tick()

    final = await hub.get_channel(channel.channel_id)
    assert final.state == ChannelState.CLOSED
    assert "expectation_violated:turn_within" in final.close_reason
    await hub.close()


@pytest.mark.asyncio
async def test_workflow_unknown_expected_speaker_still_auto_closes() -> None:
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        clock=clock,
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))
    graph = TransitionGraph(initial_speaker=alice.agent_id, default_target=AgentTarget("nobody"))
    channel = await alice.open(
        type=WORKFLOW_TYPE,
        target=[bob.agent_id],
        knobs={"graph": graph.to_dict()},
    )
    bob.on_envelope(_silent_handler)
    await hub.post_envelope(
        Envelope(
            channel_id=channel.channel_id,
            sender_id=alice.agent_id,
            audience=None,
            event_type=EV_TEXT,
            event_data={"text": "handoff to an unknown agent"},
        )
    )
    assert getattr(hub._adapter_states[channel.channel_id], "expected_next_speaker") == "nobody"

    clock.advance(601)
    await hub._expectation_tick()

    final = await hub.get_channel(channel.channel_id)
    assert final.state == ChannelState.CLOSED
    assert "expectation_violated:turn_within" in final.close_reason
    await hub.close()


@pytest.mark.asyncio
async def test_workflow_turn_warn_and_auto_close_thresholds_deduplicate() -> None:
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        clock=clock,
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )
    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))
    graph = TransitionGraph.round_robin([alice.agent_id, bob.agent_id])
    channel = await alice.open(
        type=WORKFLOW_TYPE,
        target=[bob.agent_id],
        knobs={"graph": graph.to_dict()},
    )
    bob_capture = _CaptureHandler()
    bob.on_envelope(bob_capture)
    await hub.post_envelope(
        Envelope(
            channel_id=channel.channel_id,
            sender_id=alice.agent_id,
            audience=None,
            event_type=EV_TEXT,
            event_data={"text": "waiting for bob"},
        )
    )

    clock.advance(121)
    await hub._expectation_tick()
    await asyncio.sleep(0.05)
    await hub._expectation_tick()
    warn_wal = await hub.read_wal(channel.channel_id)
    warn_events = [e for e in warn_wal if e.event_type == EV_EXPECTATION_VIOLATED]
    assert len(warn_events) == 1
    detail = warn_events[0].event_data.get("detail")
    assert isinstance(detail, dict)
    assert detail.get("threshold_seconds") == 120
    assert hub._active_channels[channel.channel_id].state == ChannelState.ACTIVE

    clock.advance(480)
    await hub._expectation_tick()

    final = await hub.get_channel(channel.channel_id)
    assert final.state == ChannelState.CLOSED
    assert "expectation_violated:turn_within" in final.close_reason
    final_wal = await hub.read_wal(channel.channel_id)
    assert sum(e.event_type == EV_EXPECTATION_VIOLATED for e in final_wal) == 1
    assert sum(e.event_type == EV_EXPECTATION_VIOLATED for e in bob_capture.received) == 1
    await hub.close()


@pytest.mark.asyncio
async def test_expectation_tick_uses_consistent_state_and_wal_snapshot() -> None:
    """A concurrent reply cannot pair fresh WAL with stale turn state."""

    class _BlockingWalReadStore(MemoryKnowledgeStore):
        def __init__(self) -> None:
            super().__init__()
            self.block_next_wal_read = False
            self.read_entered = asyncio.Event()
            self.allow_read = asyncio.Event()

        async def read(self, path: str) -> str | None:
            if self.block_next_wal_read and path.endswith("/wal.jsonl"):
                self.block_next_wal_read = False
                self.read_entered.set()
                await self.allow_read.wait()
            return await super().read(path)

    class _CapturingTurnEvaluator:
        name = "turn_within"

        def __init__(self) -> None:
            self.contexts: list[ExpectationContext] = []

        def evaluate(self, expectation: Expectation, context: ExpectationContext) -> Violation | None:
            self.contexts.append(context)
            return TurnWithinEvaluator().evaluate(expectation, context)

    clock = _MockClock("2026-01-01T00:00:00+00:00")
    store = _BlockingWalReadStore()
    hub = await Hub.open(store, clock=clock, ttl_sweep_interval=0, expectation_sweep_interval=0)
    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))
    channel = await alice.open(
        type=WORKFLOW_TYPE,
        target=[bob.agent_id],
        knobs={"graph": TransitionGraph.round_robin([alice.agent_id, bob.agent_id]).to_dict()},
    )
    bob.on_envelope(_silent_handler)
    await hub.post_envelope(
        Envelope(
            channel_id=channel.channel_id,
            sender_id=alice.agent_id,
            audience=None,
            event_type=EV_TEXT,
            event_data={"text": "bob's turn"},
        )
    )
    clock.advance(601)

    capturing_evaluator = _CapturingTurnEvaluator()
    hub.register_expectation_evaluator(capturing_evaluator)
    store.block_next_wal_read = True
    tick = asyncio.create_task(hub._expectation_tick())
    await store.read_entered.wait()
    post = asyncio.create_task(
        hub.post_envelope(
            Envelope(
                channel_id=channel.channel_id,
                sender_id=bob.agent_id,
                audience=None,
                event_type=EV_TEXT,
                event_data={"text": "bob replied"},
            )
        )
    )
    await asyncio.sleep(0)
    store.allow_read.set()
    await asyncio.gather(tick, post, return_exceptions=True)

    assert len(capturing_evaluator.contexts) == 2
    for context in capturing_evaluator.contexts:
        text_envelopes = [envelope for envelope in context.wal if envelope.event_type == EV_TEXT]
        assert text_envelopes
        latest_speaker = text_envelopes[-1].sender_id
        expected_next_speaker = alice.agent_id if latest_speaker == bob.agent_id else bob.agent_id
        assert getattr(context.state, "expected_next_speaker") == expected_next_speaker

    await hub.close()


@pytest.mark.asyncio
async def test_turn_within_deduplicates_per_turn_not_per_participant() -> None:
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    hub = await Hub.open(MemoryKnowledgeStore(), clock=clock, ttl_sweep_interval=0, expectation_sweep_interval=0)
    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))
    channel = await alice.open(type=DISCUSSION_TYPE, target=[bob.agent_id])

    async def post_text(sender_id: str, text: str) -> None:
        await hub.post_envelope(
            Envelope(
                channel_id=channel.channel_id,
                sender_id=sender_id,
                audience=None,
                event_type=EV_TEXT,
                event_data={"text": text},
            )
        )

    await post_text(alice.agent_id, "first turn")
    clock.advance(121)
    await hub._expectation_tick()
    wal = await hub.read_wal(channel.channel_id)
    violations = [envelope for envelope in wal if envelope.event_type == EV_EXPECTATION_VIOLATED]
    assert len(violations) == 1
    assert violations[0].audience == [bob.agent_id]
    await hub._expectation_tick()
    wal = await hub.read_wal(channel.channel_id)
    assert sum(envelope.event_type == EV_EXPECTATION_VIOLATED for envelope in wal) == 1

    await post_text(bob.agent_id, "second turn")
    await post_text(alice.agent_id, "third turn")
    clock.advance(121)
    await hub._expectation_tick()

    wal = await hub.read_wal(channel.channel_id)
    violations = [envelope for envelope in wal if envelope.event_type == EV_EXPECTATION_VIOLATED]
    assert len(violations) == 2
    for envelope in violations:
        assert envelope.event_data["violators"] == [bob.agent_id]
        detail = envelope.event_data.get("detail")
        assert isinstance(detail, dict)
        assert detail.get("threshold_seconds") == 120

    await hub.close()


@pytest.mark.asyncio
async def test_turn_within_warns_for_initial_speaker_on_fresh_channels() -> None:
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    hub = await Hub.open(MemoryKnowledgeStore(), clock=clock, ttl_sweep_interval=0, expectation_sweep_interval=0)
    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))

    discussion = await alice.open(type=DISCUSSION_TYPE, target=[bob.agent_id])
    workflow = await alice.open(
        type=WORKFLOW_TYPE,
        target=[bob.agent_id],
        knobs={"graph": TransitionGraph.round_robin([alice.agent_id, bob.agent_id]).to_dict()},
    )
    assert not any(envelope.event_type == EV_TEXT for envelope in await hub.read_wal(discussion.channel_id))
    assert not any(envelope.event_type == EV_TEXT for envelope in await hub.read_wal(workflow.channel_id))
    discussion_expected = hub._adapters[(DISCUSSION_TYPE, 1)].expected_next(
        hub._channels[discussion.channel_id], hub._adapter_states[discussion.channel_id]
    )
    workflow_expected = hub._adapters[(WORKFLOW_TYPE, 1)].expected_next(
        hub._channels[workflow.channel_id], hub._adapter_states[workflow.channel_id]
    )
    assert discussion_expected is not None
    assert workflow_expected is not None
    assert discussion_expected.triggering_envelope_id is None
    assert workflow_expected.triggering_envelope_id is None

    clock.advance(121)
    await hub._expectation_tick()

    for channel_id in (discussion.channel_id, workflow.channel_id):
        wal = await hub.read_wal(channel_id)
        violations = [envelope for envelope in wal if envelope.event_type == EV_EXPECTATION_VIOLATED]
        assert len(violations) == 1
        assert violations[0].event_data["expectation"] == "turn_within"
        assert violations[0].event_data["violators"] == [alice.agent_id]
        detail = violations[0].event_data.get("detail")
        assert isinstance(detail, dict)
        assert detail.get("threshold_seconds") == 120

    await hub.close()


@pytest.mark.asyncio
async def test_discussion_adapter_turn_warn_targets_current_expected_speaker() -> None:
    clock = _MockClock("2026-01-01T00:00:00+00:00")
    hub = await Hub.open(MemoryKnowledgeStore(), clock=clock, ttl_sweep_interval=0, expectation_sweep_interval=0)
    alice = await hub.register(_agent("alice"))
    bob = await hub.register(_agent("bob"))
    channel = await alice.open(type=DISCUSSION_TYPE, target=[bob.agent_id])

    bob_capture = _CaptureHandler()
    alice_capture = _CaptureHandler()
    bob.on_envelope(bob_capture)
    alice.on_envelope(alice_capture)
    await hub.post_envelope(
        Envelope(
            channel_id=channel.channel_id,
            sender_id=alice.agent_id,
            audience=None,
            event_type=EV_TEXT,
            event_data={"text": "waiting for bob"},
        )
    )
    assert hub._adapter_states[channel.channel_id].expected_next_speaker == bob.agent_id

    clock.advance(121)
    await hub._expectation_tick()
    await asyncio.sleep(0.05)

    wal = await hub.read_wal(channel.channel_id)
    violations = [envelope for envelope in wal if envelope.event_type == EV_EXPECTATION_VIOLATED]
    assert len(violations) == 1
    assert violations[0].audience == [bob.agent_id]
    assert violations[0].event_data["violators"] == [bob.agent_id]
    assert any(envelope.envelope_id == violations[0].envelope_id for envelope in bob_capture.received)
    assert not any(envelope.event_type == EV_EXPECTATION_VIOLATED for envelope in alice_capture.received)

    await hub.close()
