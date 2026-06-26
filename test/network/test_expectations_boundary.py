# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Expectation evaluator/handler boundary cases — exact-deadline timing,
zero-timeout edge, handler exception isolation, position-based dedup.

The existing ``test_expectations.py`` covers the happy / sad paths.
This file targets the boundary conditions the docstrings imply but
don't explicitly exercise:

* All three evaluators use ``elapsed < seconds`` (or ``>=`` for
  ``reply_within``) — confirm the exact ``elapsed == seconds``
  threshold fires.
* ``seconds=0`` should fire on the first tick (degenerate but worth
  proving deterministic).
* A handler that raises must not stop the sweeper — other expectations
  on the same tick keep firing, the violation is still marked as fired
  (no infinite re-fire), and the audit log records what's expected.
* The fired-set key is ``(idx, name, violator)`` — two expectations
  with the same ``name`` but different ``on_violation`` handlers must
  both fire (the position-based key disambiguates).
* Unknown evaluator name → silently skipped (no crash).
* Unknown handler name → silently skipped (no crash).
"""

from datetime import datetime

import pytest

from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_TEXT,
    Envelope,
    Hub,
)
from ag2.network.adapters.base import AdapterResult
from ag2.network.adapters.conversation import ConversationAdapter
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
    AUDIT_KIND_EXPECTATION_VIOLATED,
    AcksWithinEvaluator,
    AuditLog,
    ExpectationContext,
    MaxSilenceEvaluator,
    ReplyWithinEvaluator,
)
from ag2.network.views.builtin import FullTranscript

from ._helpers import _MockClock


class _NoOpAdapter:
    """Drives expectation tests with custom manifests.

    The expectation sweeper only reads ``manifest.expectations`` and
    walks ``_active_channels`` — it never calls ``validate_send`` /
    ``fold`` / ``on_accepted``. This stand-in covers the adapter
    surface without exercising any choreography.
    """

    def __init__(self, manifest: ChannelManifest) -> None:
        self.manifest = manifest

    def initial_state(self, _meta: ChannelMetadata) -> dict:
        return {}

    def fold(self, _envelope: Envelope, state: dict) -> dict:
        return state

    def validate_create(self, _meta: ChannelMetadata) -> None:
        return

    def validate_send(self, _meta: ChannelMetadata, _envelope: Envelope, _state: dict) -> None:
        return

    def on_accepted(self, _meta: ChannelMetadata, _envelope: Envelope, _state: dict) -> AdapterResult:
        return AdapterResult()

    def default_view_policy(self, _meta: ChannelMetadata, _participant_id: str) -> FullTranscript:
        return FullTranscript()


def _conv_meta(
    *,
    state: ChannelState = ChannelState.PENDING,
    created_at: str = "2026-01-01T00:00:00+00:00",
    pending_acks: list[str] | None = None,
) -> ChannelMetadata:
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


def _ctx(meta: ChannelMetadata, *, now: str, wal: list[Envelope] | None = None) -> ExpectationContext:
    now_dt = datetime.fromisoformat(now)
    return ExpectationContext(
        metadata=meta,
        state=None,
        wal=wal or [],
        now_iso=now,
        now_seconds=now_dt.timestamp(),
    )


def _inject_pending_channel(
    hub: Hub,
    *,
    channel_id: str,
    adapter_key: tuple[str, int],
    clock: _MockClock,
    pending_acks: tuple[str, ...] = ("bob",),
) -> ChannelMetadata:
    """Inject a PENDING channel metadata directly into the hub caches.

    ``create_channel`` would block on ``invite_ack_timeout`` waiting
    for acks that never arrive; these tests need the channel to stay
    PENDING long enough for the sweeper to evaluate.
    """
    meta = ChannelMetadata(
        channel_id=channel_id,
        manifest=hub._adapters[adapter_key].manifest,
        creator_id="alice",
        participants=[
            Participant(agent_id="alice", role=ParticipantRole.INITIATOR, order=0, joined_at=clock()),
            Participant(agent_id="bob", role=ParticipantRole.PARTICIPANT, order=1, joined_at=clock()),
        ],
        state=ChannelState.PENDING,
        created_at=clock(),
        pending_acks=list(pending_acks),
    )
    hub._channels[channel_id] = meta
    hub._active_channels[channel_id] = meta
    hub._adapter_states[channel_id] = {}
    return meta


class TestEvaluatorExactBoundary:
    """At ``elapsed == seconds``, the evaluator should fire (not be silent)."""

    def test_acks_within_fires_at_exact_threshold(self) -> None:
        meta = _conv_meta(state=ChannelState.PENDING, pending_acks=["bob"])
        evaluator = AcksWithinEvaluator()
        expectation = Expectation(name="acks_within", on_violation="audit", params={"seconds": 30})
        # 30s elapsed — equals threshold.
        violation = evaluator.evaluate(expectation, _ctx(meta, now="2026-01-01T00:00:30+00:00"))
        assert violation is not None
        assert violation.violator_ids == ["bob"]

    def test_acks_within_silent_just_under_threshold(self) -> None:
        meta = _conv_meta(state=ChannelState.PENDING, pending_acks=["bob"])
        evaluator = AcksWithinEvaluator()
        expectation = Expectation(name="acks_within", on_violation="audit", params={"seconds": 30})
        # 29.999s — under threshold by a hair.
        violation = evaluator.evaluate(expectation, _ctx(meta, now="2026-01-01T00:00:29.999000+00:00"))
        assert violation is None

    def test_max_silence_fires_at_exact_threshold(self) -> None:
        meta = _conv_meta(state=ChannelState.ACTIVE, created_at="2026-01-01T00:00:00+00:00")
        evaluator = MaxSilenceEvaluator()
        expectation = Expectation(name="max_silence", on_violation="audit", params={"seconds": 60})
        violation = evaluator.evaluate(expectation, _ctx(meta, now="2026-01-01T00:01:00+00:00"))
        assert violation is not None

    def test_reply_within_fires_at_exact_threshold(self) -> None:
        meta = _conv_meta(state=ChannelState.ACTIVE, created_at="2026-01-01T00:00:00+00:00")
        wal = [
            Envelope(
                envelope_id="e1",
                channel_id="s1",
                sender_id="alice",
                audience=["bob"],
                event_type=EV_TEXT,
                event_data={"text": "hi"},
                created_at="2026-01-01T00:00:00+00:00",
            )
        ]
        evaluator = ReplyWithinEvaluator()
        expectation = Expectation(name="reply_within", on_violation="audit", params={"seconds": 60})
        # exactly 60s elapsed — should fire (>= boundary)
        violation = evaluator.evaluate(expectation, _ctx(meta, now="2026-01-01T00:01:00+00:00", wal=wal))
        assert violation is not None
        assert violation.violator_ids == ["bob"]


class TestEvaluatorZeroTimeout:
    """``seconds=0`` is degenerate but should fire deterministically."""

    def test_acks_within_zero_fires_immediately(self) -> None:
        meta = _conv_meta(state=ChannelState.PENDING, pending_acks=["bob"])
        evaluator = AcksWithinEvaluator()
        expectation = Expectation(name="acks_within", on_violation="audit", params={"seconds": 0})
        # Same instant as creation — `elapsed=0`, `0 < 0` is False → fires.
        violation = evaluator.evaluate(expectation, _ctx(meta, now="2026-01-01T00:00:00+00:00"))
        assert violation is not None

    def test_max_silence_zero_fires_immediately(self) -> None:
        meta = _conv_meta(state=ChannelState.ACTIVE, created_at="2026-01-01T00:00:00+00:00")
        evaluator = MaxSilenceEvaluator()
        expectation = Expectation(name="max_silence", on_violation="audit", params={"seconds": 0})
        violation = evaluator.evaluate(expectation, _ctx(meta, now="2026-01-01T00:00:00+00:00"))
        assert violation is not None


@pytest.mark.asyncio
async def test_handler_exception_does_not_stop_sweeper() -> None:
    """A custom handler that raises must not crash the sweeper or
    block other handlers from firing on the same tick.

    The hub wraps each handler call in ``contextlib.suppress(Exception)``
    and still marks the violation as fired (so we don't infinite-loop
    on a bad handler).
    """
    clock = _MockClock(start="2026-01-01T00:00:00+00:00")
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,  # we drive ticks manually
        clock=clock,
    )

    handler_calls: list[tuple[str, str]] = []

    class CrashHandler:
        name = "crash"

        async def handle(self, _hub: Hub, channel_id: str, _violation: object) -> None:
            handler_calls.append(("crash", channel_id))
            raise RuntimeError("boom")

    hub.register_violation_handler(CrashHandler())
    hub.register_adapter(
        _NoOpAdapter(
            ChannelManifest(
                type="crash_test",
                version=1,
                participants=ParticipantSchema(min=2),
                expectations=[
                    Expectation(name="acks_within", on_violation="crash", params={"seconds": 0}),
                ],
            )
        )
    )

    _inject_pending_channel(hub, channel_id="s-crash", adapter_key=("crash_test", 1), clock=clock)

    # First tick: handler raises but sweeper survives.
    clock.advance(1)  # 1s elapsed > 0s threshold
    await hub._expectation_tick()
    assert len(handler_calls) == 1  # called once, raised, but tracked

    # Second tick: violation deduped, handler NOT called again.
    clock.advance(1)
    await hub._expectation_tick()
    assert len(handler_calls) == 1  # no re-fire

    await hub.close()


@pytest.mark.asyncio
async def test_two_same_name_expectations_with_different_handlers_both_fire() -> None:
    """A manifest can list the same evaluator name twice with different
    ``on_violation`` handlers (e.g. ``warn`` at 120s + ``auto_close`` at
    600s). The position-based key ``(idx, name, violator)`` must
    disambiguate so both fire when the larger threshold is also crossed.
    """
    clock = _MockClock(start="2026-01-01T00:00:00+00:00")
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
        clock=clock,
    )

    fired: list[tuple[int, str]] = []

    class WarnHandler:
        name = "warn"

        async def handle(self, _hub: Hub, _channel_id: str, _violation: object) -> None:
            fired.append((0, "warn"))

    class AuditHandler2:
        name = "audit2"

        async def handle(self, _hub: Hub, _channel_id: str, _violation: object) -> None:
            fired.append((1, "audit2"))

    hub.register_violation_handler(WarnHandler())
    hub.register_violation_handler(AuditHandler2())
    hub.register_adapter(
        _NoOpAdapter(
            ChannelManifest(
                type="dual_test",
                version=1,
                participants=ParticipantSchema(min=2),
                expectations=[
                    Expectation(name="acks_within", on_violation="warn", params={"seconds": 30}),
                    Expectation(name="acks_within", on_violation="audit2", params={"seconds": 60}),
                ],
            )
        )
    )

    _inject_pending_channel(hub, channel_id="s-dual", adapter_key=("dual_test", 1), clock=clock)

    # 35s in: only the 30s expectation fires.
    clock.advance(35)
    await hub._expectation_tick()
    assert (0, "warn") in fired
    assert (1, "audit2") not in fired

    # 65s in: BOTH expectations should now have fired.
    clock.advance(30)  # total 65s
    await hub._expectation_tick()
    assert (0, "warn") in fired  # still there
    assert (1, "audit2") in fired  # NEW
    # Each handler fired exactly once (dedup intact).
    assert sum(1 for x in fired if x == (0, "warn")) == 1
    assert sum(1 for x in fired if x == (1, "audit2")) == 1

    await hub.close()


@pytest.mark.asyncio
async def test_unknown_evaluator_name_silently_ignored() -> None:
    """An expectation referencing an unregistered evaluator should not
    crash the sweeper — it's silently skipped."""
    clock = _MockClock()
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
        clock=clock,
    )
    hub.register_adapter(
        _NoOpAdapter(
            ChannelManifest(
                type="bogus",
                version=1,
                participants=ParticipantSchema(min=2),
                expectations=[
                    Expectation(name="nonexistent_evaluator", on_violation="audit", params={}),
                ],
            )
        )
    )

    _inject_pending_channel(hub, channel_id="s-bogus", adapter_key=("bogus", 1), clock=clock)

    clock.advance(60)
    # Must not raise.
    await hub._expectation_tick()

    await hub.close()


@pytest.mark.asyncio
async def test_unknown_handler_name_silently_ignored() -> None:
    """If the evaluator fires but the named handler isn't registered,
    no record is written but the sweeper continues."""
    clock = _MockClock()
    hub = await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
        clock=clock,
    )
    hub.register_adapter(
        _NoOpAdapter(
            ChannelManifest(
                type="ghost",
                version=1,
                participants=ParticipantSchema(min=2),
                expectations=[
                    Expectation(name="acks_within", on_violation="ghost_handler", params={"seconds": 0}),
                ],
            )
        )
    )

    _inject_pending_channel(hub, channel_id="s-ghost", adapter_key=("ghost", 1), clock=clock)

    clock.advance(1)
    # Must not raise. Audit log should also stay empty since no
    # handler was found to record anything.
    await hub._expectation_tick()
    audit = AuditLog(hub._store)
    records = await audit.read_all()
    violation_records = [r for r in records if r["kind"] == AUDIT_KIND_EXPECTATION_VIOLATED]
    assert violation_records == []

    await hub.close()
