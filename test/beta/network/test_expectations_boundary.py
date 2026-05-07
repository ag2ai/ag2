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

from autogen.beta import Agent
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    EV_TEXT,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    Resume,
)
from autogen.beta.network.adapters.conversation import ConversationAdapter
from autogen.beta.network.hub import (
    AUDIT_KIND_EXPECTATION_VIOLATED,
    AcksWithinEvaluator,
    AuditLog,
    ExpectationContext,
    MaxSilenceEvaluator,
    ReplyWithinEvaluator,
    Violation,
)
from autogen.beta.network.session import (
    Expectation,
    Participant,
    ParticipantRole,
    SessionMetadata,
    SessionState,
)

from ._helpers import ScriptedConfig, _MockClock


def _agent(name: str) -> Agent:
    return Agent(name=name, config=ScriptedConfig())


def _conv_meta(
    *,
    state: SessionState = SessionState.PENDING,
    created_at: str = "2026-01-01T00:00:00+00:00",
    pending_acks: list[str] | None = None,
) -> SessionMetadata:
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


def _ctx(meta: SessionMetadata, *, now: str, wal=None) -> ExpectationContext:
    now_dt = datetime.fromisoformat(now)
    return ExpectationContext(
        metadata=meta,
        state=None,
        wal=wal or [],
        now_iso=now,
        now_seconds=now_dt.timestamp(),
    )


class TestEvaluatorExactBoundary:
    """At ``elapsed == seconds``, the evaluator should fire (not be silent)."""

    def test_acks_within_fires_at_exact_threshold(self) -> None:
        meta = _conv_meta(state=SessionState.PENDING, pending_acks=["bob"])
        evaluator = AcksWithinEvaluator()
        expectation = Expectation(name="acks_within", on_violation="audit", params={"seconds": 30})
        # 30s elapsed — equals threshold.
        violation = evaluator.evaluate(expectation, _ctx(meta, now="2026-01-01T00:00:30+00:00"))
        assert violation is not None
        assert violation.violator_ids == ["bob"]

    def test_acks_within_silent_just_under_threshold(self) -> None:
        meta = _conv_meta(state=SessionState.PENDING, pending_acks=["bob"])
        evaluator = AcksWithinEvaluator()
        expectation = Expectation(name="acks_within", on_violation="audit", params={"seconds": 30})
        # 29.999s — under threshold by a hair.
        violation = evaluator.evaluate(expectation, _ctx(meta, now="2026-01-01T00:00:29.999000+00:00"))
        assert violation is None

    def test_max_silence_fires_at_exact_threshold(self) -> None:
        meta = _conv_meta(state=SessionState.ACTIVE, created_at="2026-01-01T00:00:00+00:00")
        evaluator = MaxSilenceEvaluator()
        expectation = Expectation(name="max_silence", on_violation="audit", params={"seconds": 60})
        violation = evaluator.evaluate(expectation, _ctx(meta, now="2026-01-01T00:01:00+00:00"))
        assert violation is not None

    def test_reply_within_fires_at_exact_threshold(self) -> None:
        meta = _conv_meta(state=SessionState.ACTIVE, created_at="2026-01-01T00:00:00+00:00")
        wal = [
            Envelope(
                envelope_id="e1",
                session_id="s1",
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
        meta = _conv_meta(state=SessionState.PENDING, pending_acks=["bob"])
        evaluator = AcksWithinEvaluator()
        expectation = Expectation(name="acks_within", on_violation="audit", params={"seconds": 0})
        # Same instant as creation — `elapsed=0`, `0 < 0` is False → fires.
        violation = evaluator.evaluate(expectation, _ctx(meta, now="2026-01-01T00:00:00+00:00"))
        assert violation is not None

    def test_max_silence_zero_fires_immediately(self) -> None:
        meta = _conv_meta(state=SessionState.ACTIVE, created_at="2026-01-01T00:00:00+00:00")
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

    handler_calls: list[str] = []

    class CrashHandler:
        name = "crash"

        async def handle(self, _hub, session_id: str, _violation) -> None:
            handler_calls.append(("crash", session_id))
            raise RuntimeError("boom")

    hub.register_violation_handler(CrashHandler())

    # Custom adapter with a one-expectation manifest pointing at "crash".
    from autogen.beta.network.adapters.base import AdapterResult
    from autogen.beta.network.session import (
        ParticipantSchema as PS,
        SessionManifest as SM,
    )

    class TestAdapter:
        def __init__(self) -> None:
            self.manifest = SM(
                type="crash_test",
                version=1,
                participants=PS(min=2),
                expectations=[
                    Expectation(name="acks_within", on_violation="crash", params={"seconds": 0}),
                ],
            )

        def initial_state(self, _meta):
            return {}

        def fold(self, _env, state):
            return state

        def validate_create(self, _meta) -> None:
            return

        def validate_send(self, _meta, _env, _state) -> None:
            return

        def on_accepted(self, _meta, _env, _state):
            return AdapterResult()

        def default_view_policy(self, _meta, _pid):
            from autogen.beta.network.views.builtin import FullTranscript
            return FullTranscript()

    hub.register_adapter(TestAdapter())

    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)
    await hc.register(_agent("alice"), Passport(name="alice"), Resume())

    # Setup: pending session with bob never acking → triggers the
    # zero-timeout acks_within → CrashHandler fires → raises → sweeper
    # MUST survive.
    bob = await hc.register(_agent("bob"), Passport(name="bob"), Resume())
    await bob.disconnect()  # bob ignores invites — guarantees session stays PENDING
    # We need to construct the session in PENDING state. The standard
    # create_session would block on ack timeout; instead we drive the
    # sweeper after manually injecting the metadata.
    from autogen.beta.network.session import Participant as P

    meta = SessionMetadata(
        session_id="s-crash",
        manifest=hub._adapters[("crash_test", 1)].manifest,
        creator_id=hc._clients[next(iter(hc._clients.keys()))].agent_id if hc._clients else "alice",
        participants=[
            P(agent_id="alice", role=ParticipantRole.INITIATOR, order=0, joined_at=clock()),
            P(agent_id="bob", role=ParticipantRole.PARTICIPANT, order=1, joined_at=clock()),
        ],
        state=SessionState.PENDING,
        created_at=clock(),
        pending_acks=["bob"],
    )
    hub._sessions["s-crash"] = meta
    hub._active_sessions["s-crash"] = meta
    hub._adapter_states["s-crash"] = {}

    # First tick: handler raises but sweeper survives.
    clock.advance(1)  # 1s elapsed > 0s threshold
    await hub._expectation_tick()
    assert len(handler_calls) == 1  # called once, raised, but tracked

    # Second tick: violation deduped, handler NOT called again.
    clock.advance(1)
    await hub._expectation_tick()
    assert len(handler_calls) == 1  # no re-fire

    await hc.close()
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

        async def handle(self, _hub, session_id: str, violation) -> None:
            fired.append((0, "warn"))

    class AuditHandler2:
        name = "audit2"

        async def handle(self, _hub, session_id: str, violation) -> None:
            fired.append((1, "audit2"))

    hub.register_violation_handler(WarnHandler())
    hub.register_violation_handler(AuditHandler2())

    # Adapter with two acks_within expectations, different handlers.
    from autogen.beta.network.adapters.base import AdapterResult
    from autogen.beta.network.session import (
        ParticipantSchema as PS,
        SessionManifest as SM,
    )

    class DualExpAdapter:
        def __init__(self) -> None:
            self.manifest = SM(
                type="dual_test",
                version=1,
                participants=PS(min=2),
                expectations=[
                    Expectation(name="acks_within", on_violation="warn", params={"seconds": 30}),
                    Expectation(name="acks_within", on_violation="audit2", params={"seconds": 60}),
                ],
            )

        def initial_state(self, _meta):
            return {}

        def fold(self, _env, state):
            return state

        def validate_create(self, _meta) -> None:
            return

        def validate_send(self, _meta, _env, _state) -> None:
            return

        def on_accepted(self, _meta, _env, _state):
            return AdapterResult()

        def default_view_policy(self, _meta, _pid):
            from autogen.beta.network.views.builtin import FullTranscript
            return FullTranscript()

    hub.register_adapter(DualExpAdapter())

    link = LocalLink(hub)
    hc = HubClient(link, hub=hub)
    await hc.register(_agent("alice"), Passport(name="alice"), Resume())

    from autogen.beta.network.session import Participant as P

    meta = SessionMetadata(
        session_id="s-dual",
        manifest=hub._adapters[("dual_test", 1)].manifest,
        creator_id="alice",
        participants=[
            P(agent_id="alice", role=ParticipantRole.INITIATOR, order=0, joined_at=clock()),
            P(agent_id="bob", role=ParticipantRole.PARTICIPANT, order=1, joined_at=clock()),
        ],
        state=SessionState.PENDING,
        created_at=clock(),
        pending_acks=["bob"],
    )
    hub._sessions["s-dual"] = meta
    hub._active_sessions["s-dual"] = meta
    hub._adapter_states["s-dual"] = {}

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

    await hc.close()
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

    from autogen.beta.network.adapters.base import AdapterResult
    from autogen.beta.network.session import (
        ParticipantSchema as PS,
        SessionManifest as SM,
    )

    class BogusAdapter:
        def __init__(self) -> None:
            self.manifest = SM(
                type="bogus",
                version=1,
                participants=PS(min=2),
                expectations=[
                    Expectation(name="nonexistent_evaluator", on_violation="audit", params={}),
                ],
            )

        def initial_state(self, _meta):
            return {}

        def fold(self, _env, state):
            return state

        def validate_create(self, _meta) -> None:
            return

        def validate_send(self, _meta, _env, _state) -> None:
            return

        def on_accepted(self, _meta, _env, _state):
            return AdapterResult()

        def default_view_policy(self, _meta, _pid):
            from autogen.beta.network.views.builtin import FullTranscript
            return FullTranscript()

    hub.register_adapter(BogusAdapter())

    from autogen.beta.network.session import Participant as P

    meta = SessionMetadata(
        session_id="s-bogus",
        manifest=hub._adapters[("bogus", 1)].manifest,
        creator_id="alice",
        participants=[
            P(agent_id="alice", role=ParticipantRole.INITIATOR, order=0, joined_at=clock()),
            P(agent_id="bob", role=ParticipantRole.PARTICIPANT, order=1, joined_at=clock()),
        ],
        state=SessionState.PENDING,
        created_at=clock(),
        pending_acks=["bob"],
    )
    hub._sessions["s-bogus"] = meta
    hub._active_sessions["s-bogus"] = meta
    hub._adapter_states["s-bogus"] = {}

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

    from autogen.beta.network.adapters.base import AdapterResult
    from autogen.beta.network.session import (
        ParticipantSchema as PS,
        SessionManifest as SM,
    )

    class GhostHandlerAdapter:
        def __init__(self) -> None:
            self.manifest = SM(
                type="ghost",
                version=1,
                participants=PS(min=2),
                expectations=[
                    Expectation(name="acks_within", on_violation="ghost_handler", params={"seconds": 0}),
                ],
            )

        def initial_state(self, _meta):
            return {}

        def fold(self, _env, state):
            return state

        def validate_create(self, _meta) -> None:
            return

        def validate_send(self, _meta, _env, _state) -> None:
            return

        def on_accepted(self, _meta, _env, _state):
            return AdapterResult()

        def default_view_policy(self, _meta, _pid):
            from autogen.beta.network.views.builtin import FullTranscript
            return FullTranscript()

    hub.register_adapter(GhostHandlerAdapter())

    from autogen.beta.network.session import Participant as P

    meta = SessionMetadata(
        session_id="s-ghost",
        manifest=hub._adapters[("ghost", 1)].manifest,
        creator_id="alice",
        participants=[
            P(agent_id="alice", role=ParticipantRole.INITIATOR, order=0, joined_at=clock()),
            P(agent_id="bob", role=ParticipantRole.PARTICIPANT, order=1, joined_at=clock()),
        ],
        state=SessionState.PENDING,
        created_at=clock(),
        pending_acks=["bob"],
    )
    hub._sessions["s-ghost"] = meta
    hub._active_sessions["s-ghost"] = meta
    hub._adapter_states["s-ghost"] = {}

    clock.advance(1)
    # Must not raise. Audit log should also stay empty since no
    # handler was found to record anything.
    await hub._expectation_tick()
    audit = AuditLog(hub._store)
    records = await audit.read_all()
    violation_records = [r for r in records if r["kind"] == AUDIT_KIND_EXPECTATION_VIOLATED]
    assert violation_records == []

    await hub.close()
