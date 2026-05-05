# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Built-in expectation evaluators + violation handlers.

Expectations are protocol-shape contracts declared on
``SessionManifest.expectations``. The hub evaluates them on a periodic
sweeper tick — independently of the per-envelope ``validate_send`` /
``on_accepted`` path. ``validate_send`` rejects bad **sends**;
expectations react to bad **silence** (or bad pacing).

Evaluators
----------
* ``acks_within(seconds)`` — invitee hasn't ack'd or rejected within T
  after ``EV_SESSION_INVITE``. Fires only while the session is PENDING.
* ``reply_within(seconds)`` — a participant with envelopes addressed
  to them hasn't sent a response within T.
* ``max_silence(seconds)`` — session has had no content envelopes from
  anyone for T.

Handlers
--------
* ``audit`` — record an entry in ``audit.jsonl``; no envelope sent.
* ``notify_session`` — audit + post ``EV_EXPECTATION_VIOLATED`` to
  every participant.
* ``auto_close`` — audit + transition the session to ``CLOSED`` with
  ``close_reason="expectation_violated:{name}"``.

All handlers are passive: the hub records, signals, or closes; it
never re-tries, substitutes content, or makes outcome decisions for
the agent.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from ..envelope import EV_EXPECTATION_VIOLATED, EV_TEXT, Envelope
from ..session import Expectation, SessionMetadata, SessionState
from .audit import AUDIT_KIND_EXPECTATION_VIOLATED

if TYPE_CHECKING:
    from .core import Hub

__all__ = (
    "AcksWithinEvaluator",
    "AutoCloseHandler",
    "AuditHandler",
    "ExpectationContext",
    "ExpectationEvaluator",
    "MaxSilenceEvaluator",
    "NotifySessionHandler",
    "ReplyWithinEvaluator",
    "Violation",
    "ViolationHandler",
    "default_evaluators",
    "default_handlers",
)


@dataclass(slots=True)
class Violation:
    """Result of an evaluator firing.

    ``violator_ids`` is the list of participants the violation applies
    to; an empty list represents a session-wide violation (e.g.
    ``max_silence`` — nobody specifically is silent, the session is).
    """

    expectation: Expectation
    violator_ids: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExpectationContext:
    """Inputs handed to every evaluator on each sweeper tick."""

    metadata: SessionMetadata
    state: object  # opaque AdapterState
    wal: list[Envelope]
    now_iso: str
    now_seconds: float


class ExpectationEvaluator(Protocol):
    """Pure predicate over ``(metadata, state, wal, clock)``.

    Returns ``None`` for "no violation right now." Evaluators must be
    deterministic functions of their inputs — the sweeper is called on
    a periodic tick, so non-deterministic evaluators would flap.
    """

    name: str

    def evaluate(
        self,
        expectation: Expectation,
        context: ExpectationContext,
    ) -> Violation | None: ...


class ViolationHandler(Protocol):
    """What the hub does when an evaluator fires.

    Handlers are async because they may post envelopes or transition
    sessions. They must be tolerant of duplicate calls — the sweeper
    deduplicates per (session, expectation, violator) before invoking,
    but transient re-fires across hub restarts are possible since the
    fired-violation cache is in-memory only.
    """

    name: str

    async def handle(
        self,
        hub: "Hub",
        session_id: str,
        violation: Violation,
    ) -> None: ...


# ── Evaluators ──────────────────────────────────────────────────────────────


def _parse_iso_seconds(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _is_content_event(event_type: str) -> bool:
    """Substantive (non-protocol, non-task) envelope."""
    if event_type.startswith("ag2.session."):
        return False
    if event_type.startswith("ag2.task."):
        return False
    if event_type == EV_EXPECTATION_VIOLATED:
        return False
    return True


class AcksWithinEvaluator:
    """Fire when invitees haven't acked within ``params.seconds``.

    Only meaningful while the session is in ``PENDING`` state. Returns
    one violation listing every still-pending invitee.
    """

    name = "acks_within"

    def evaluate(
        self,
        expectation: Expectation,
        context: ExpectationContext,
    ) -> Violation | None:
        if context.metadata.state != SessionState.PENDING:
            return None
        seconds = float(expectation.params.get("seconds", 30))
        elapsed = context.now_seconds - _parse_iso_seconds(context.metadata.created_at)
        if elapsed < seconds:
            return None
        if not context.metadata.pending_acks:
            return None
        return Violation(
            expectation=expectation,
            violator_ids=list(context.metadata.pending_acks),
            detail={
                "elapsed_seconds": elapsed,
                "threshold_seconds": seconds,
            },
        )


class ReplyWithinEvaluator:
    """Fire when a participant addressed by a send hasn't replied in T.

    For each participant, finds the most recent ``EV_TEXT`` envelope
    addressed to them (broadcast or via ``audience``). If they haven't
    sent any ``EV_TEXT`` after that timestamp, and that timestamp is
    older than ``params.seconds``, they're a violator.
    """

    name = "reply_within"

    def evaluate(
        self,
        expectation: Expectation,
        context: ExpectationContext,
    ) -> Violation | None:
        if context.metadata.state != SessionState.ACTIVE:
            return None
        seconds = float(expectation.params.get("seconds", 600))

        # Index: latest EV_TEXT addressed to each participant + each
        # participant's latest sent EV_TEXT.
        latest_in: dict[str, Envelope] = {}
        latest_out: dict[str, Envelope] = {}
        for env in context.wal:
            if env.event_type != EV_TEXT:
                continue
            sender = env.sender_id
            latest_out[sender] = env
            for p in context.metadata.participants:
                pid = p.agent_id
                if pid == sender:
                    continue
                if env.audience is not None and pid not in env.audience:
                    continue
                latest_in[pid] = env

        violators: list[str] = []
        for p in context.metadata.participants:
            pid = p.agent_id
            inbound = latest_in.get(pid)
            if inbound is None:
                continue
            outbound = latest_out.get(pid)
            inbound_at = _parse_iso_seconds(inbound.created_at)
            if outbound is not None:
                outbound_at = _parse_iso_seconds(outbound.created_at)
                if outbound_at >= inbound_at:
                    continue
            if context.now_seconds - inbound_at >= seconds:
                violators.append(pid)

        if not violators:
            return None
        return Violation(
            expectation=expectation,
            violator_ids=violators,
            detail={"threshold_seconds": seconds},
        )


class MaxSilenceEvaluator:
    """Fire when the session has had no content envelope for T.

    Session-wide — ``violator_ids`` is empty. Anchors on the latest
    content envelope's timestamp, falling back to session creation
    when no content has been posted.
    """

    name = "max_silence"

    def evaluate(
        self,
        expectation: Expectation,
        context: ExpectationContext,
    ) -> Violation | None:
        if context.metadata.state != SessionState.ACTIVE:
            return None
        seconds = float(expectation.params.get("seconds", 3600))
        anchor_iso = context.metadata.created_at
        for env in reversed(context.wal):
            if _is_content_event(env.event_type):
                anchor_iso = env.created_at
                break
        elapsed = context.now_seconds - _parse_iso_seconds(anchor_iso)
        if elapsed < seconds:
            return None
        return Violation(
            expectation=expectation,
            violator_ids=[],
            detail={
                "elapsed_seconds": elapsed,
                "threshold_seconds": seconds,
            },
        )


# ── Handlers ────────────────────────────────────────────────────────────────


async def _audit_violation(
    hub: "Hub",
    session_id: str,
    violation: Violation,
) -> None:
    await hub._audit_log.append({
        "at": hub._clock(),
        "kind": AUDIT_KIND_EXPECTATION_VIOLATED,
        "session_id": session_id,
        "expectation": violation.expectation.name,
        "on_violation": violation.expectation.on_violation,
        "params": dict(violation.expectation.params),
        "violators": list(violation.violator_ids),
        "detail": dict(violation.detail),
    })


class AuditHandler:
    """Record the violation; no envelope, no state change."""

    name = "audit"

    async def handle(
        self,
        hub: "Hub",
        session_id: str,
        violation: Violation,
    ) -> None:
        await _audit_violation(hub, session_id, violation)


class NotifySessionHandler:
    """Audit + broadcast ``EV_EXPECTATION_VIOLATED`` to every participant."""

    name = "notify_session"

    async def handle(
        self,
        hub: "Hub",
        session_id: str,
        violation: Violation,
    ) -> None:
        await _audit_violation(hub, session_id, violation)
        metadata = hub._sessions.get(session_id)
        if metadata is None or metadata.is_terminal():
            return
        envelope = Envelope(
            session_id=session_id,
            sender_id=metadata.creator_id,
            audience=None,
            event_type=EV_EXPECTATION_VIOLATED,
            event_data={
                "expectation": violation.expectation.name,
                "violators": list(violation.violator_ids),
                "detail": dict(violation.detail),
            },
        )
        try:
            await hub.post_envelope(envelope)
        except Exception:
            # Posting violations is best-effort — a closed/closing
            # session shouldn't crash the sweeper.
            pass


class AutoCloseHandler:
    """Audit + transition the session to ``CLOSED``."""

    name = "auto_close"

    async def handle(
        self,
        hub: "Hub",
        session_id: str,
        violation: Violation,
    ) -> None:
        await _audit_violation(hub, session_id, violation)
        metadata = hub._sessions.get(session_id)
        if metadata is None or metadata.is_terminal():
            return
        try:
            await hub.close_session(
                session_id,
                reason=f"expectation_violated:{violation.expectation.name}",
            )
        except Exception:
            pass


# ── Factories ───────────────────────────────────────────────────────────────


def default_evaluators() -> list[ExpectationEvaluator]:
    return [AcksWithinEvaluator(), ReplyWithinEvaluator(), MaxSilenceEvaluator()]


def default_handlers() -> list[ViolationHandler]:
    return [AuditHandler(), NotifySessionHandler(), AutoCloseHandler()]
