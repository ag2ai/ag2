# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Built-in expectation evaluators + violation handlers.

Expectations are protocol-shape contracts declared on
``ChannelManifest.expectations``. The hub evaluates them on a periodic
sweeper tick — independently of the per-envelope ``validate_send`` /
``on_accepted`` path. ``validate_send`` rejects bad **sends**;
expectations react to bad **silence** (or bad pacing).

Evaluators
----------
* ``acks_within(seconds)`` — invitee hasn't ack'd or rejected within T
  after ``EV_CHANNEL_INVITE``. Fires only while the channel is PENDING.
* ``reply_within(seconds)`` — a participant with envelopes addressed
  to them hasn't sent a response within T.
* ``turn_within(seconds)`` — the expected next speaker hasn't acted within T.
* ``max_silence(seconds)`` — channel has had no content envelopes from
  anyone for T.

Handlers
--------
* ``audit`` — record an entry in ``audit.jsonl``; no envelope sent.
* ``notify_channel`` — audit + post ``EV_EXPECTATION_VIOLATED`` to
  every participant.
* ``warn`` — post ``EV_EXPECTATION_VIOLATED`` to the violating participant(s).
* ``auto_close`` — audit + transition the channel to ``CLOSED`` with
  ``close_reason="expectation_violated:{name}"``.

All handlers are passive: the hub records, signals, or closes; it
never re-tries, substitutes content, or makes outcome decisions for
the agent.
"""

import contextlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from ..adapters.base import ExpectedTurn
from ..channel import ChannelMetadata, ChannelState, Expectation
from ..envelope import EV_CHANNEL_OPENED, EV_EXPECTATION_VIOLATED, EV_TEXT, Envelope

if TYPE_CHECKING:
    from .core import Hub

logger = logging.getLogger(__name__)

__all__ = (
    "AcksWithinEvaluator",
    "AuditHandler",
    "AutoCloseHandler",
    "ExpectationContext",
    "ExpectationEvaluator",
    "MaxSilenceEvaluator",
    "NotifyChannelHandler",
    "ReplyWithinEvaluator",
    "TurnWithinEvaluator",
    "Violation",
    "ViolationHandler",
    "WarnHandler",
    "default_evaluators",
    "default_handlers",
)


@dataclass(slots=True)
class Violation:
    """Result of an evaluator firing.

    ``violator_ids`` is the list of participants the violation applies
    to; an empty list represents a channel-wide violation (e.g.
    ``max_silence`` — nobody specifically is silent, the channel is).
    ``turn_key`` names the turn a turn-scoped violation belongs to (the
    envelope that handed over the turn), so the same participant can fire
    again on a later turn. ``None`` for a channel's opening turn and for
    evaluators that are not turn-scoped.
    """

    expectation: Expectation
    violator_ids: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    turn_key: str | None = None


@dataclass(slots=True)
class ExpectationContext:
    """Inputs handed to every evaluator on each sweeper tick."""

    metadata: ChannelMetadata
    state: object  # opaque AdapterState
    wal: list[Envelope]
    now_iso: str
    now_seconds: float
    expected_turn: ExpectedTurn | None = None


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
    channels. They must be tolerant of duplicate calls — the sweeper
    deduplicates per (channel, expectation, violator, turn) before invoking,
    but transient re-fires across hub restarts are possible since the
    fired-violation cache is in-memory only.
    """

    name: str

    async def handle(
        self,
        hub: "Hub",
        channel_id: str,
        violation: Violation,
    ) -> None: ...


# ── Evaluators ──────────────────────────────────────────────────────────────


def _parse_iso_seconds(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _is_content_event(event_type: str) -> bool:
    """Substantive (non-protocol, non-task) envelope."""
    if event_type.startswith("ag2.channel."):
        return False
    if event_type.startswith("ag2.task."):
        return False
    return event_type != EV_EXPECTATION_VIOLATED


class AcksWithinEvaluator:
    """Fire when invitees haven't acked within ``params.seconds``.

    Only meaningful while the channel is in ``PENDING`` state. Returns
    one violation listing every still-pending invitee.
    """

    name = "acks_within"

    def evaluate(
        self,
        expectation: Expectation,
        context: ExpectationContext,
    ) -> Violation | None:
        if context.metadata.state != ChannelState.PENDING:
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
        if context.metadata.state != ChannelState.ACTIVE:
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
    """Fire when the channel has had no content envelope for T.

    Channel-wide — ``violator_ids`` is empty. Anchors on the latest
    content envelope's timestamp, falling back to channel creation
    when no content has been posted.
    """

    name = "max_silence"

    def evaluate(
        self,
        expectation: Expectation,
        context: ExpectationContext,
    ) -> Violation | None:
        if context.metadata.state != ChannelState.ACTIVE:
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


class TurnWithinEvaluator:
    """Fire when the adapter's expected next speaker misses their turn window.

    The window starts at the envelope that handed over the turn
    (``ExpectedTurn.triggering_envelope_id``), or at ``EV_CHANNEL_OPENED``
    for a channel's opening turn, so time spent waiting for invite acks
    doesn't count against the first speaker. Fires once per turn.
    """

    name = "turn_within"

    def evaluate(
        self,
        expectation: Expectation,
        context: ExpectationContext,
    ) -> Violation | None:
        if context.metadata.state != ChannelState.ACTIVE:
            return None
        expected = context.expected_turn
        if expected is None:
            return None
        anchor_iso = _turn_started_at(context, expected)
        if anchor_iso is None:
            return None
        seconds = float(expectation.params.get("seconds", 600))
        elapsed = context.now_seconds - _parse_iso_seconds(anchor_iso)
        if elapsed < seconds:
            return None
        return Violation(
            expectation=expectation,
            violator_ids=[expected.agent_id],
            detail={
                "elapsed_seconds": elapsed,
                "threshold_seconds": seconds,
            },
            turn_key=expected.triggering_envelope_id,
        )


def _turn_started_at(context: ExpectationContext, expected: ExpectedTurn) -> str | None:
    """``created_at`` of the envelope that started ``expected``'s turn.

    ``None`` if that envelope isn't in the WAL. The hub folds state only
    after appending to the append-only WAL, so this means the inputs are
    inconsistent and there's nothing sound to measure from. The turn then
    never times out, so it is logged rather than skipped silently.
    """
    if expected.triggering_envelope_id is None:
        for env in context.wal:
            if env.event_type == EV_CHANNEL_OPENED:
                return env.created_at
        return context.metadata.created_at
    for env in context.wal:
        if env.envelope_id == expected.triggering_envelope_id:
            return env.created_at
    logger.warning(
        "turn_within on channel %s cannot find triggering envelope %s in the WAL",
        context.metadata.channel_id,
        expected.triggering_envelope_id,
    )
    return None


# ── Handlers ────────────────────────────────────────────────────────────────


class AuditHandler:
    """No-op handler — audit emission is owned by the ``AuditLog`` listener.

    Kept so manifests authored against the V1 vocabulary
    (``on_violation: "audit"``) still resolve. The hub fans out
    ``on_expectation_fired`` to every listener (including the built-in
    ``AuditLog``) before invoking the per-violation handler, so this
    handler has nothing to do.
    """

    name = "audit"

    async def handle(
        self,
        hub: "Hub",
        channel_id: str,
        violation: Violation,
    ) -> None:
        return None


class NotifyChannelHandler:
    """Broadcast ``EV_EXPECTATION_VIOLATED`` to every channel participant.

    Audit emission is owned by the ``AuditLog`` listener.
    """

    name = "notify_channel"

    async def handle(
        self,
        hub: "Hub",
        channel_id: str,
        violation: Violation,
    ) -> None:
        await _post_violation(hub, channel_id, violation, to_violators=False)


class WarnHandler:
    """Post ``EV_EXPECTATION_VIOLATED`` only to the violating participant(s).

    Violators outside the channel (e.g. a workflow handed off to an agent
    that isn't a participant) are never addressed; if no violator is a
    participant, or the violation is channel-wide, it is broadcast to the
    channel, as with ``notify_channel``. Audit emission is owned by the
    ``AuditLog`` listener.
    """

    name = "warn"

    async def handle(
        self,
        hub: "Hub",
        channel_id: str,
        violation: Violation,
    ) -> None:
        await _post_violation(hub, channel_id, violation, to_violators=True)


async def _post_violation(
    hub: "Hub",
    channel_id: str,
    violation: Violation,
    *,
    to_violators: bool,
) -> None:
    metadata = hub._channels.get(channel_id)
    if metadata is None or metadata.is_terminal():
        return
    audience: list[str] | None = None
    if to_violators:
        participant_ids = set(metadata.participant_ids())
        audience = [vid for vid in violation.violator_ids if vid in participant_ids] or None
    envelope = Envelope(
        channel_id=channel_id,
        sender_id=metadata.creator_id,
        audience=audience,
        event_type=EV_EXPECTATION_VIOLATED,
        event_data={
            "expectation": violation.expectation.name,
            "violators": list(violation.violator_ids),
            "detail": dict(violation.detail),
        },
    )
    # Posting violations is best-effort — a closed/closing
    # channel shouldn't crash the sweeper.
    with contextlib.suppress(Exception):
        await hub.post_envelope(envelope)


class AutoCloseHandler:
    """Transition the channel to ``CLOSED``.

    Audit emission is owned by the ``AuditLog`` listener.
    """

    name = "auto_close"

    async def handle(
        self,
        hub: "Hub",
        channel_id: str,
        violation: Violation,
    ) -> None:
        metadata = hub._channels.get(channel_id)
        if metadata is None or metadata.is_terminal():
            return
        with contextlib.suppress(Exception):
            await hub.close_channel(
                channel_id,
                reason=f"expectation_violated:{violation.expectation.name}",
            )


# ── Factories ───────────────────────────────────────────────────────────────


def default_evaluators() -> list[ExpectationEvaluator]:
    return [AcksWithinEvaluator(), ReplyWithinEvaluator(), TurnWithinEvaluator(), MaxSilenceEvaluator()]


def default_handlers() -> list[ViolationHandler]:
    return [AuditHandler(), NotifyChannelHandler(), WarnHandler(), AutoCloseHandler()]
