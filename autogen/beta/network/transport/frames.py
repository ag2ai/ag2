# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Wire frames for the Link protocol.

The Link frame vocabulary is declared in full (so later phases add zero new
frame types) even though Phase 1 exercises only a subset. Frames are plain
dataclasses with a ``type`` tag; :func:`encode_frame` / :func:`decode_frame`
handle (de)serialization for remote transports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Union

from ..envelope import Envelope

# ---------------------------------------------------------------------------
# Frame dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class HelloFrame:
    """Client → hub handshake. Carries identity + auth claim."""

    identity: dict[str, Any]
    rule: dict[str, Any]
    auth_claim: dict[str, Any] = field(default_factory=dict)
    resume_actor_id: str | None = None
    last_seq: int | None = None

    type: ClassVar[str] = "hello"


@dataclass(slots=True)
class WelcomeFrame:
    """Hub → client response to ``hello``. Stamps the actor id."""

    actor_id: str
    hub_id: str

    type: ClassVar[str] = "welcome"


@dataclass(slots=True)
class NotifyFrame:
    """Hub → client: here is a new envelope for you."""

    envelope: Envelope

    type: ClassVar[str] = "notify"


@dataclass(slots=True)
class ReceiptFrame:
    """Client → hub: I durably accepted (or rejected) the envelope."""

    envelope_id: str
    status: str = "ack"  # ack | nack
    reason: str | None = None

    type: ClassVar[str] = "receipt"


@dataclass(slots=True)
class SendFrame:
    """Client → hub: post an envelope into a session."""

    envelope: Envelope
    idempotency_key: str | None = None

    type: ClassVar[str] = "send"


@dataclass(slots=True)
class AcceptFrame:
    """Hub → client: envelope accepted.

    Carries the hub-stamped ``envelope_id`` and ``wal_offset`` — the byte
    position in the session WAL immediately after this envelope. A client
    that wants to wait for a correlated reply (``Session.ask``) uses this
    offset as the ``since`` cursor on its subscription so the hub does not
    re-replay the whole WAL.
    """

    envelope_id: str
    wal_offset: int = 0
    request_id: str | None = None

    type: ClassVar[str] = "accept"


@dataclass(slots=True)
class ErrorFrame:
    """Hub → client: something went wrong. Carries a stable error code."""

    code: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    type: ClassVar[str] = "error"


@dataclass(slots=True)
class SubscribeFrame:
    """Client → hub: open a push subscription to a session or task."""

    subscription_id: str
    session_id: str | None = None
    task_id: str | None = None
    since: int | None = None
    causation_id: str | None = None

    type: ClassVar[str] = "subscribe"


@dataclass(slots=True)
class UnsubscribeFrame:
    subscription_id: str

    type: ClassVar[str] = "unsubscribe"


@dataclass(slots=True)
class EventFrame:
    """Hub → client: subscription delivery."""

    subscription_id: str
    envelope: Envelope

    type: ClassVar[str] = "event"


@dataclass(slots=True)
class ChunkFrame:
    """Hub → client: streaming content token (Phase 2+)."""

    envelope_id: str
    chunk_index: int
    content: str
    final: bool = False

    type: ClassVar[str] = "chunk"


@dataclass(slots=True)
class RuleChangedFrame:
    """Hub → client: push updated transforms for this actor's rule."""

    actor_id: str
    transforms: list[dict[str, Any]]
    version: int

    type: ClassVar[str] = "rule_changed"


@dataclass(slots=True)
class PingFrame:
    timestamp_ms: int

    type: ClassVar[str] = "ping"


@dataclass(slots=True)
class PongFrame:
    timestamp_ms: int

    type: ClassVar[str] = "pong"


Frame = Union[  # noqa: UP007
    HelloFrame,
    WelcomeFrame,
    NotifyFrame,
    ReceiptFrame,
    SendFrame,
    AcceptFrame,
    ErrorFrame,
    SubscribeFrame,
    UnsubscribeFrame,
    EventFrame,
    ChunkFrame,
    RuleChangedFrame,
    PingFrame,
    PongFrame,
]


_FRAME_TYPES: dict[str, type] = {
    "hello": HelloFrame,
    "welcome": WelcomeFrame,
    "notify": NotifyFrame,
    "receipt": ReceiptFrame,
    "send": SendFrame,
    "accept": AcceptFrame,
    "error": ErrorFrame,
    "subscribe": SubscribeFrame,
    "unsubscribe": UnsubscribeFrame,
    "event": EventFrame,
    "chunk": ChunkFrame,
    "rule_changed": RuleChangedFrame,
    "ping": PingFrame,
    "pong": PongFrame,
}


# ---------------------------------------------------------------------------
# (de)serialization
# ---------------------------------------------------------------------------


def encode_frame(frame: Frame) -> str:
    """Encode a frame to a JSON string for wire transport."""

    return json.dumps(_frame_to_dict(frame), sort_keys=True)


def decode_frame(payload: str | dict[str, Any]) -> Frame:
    """Decode a JSON string or already-parsed dict into a Frame."""

    data = json.loads(payload) if isinstance(payload, str) else payload
    frame_type = data.get("type")
    cls = _FRAME_TYPES.get(frame_type)
    if cls is None:
        from ..errors import FrameError

        raise FrameError(f"unknown frame type {frame_type!r}")
    return _dict_to_frame(cls, data)


def _frame_to_dict(frame: Frame) -> dict[str, Any]:
    if isinstance(frame, (NotifyFrame, SendFrame)):
        data: dict[str, Any] = {
            "type": frame.type,
            "envelope": frame.envelope.to_dict(),
        }
        if isinstance(frame, SendFrame) and frame.idempotency_key is not None:
            data["idempotency_key"] = frame.idempotency_key
        return data
    if isinstance(frame, EventFrame):
        return {
            "type": frame.type,
            "subscription_id": frame.subscription_id,
            "envelope": frame.envelope.to_dict(),
        }
    if isinstance(frame, HelloFrame):
        return {
            "type": frame.type,
            "identity": frame.identity,
            "rule": frame.rule,
            "auth_claim": frame.auth_claim,
            "resume_actor_id": frame.resume_actor_id,
            "last_seq": frame.last_seq,
        }
    if isinstance(frame, WelcomeFrame):
        return {"type": frame.type, "actor_id": frame.actor_id, "hub_id": frame.hub_id}
    if isinstance(frame, ReceiptFrame):
        return {
            "type": frame.type,
            "envelope_id": frame.envelope_id,
            "status": frame.status,
            "reason": frame.reason,
        }
    if isinstance(frame, AcceptFrame):
        return {
            "type": frame.type,
            "envelope_id": frame.envelope_id,
            "wal_offset": frame.wal_offset,
            "request_id": frame.request_id,
        }
    if isinstance(frame, ErrorFrame):
        return {
            "type": frame.type,
            "code": frame.code,
            "message": frame.message,
            "request_id": frame.request_id,
            "details": frame.details,
        }
    if isinstance(frame, SubscribeFrame):
        return {
            "type": frame.type,
            "subscription_id": frame.subscription_id,
            "session_id": frame.session_id,
            "task_id": frame.task_id,
            "since": frame.since,
            "causation_id": frame.causation_id,
        }
    if isinstance(frame, UnsubscribeFrame):
        return {"type": frame.type, "subscription_id": frame.subscription_id}
    if isinstance(frame, ChunkFrame):
        return {
            "type": frame.type,
            "envelope_id": frame.envelope_id,
            "chunk_index": frame.chunk_index,
            "content": frame.content,
            "final": frame.final,
        }
    if isinstance(frame, RuleChangedFrame):
        return {
            "type": frame.type,
            "actor_id": frame.actor_id,
            "transforms": frame.transforms,
            "version": frame.version,
        }
    if isinstance(frame, (PingFrame, PongFrame)):
        return {"type": frame.type, "timestamp_ms": frame.timestamp_ms}
    raise TypeError(f"cannot encode frame of type {type(frame)!r}")


def _dict_to_frame(cls: type, data: dict[str, Any]) -> Frame:
    if cls is NotifyFrame:
        return NotifyFrame(envelope=Envelope.from_dict(data["envelope"]))
    if cls is SendFrame:
        return SendFrame(
            envelope=Envelope.from_dict(data["envelope"]),
            idempotency_key=data.get("idempotency_key"),
        )
    if cls is EventFrame:
        return EventFrame(
            subscription_id=data["subscription_id"],
            envelope=Envelope.from_dict(data["envelope"]),
        )
    if cls is HelloFrame:
        return HelloFrame(
            identity=data.get("identity", {}),
            rule=data.get("rule", {}),
            auth_claim=data.get("auth_claim", {}),
            resume_actor_id=data.get("resume_actor_id"),
            last_seq=data.get("last_seq"),
        )
    if cls is WelcomeFrame:
        return WelcomeFrame(actor_id=data["actor_id"], hub_id=data["hub_id"])
    if cls is ReceiptFrame:
        return ReceiptFrame(
            envelope_id=data["envelope_id"],
            status=data.get("status", "ack"),
            reason=data.get("reason"),
        )
    if cls is AcceptFrame:
        return AcceptFrame(
            envelope_id=data["envelope_id"],
            wal_offset=int(data.get("wal_offset", 0)),
            request_id=data.get("request_id"),
        )
    if cls is ErrorFrame:
        return ErrorFrame(
            code=data["code"],
            message=data["message"],
            request_id=data.get("request_id"),
            details=dict(data.get("details", {})),
        )
    if cls is SubscribeFrame:
        return SubscribeFrame(
            subscription_id=data["subscription_id"],
            session_id=data.get("session_id"),
            task_id=data.get("task_id"),
            since=data.get("since"),
            causation_id=data.get("causation_id"),
        )
    if cls is UnsubscribeFrame:
        return UnsubscribeFrame(subscription_id=data["subscription_id"])
    if cls is ChunkFrame:
        return ChunkFrame(
            envelope_id=data["envelope_id"],
            chunk_index=int(data["chunk_index"]),
            content=data["content"],
            final=bool(data.get("final", False)),
        )
    if cls is RuleChangedFrame:
        return RuleChangedFrame(
            actor_id=data["actor_id"],
            transforms=list(data.get("transforms", [])),
            version=int(data.get("version", 1)),
        )
    if cls is PingFrame:
        return PingFrame(timestamp_ms=int(data["timestamp_ms"]))
    if cls is PongFrame:
        return PongFrame(timestamp_ms=int(data["timestamp_ms"]))
    raise TypeError(f"cannot decode frame class {cls!r}")
