# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Wire frames for the ``Link`` Protocol.

V1 ships the subset of frames required by ``LocalLink`` plus
``HelloFrame`` / ``WelcomeFrame`` so the same wire vocabulary works
for ``WsLink`` (Phase 3) without renaming. Streaming ``ChunkFrame``
and Phase-3 ``RuleChangedFrame`` are intentionally omitted.

``encode_frame`` / ``decode_frame`` produce JSON-compatible dicts.
``LocalLink`` passes Frame dataclasses through in-memory queues
without serialisation; the encode/decode helpers exist so the same
frame vocabulary serialises losslessly on ``WsLink``.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, TypeAlias

from ..envelope import Envelope

__all__ = (
    "AcceptFrame",
    "ErrorFrame",
    "EventFrame",
    "Frame",
    "HelloFrame",
    "NotifyFrame",
    "PingFrame",
    "PongFrame",
    "ReceiptFrame",
    "SendFrame",
    "SubscribeFrame",
    "UnsubscribeFrame",
    "WelcomeFrame",
    "decode_frame",
    "encode_frame",
)


@dataclass(slots=True)
class HelloFrame:
    """client → hub: open the connection and authenticate.

    ``name`` lets the hub bind the connection to an existing identity
    (re-connect) or onboard a new one. ``auth_scheme`` + ``auth_claim``
    feed the registered ``AuthAdapter``. V1 ships ``NoAuth`` only.
    """

    kind: ClassVar[str] = "hello"
    name: str
    auth_scheme: str = "none"
    auth_claim: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WelcomeFrame:
    """hub → client: handshake accepted; carries hub clock + connection id."""

    kind: ClassVar[str] = "welcome"
    endpoint_id: str
    hub_time: str  # ISO-Z


@dataclass(slots=True)
class PingFrame:
    """Heartbeat — both directions. ``LocalLink`` skips wire pings."""

    kind: ClassVar[str] = "ping"


@dataclass(slots=True)
class PongFrame:
    """Heartbeat reply — both directions."""

    kind: ClassVar[str] = "pong"


@dataclass(slots=True)
class SendFrame:
    """client → hub: post an envelope into a session.

    Hub stamps ``envelope_id`` and ``created_at`` at accept and replies
    with ``AcceptFrame`` (or ``ErrorFrame`` on rejection).
    """

    kind: ClassVar[str] = "send"
    envelope: Envelope


@dataclass(slots=True)
class AcceptFrame:
    """hub → client: ack of a ``send`` with the hub-stamped ``envelope_id``."""

    kind: ClassVar[str] = "accept"
    envelope_id: str


@dataclass(slots=True)
class ErrorFrame:
    """hub → client: structured rejection.

    ``code`` is a stable identifier (``"protocol_error"``,
    ``"access_denied"``, ``"not_found"``, ``"inbox_full"``, …).
    ``envelope_id`` is set when the error relates to a specific send.
    """

    kind: ClassVar[str] = "error"
    code: str
    message: str
    envelope_id: str | None = None


@dataclass(slots=True)
class NotifyFrame:
    """hub → client: deliver an envelope to a participant.

    The receiving ``AgentClient`` routes to the registered notify
    handler for the envelope's session.
    """

    kind: ClassVar[str] = "notify"
    envelope: Envelope


@dataclass(slots=True)
class ReceiptFrame:
    """client → hub: ack or nack a ``notify``.

    ``status`` is ``"ack"`` (advances ``inbox.cursor`` in Phase 3) or
    ``"nack"`` (records to ``inbox_nacks.jsonl``). ``reason`` is a
    free-form diagnostic for the audit log.
    """

    kind: ClassVar[str] = "receipt"
    envelope_id: str
    status: str  # "ack" | "nack"
    reason: str = ""


@dataclass(slots=True)
class SubscribeFrame:
    """client → hub: open a push subscription on a session or task.

    At least one of ``session_id`` / ``task_id`` must be set.
    ``since_envelope_id`` is the cursor for at-least-once replay
    (Phase 3 — V1 in-process is exactly-once by lock).
    """

    kind: ClassVar[str] = "subscribe"
    subscription_id: str
    session_id: str | None = None
    task_id: str | None = None
    event_types: list[str] | None = None
    since_envelope_id: str | None = None


@dataclass(slots=True)
class UnsubscribeFrame:
    """client → hub: close a subscription."""

    kind: ClassVar[str] = "unsubscribe"
    subscription_id: str


@dataclass(slots=True)
class EventFrame:
    """hub → client: subscription delivery."""

    kind: ClassVar[str] = "event"
    subscription_id: str
    envelope: Envelope


Frame: TypeAlias = (
    HelloFrame
    | WelcomeFrame
    | PingFrame
    | PongFrame
    | SendFrame
    | AcceptFrame
    | ErrorFrame
    | NotifyFrame
    | ReceiptFrame
    | SubscribeFrame
    | UnsubscribeFrame
    | EventFrame
)


_FRAME_CLASSES: dict[str, type] = {
    "hello": HelloFrame,
    "welcome": WelcomeFrame,
    "ping": PingFrame,
    "pong": PongFrame,
    "send": SendFrame,
    "accept": AcceptFrame,
    "error": ErrorFrame,
    "notify": NotifyFrame,
    "receipt": ReceiptFrame,
    "subscribe": SubscribeFrame,
    "unsubscribe": UnsubscribeFrame,
    "event": EventFrame,
}


def encode_frame(frame: Frame) -> dict[str, Any]:
    """Serialise a Frame to a JSON-compatible dict.

    Adds the ``kind`` discriminator (a ``ClassVar``, so ``asdict`` does
    not include it). Nested ``Envelope`` is auto-flattened by
    ``dataclasses.asdict``.
    """
    data = asdict(frame)
    data["kind"] = frame.kind
    return data


def decode_frame(data: dict[str, Any]) -> Frame:
    """Reconstruct a Frame from a JSON dict.

    Looks up the dataclass via ``kind``, rehydrates a nested
    ``Envelope`` if present, and constructs the frame. Raises
    ``ValueError`` on unknown ``kind``.
    """
    payload = dict(data)  # shallow copy — caller's dict is preserved
    kind = payload.pop("kind", None)
    if kind not in _FRAME_CLASSES:
        raise ValueError(f"unknown frame kind: {kind!r}")
    cls = _FRAME_CLASSES[kind]
    if "envelope" in payload and isinstance(payload["envelope"], dict):
        payload["envelope"] = Envelope.from_dict(payload["envelope"])
    return cls(**payload)
