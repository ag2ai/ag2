# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Wire frames for the ``Link`` Protocol.

The vocabulary covers what ``LocalLink`` needs plus ``HelloFrame`` /
``WelcomeFrame`` so the same shape works over a network transport
without renaming.

``encode_frame`` / ``decode_frame`` produce JSON-compatible dicts.
``LocalLink`` passes Frame dataclasses through in-memory queues
without serialisation; the encode/decode helpers exist so the same
frame vocabulary serialises losslessly over the wire.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, TypeAlias

from ..envelope import Envelope

__all__ = (
    "AcceptFrame",
    "ChunkFrame",
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
    feed the registered ``AuthAdapter`` (defaults to ``NoAuth``).
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
    """client → hub: post an envelope into a channel.

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
    """hub → client: deliver an envelope to a specific participant.

    ``recipient_id`` is the agent id this delivery is for — the hub
    already iterates per-recipient when dispatching, so stamping the
    target on the frame lets the ``HubClient`` demux directly without
    re-walking the channel participants. Required so broadcasts
    (``audience=None``) route correctly when one connection hosts
    multiple identities.
    """

    kind: ClassVar[str] = "notify"
    envelope: Envelope
    recipient_id: str = ""


@dataclass(slots=True)
class ChunkFrame:
    """hub → client: a streaming token chunk during an agent's turn.

    Transient — never WAL-persisted. Delivered while the sending
    agent's ``agent.ask`` is in progress so observers can display tokens
    in real-time before the final ``NotifyFrame`` carrying the complete
    envelope arrives.

    ``parent_envelope_id`` is the WAL envelope that triggered the turn
    (the question the agent is answering). ``recipient_id`` mirrors
    ``NotifyFrame``'s per-recipient stamp so ``HubClient`` can demux
    without re-walking channel participants.
    """

    kind: ClassVar[str] = "chunk"
    channel_id: str
    sender_id: str
    content: str
    parent_envelope_id: str
    recipient_id: str = ""


@dataclass(slots=True)
class ReceiptFrame:
    """client → hub: ack or nack a ``notify``.

    ``status`` is ``"ack"`` (advances the cross-process ``inbox.cursor``
    when the transport supports replay) or ``"nack"`` (records to
    ``inbox_nacks.jsonl``). ``reason`` is a free-form diagnostic for
    the audit log.
    """

    kind: ClassVar[str] = "receipt"
    envelope_id: str
    status: str  # "ack" | "nack"
    reason: str = ""


@dataclass(slots=True)
class SubscribeFrame:
    """client → hub: open a push subscription on a channel or task.

    At least one of ``channel_id`` / ``task_id`` must be set.
    ``since_envelope_id`` is the cursor for at-least-once replay over
    a reconnecting transport; in-process delivery is exactly-once by
    per-channel lock.
    """

    kind: ClassVar[str] = "subscribe"
    subscription_id: str
    channel_id: str | None = None
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
    | ChunkFrame
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
    "chunk": ChunkFrame,
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
