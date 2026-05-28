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
    "ErrorFrame",
    "Frame",
    "HelloFrame",
    "NotifyFrame",
    "PingFrame",
    "PongFrame",
    "ReceiptFrame",
    "SendFrame",
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

    ``since_envelope_id`` is the client's high-water mark — the last
    envelope_id it remembers acknowledging. When set, the hub replays
    every envelope addressed to this name with ``envelope_id`` greater
    than ``since_envelope_id`` as fresh ``NotifyFrame`` deliveries
    before the connection sees any new traffic. ``None`` (the default)
    skips replay.
    """

    kind: ClassVar[str] = "hello"
    name: str
    auth_scheme: str = "none"
    auth_claim: dict[str, Any] = field(default_factory=dict)
    since_envelope_id: str | None = None


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
class ReceiptFrame:
    """client → hub: ack or nack a ``notify``.

    ``recipient_id`` names the agent acknowledging delivery — required
    because a single endpoint may host several registered identities,
    and the hub must know whose cursor to advance. Mirrors
    :attr:`NotifyFrame.recipient_id`. ``channel_id`` names the channel
    the acked envelope belongs to; the hub keeps one cursor per
    (recipient, channel), so an ack that omits it cannot be attributed
    and is dropped.

    ``status`` is ``"ack"`` (the agent has processed the envelope; the
    hub advances that recipient's cursor for the channel so it is not
    replayed on reconnect) or ``"nack"`` (the agent could not process
    it; the hub leaves the cursor untouched and surfaces a
    dispatch-failure event to listeners). ``reason`` is a free-form
    diagnostic.
    """

    kind: ClassVar[str] = "receipt"
    envelope_id: str
    status: str  # "ack" | "nack"
    recipient_id: str = ""
    channel_id: str = ""
    reason: str = ""


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
