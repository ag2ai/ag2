# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Wire frames for the ``Link`` Protocol.

Three families share one vocabulary:

* Handshake — ``HelloFrame`` / ``WelcomeFrame`` open and authenticate
  a connection (and trigger replay on reconnect).
* Control plane — ``RequestFrame`` / ``ResponseFrame`` carry a
  request/response RPC correlated by ``request_id``. Every hub control
  operation (register, discovery, channel lifecycle, posting an
  envelope, task ops) crosses the wire through this pair.
* Data plane — ``NotifyFrame`` (hub → client delivery) and
  ``ReceiptFrame`` (client → hub ack/nack) are the async push path.

``PingFrame`` / ``PongFrame`` stay defined as a heartbeat vocabulary;
``LocalLink`` skips wire pings and ``WsLink`` delegates heartbeat to
the WebSocket library, but the frames remain available to any binding
that wants application-level pings.

``encode_frame`` / ``decode_frame`` produce JSON-compatible dicts.
``LocalLink`` passes Frame dataclasses through in-memory queues
without serialisation; the encode/decode helpers exist so the same
frame vocabulary serialises losslessly over the wire.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, TypeAlias

from ..envelope import Envelope

__all__ = (
    "ErrorFrame",
    "Frame",
    "HelloFrame",
    "NotifyFrame",
    "PingFrame",
    "PongFrame",
    "ReceiptFrame",
    "RequestFrame",
    "ResponseFrame",
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
class RequestFrame:
    """client → hub: invoke a control-plane operation.

    ``op`` names the operation (``"register"``, ``"create_channel"``,
    ``"post_envelope"``, ``"get_agent"``, …); ``params`` is a
    JSON-compatible argument dict whose shape is defined per ``op``.
    ``request_id`` is a client-minted correlation id the hub echoes on
    the matching :class:`ResponseFrame` so concurrent in-flight
    requests on one connection demux unambiguously.

    Posting an envelope is just another op (``"post_envelope"`` with
    ``params={"envelope": <envelope dict>}``); the hub stamps
    ``envelope_id`` / ``created_at`` and returns the id in the
    response. There is no separate send frame — the request/response
    pair carries it with correlation for free.
    """

    kind: ClassVar[str] = "request"
    request_id: str
    op: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResponseFrame:
    """hub → client: result of a :class:`RequestFrame`.

    ``request_id`` echoes the request. ``ok`` is ``True`` on success —
    ``result`` then carries the JSON-compatible return value (a dict, a
    list of dicts, a scalar, or ``None``). On failure ``ok`` is
    ``False`` and ``error_code`` / ``error_message`` describe the
    rejection; ``error_code`` mirrors :class:`ErrorFrame.code`
    (``"not_found"``, ``"access_denied"``, ``"protocol_error"``,
    ``"auth_failed"``, …) so the client can re-raise the matching
    :class:`NetworkError` subclass.
    """

    kind: ClassVar[str] = "response"
    request_id: str
    ok: bool
    result: Any = None
    error_code: str = ""
    error_message: str = ""


@dataclass(slots=True)
class ErrorFrame:
    """hub → client: structured rejection outside the request/response path.

    Used for handshake failures (a :class:`HelloFrame` with an unknown
    name or a failed auth claim). ``code`` is a stable identifier
    (``"protocol_error"``, ``"access_denied"``, ``"not_found"``,
    ``"auth_failed"``, …). Control-plane operation failures travel on
    :class:`ResponseFrame` instead, correlated by ``request_id``.
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
    | RequestFrame
    | ResponseFrame
    | ErrorFrame
    | NotifyFrame
    | ReceiptFrame
)


_FRAME_CLASSES: dict[str, type] = {
    "hello": HelloFrame,
    "welcome": WelcomeFrame,
    "ping": PingFrame,
    "pong": PongFrame,
    "request": RequestFrame,
    "response": ResponseFrame,
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
