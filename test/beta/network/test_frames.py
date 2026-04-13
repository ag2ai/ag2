# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Frame (de)serialization roundtrips for the Link wire vocabulary."""

from __future__ import annotations

import pytest

from autogen.beta.network.envelope import Envelope
from autogen.beta.network.errors import FrameError
from autogen.beta.network.transport.frames import (
    AcceptFrame,
    ChunkFrame,
    ErrorFrame,
    EventFrame,
    HelloFrame,
    NotifyFrame,
    PingFrame,
    PongFrame,
    ReceiptFrame,
    RuleChangedFrame,
    SendFrame,
    SubscribeFrame,
    UnsubscribeFrame,
    WelcomeFrame,
    decode_frame,
    encode_frame,
)


def _sample_envelope() -> Envelope:
    return Envelope.text(session_id="s1", sender_id="a1", content="hi", recipient_id="a2")


@pytest.mark.parametrize(
    "frame",
    [
        HelloFrame(
            identity={"name": "a"},
            rule={"version": 1},
            auth_claim={"token": "abc"},
        ),
        WelcomeFrame(actor_id="act-1", hub_id="hub-1"),
        NotifyFrame(envelope=_sample_envelope()),
        ReceiptFrame(envelope_id="env-1", status="ack"),
        ReceiptFrame(envelope_id="env-1", status="nack", reason="rejected"),
        SendFrame(envelope=_sample_envelope(), idempotency_key="idem-1"),
        AcceptFrame(envelope_id="env-1", wal_offset=4096, request_id="req-1"),
        AcceptFrame(envelope_id="env-2"),  # defaults
        ErrorFrame(code="access_denied", message="nope", details={"reason": "rule"}),
        SubscribeFrame(subscription_id="sub-1", session_id="s1"),
        SubscribeFrame(subscription_id="sub-2", task_id="t1", since=42),
        SubscribeFrame(subscription_id="sub-3", session_id="s1", causation_id="env-0"),
        UnsubscribeFrame(subscription_id="sub-1"),
        EventFrame(subscription_id="sub-1", envelope=_sample_envelope()),
        ChunkFrame(envelope_id="env-1", chunk_index=0, content="hello", final=False),
        ChunkFrame(envelope_id="env-1", chunk_index=5, content="", final=True),
        RuleChangedFrame(
            actor_id="act-1",
            transforms=[{"stage": "pre_receive", "apply": "redact"}],
            version=2,
        ),
        PingFrame(timestamp_ms=1712944921000),
        PongFrame(timestamp_ms=1712944921050),
    ],
)
def test_frame_roundtrips_via_json(frame: object) -> None:
    encoded = encode_frame(frame)  # type: ignore[arg-type]
    decoded = decode_frame(encoded)
    assert type(decoded) is type(frame)
    assert decoded == frame


def test_decode_frame_accepts_parsed_dict() -> None:
    frame = WelcomeFrame(actor_id="a", hub_id="h")
    data = {"type": "welcome", "actor_id": "a", "hub_id": "h"}
    decoded = decode_frame(data)
    assert decoded == frame


def test_decode_frame_rejects_unknown_type() -> None:
    with pytest.raises(FrameError):
        decode_frame({"type": "not_a_frame"})


def test_notify_frame_carries_envelope_roundtrip_fields() -> None:
    env = Envelope.text(
        session_id="s",
        sender_id="a",
        content="hi",
        recipient_id="b",
        trace_id="t",
    )
    env.envelope_id = "env-42"
    frame = NotifyFrame(envelope=env)
    decoded = decode_frame(encode_frame(frame))
    assert isinstance(decoded, NotifyFrame)
    assert decoded.envelope.envelope_id == "env-42"
    assert decoded.envelope.content() == "hi"
