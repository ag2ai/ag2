# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Envelope roundtrip + helpers."""

from __future__ import annotations

import pytest

from autogen.beta.network.envelope import EV_SESSION_INVITE, EV_TEXT, Envelope


def test_text_envelope_helper_builds_ag2_msg_text() -> None:
    env = Envelope.text(
        session_id="01-ses",
        sender_id="01-alice",
        content="hello",
        recipient_id="01-bob",
        trace_id="01-trace",
    )
    assert env.event_type == EV_TEXT
    assert env.event_data == {"content": "hello"}
    assert env.content() == "hello"
    assert env.recipient_id == "01-bob"
    assert env.trace_id == "01-trace"
    assert env.envelope_id is None  # hub will stamp it


def test_envelope_roundtrip_preserves_all_fields() -> None:
    env = Envelope(
        session_id="ses",
        sender_id="alice",
        event_type=EV_SESSION_INVITE,
        event_data={"reason": "research"},
        envelope_id="env-1",
        recipient_id="bob",
        task_id="t-1",
        causation_id="env-0",
        trace_id="trace-1",
        priority="urgent",
        created_at="2026-04-12T18:22:01Z",
        ttl_seconds=60,
        idempotency_key="idem-1",
        metadata={"audit": "yes"},
    )
    restored = Envelope.from_dict(env.to_dict())
    assert restored == env


def test_envelope_json_roundtrip() -> None:
    env = Envelope.text(session_id="s", sender_id="a", content="hey")
    restored = Envelope.from_json(env.to_json())
    assert restored.session_id == "s"
    assert restored.sender_id == "a"
    assert restored.content() == "hey"


def test_content_raises_for_non_text_event() -> None:
    env = Envelope(
        session_id="s",
        sender_id="a",
        event_type=EV_SESSION_INVITE,
        event_data={"reason": "x"},
    )
    with pytest.raises(KeyError):
        env.content()


def test_envelope_default_priority_is_normal() -> None:
    env = Envelope(session_id="s", sender_id="a", event_type=EV_TEXT)
    assert env.priority == "normal"


@pytest.mark.parametrize("prio", ["background", "normal", "urgent"])
def test_envelope_accepts_every_valid_priority(prio: str) -> None:
    env = Envelope(session_id="s", sender_id="a", event_type=EV_TEXT, priority=prio)
    assert env.priority == prio


def test_envelope_rejects_unknown_priority() -> None:
    with pytest.raises(ValueError, match="priority"):
        Envelope(session_id="s", sender_id="a", event_type=EV_TEXT, priority="critical")


def test_envelope_from_dict_rejects_unknown_priority() -> None:
    bad = {
        "session_id": "s",
        "sender_id": "a",
        "event": {"type": EV_TEXT, "data": {"content": "x"}},
        "priority": "critical",
    }
    with pytest.raises(ValueError, match="priority"):
        Envelope.from_dict(bad)
