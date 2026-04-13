# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Task event-type registry + envelope round-trip — Phase 4.

Locks the ``ag2.task.*`` event names in as :class:`EventRegistry` built-ins so
a strict-mode hub still accepts them without any operator registration, and
confirms that :class:`Envelope` carries ``task_id`` through round-trip.
"""

from __future__ import annotations

import pytest

from autogen.beta.network import (
    BUILTIN_EVENT_TYPES,
    EV_TASK_ASSIGNED,
    EV_TASK_CANCELLED,
    EV_TASK_ERROR,
    EV_TASK_EXPIRED,
    EV_TASK_PHASE_COMPLETED,
    EV_TASK_PHASE_ENTERED,
    EV_TASK_PROGRESS,
    EV_TASK_RESULT,
    Envelope,
    EventRegistry,
    TASK_EVENT_TYPES,
    TASK_TERMINAL_EVENT_TYPES,
    UnknownEventTypeError,
)


class TestTaskEventNames:
    def test_eight_task_event_types(self) -> None:
        assert TASK_EVENT_TYPES == frozenset(
            {
                EV_TASK_ASSIGNED,
                EV_TASK_PHASE_ENTERED,
                EV_TASK_PHASE_COMPLETED,
                EV_TASK_PROGRESS,
                EV_TASK_RESULT,
                EV_TASK_ERROR,
                EV_TASK_CANCELLED,
                EV_TASK_EXPIRED,
            }
        )
        assert len(TASK_EVENT_TYPES) == 8

    def test_four_terminal_events(self) -> None:
        assert TASK_TERMINAL_EVENT_TYPES == frozenset(
            {EV_TASK_RESULT, EV_TASK_ERROR, EV_TASK_CANCELLED, EV_TASK_EXPIRED}
        )
        assert TASK_TERMINAL_EVENT_TYPES.issubset(TASK_EVENT_TYPES)

    def test_stable_wire_values(self) -> None:
        assert EV_TASK_ASSIGNED == "ag2.task.assigned"
        assert EV_TASK_PHASE_ENTERED == "ag2.task.phase_entered"
        assert EV_TASK_PHASE_COMPLETED == "ag2.task.phase_completed"
        assert EV_TASK_PROGRESS == "ag2.task.progress"
        assert EV_TASK_RESULT == "ag2.task.result"
        assert EV_TASK_ERROR == "ag2.task.error"
        assert EV_TASK_CANCELLED == "ag2.task.cancelled"
        assert EV_TASK_EXPIRED == "ag2.task.expired"


class TestEventRegistryBuiltins:
    def test_task_events_are_pre_registered(self) -> None:
        for name in TASK_EVENT_TYPES:
            assert name in BUILTIN_EVENT_TYPES

    def test_permissive_registry_accepts_task_events(self) -> None:
        reg = EventRegistry()
        for name in TASK_EVENT_TYPES:
            reg.check(name)  # does not raise

    def test_strict_registry_accepts_task_events(self) -> None:
        reg = EventRegistry(strict=True)
        for name in TASK_EVENT_TYPES:
            reg.check(name)  # does not raise — they are built-ins

    def test_strict_registry_rejects_unknown_task_like_name(self) -> None:
        reg = EventRegistry(strict=True)
        with pytest.raises(UnknownEventTypeError):
            reg.check("ag2.task.reannotate")  # not a built-in

    def test_registry_reports_every_task_name(self) -> None:
        reg = EventRegistry()
        names = reg.names()
        for name in TASK_EVENT_TYPES:
            assert name in names


class TestEnvelopeTaskIdRoundTrip:
    def test_task_id_round_trips(self) -> None:
        envelope = Envelope(
            session_id="01932sess",
            sender_id="01932alice",
            event_type=EV_TASK_PROGRESS,
            event_data={"update": {"docs": 3}},
            task_id="01932task",
        )
        data = envelope.to_dict()
        assert data["task_id"] == "01932task"
        hydrated = Envelope.from_dict(data)
        assert hydrated.task_id == "01932task"
        assert hydrated.event_type == EV_TASK_PROGRESS
        assert hydrated.event_data == {"update": {"docs": 3}}

    def test_task_id_default_is_none(self) -> None:
        envelope = Envelope(
            session_id="01932sess",
            sender_id="01932alice",
            event_type="ag2.msg.text",
            event_data={"content": "hello"},
        )
        assert envelope.task_id is None
        assert envelope.to_dict()["task_id"] is None

    def test_terminal_task_events_round_trip_result_payload(self) -> None:
        # ``ag2.task.result`` carries the terminal value in event_data.
        # Round-trip preserves arbitrary JSON-serializable payloads.
        envelope = Envelope(
            session_id="01932sess",
            sender_id="01932bob",
            event_type=EV_TASK_RESULT,
            event_data={"value": {"summary": "done", "tokens": 420}},
            task_id="01932task",
        )
        hydrated = Envelope.from_json(envelope.to_json())
        assert hydrated.event_data["value"] == {"summary": "done", "tokens": 420}

    def test_task_error_round_trip(self) -> None:
        envelope = Envelope(
            session_id="01932sess",
            sender_id="01932bob",
            event_type=EV_TASK_ERROR,
            event_data={"error": "ModelTimeoutError: 30s"},
            task_id="01932task",
        )
        hydrated = Envelope.from_json(envelope.to_json())
        assert hydrated.event_data == {"error": "ModelTimeoutError: 30s"}
