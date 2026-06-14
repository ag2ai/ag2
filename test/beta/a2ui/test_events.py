# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.a2ui import A2UIMessageEvent
from autogen.beta.events import BaseEvent


class TestA2UIMessageEvent:
    def test_is_event_and_transient(self) -> None:
        event = A2UIMessageEvent({"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}})
        assert isinstance(event, BaseEvent)
        # Derived from the model response; not persisted to durable history.
        assert event.__transient__ is True

    def test_carries_message_payload(self) -> None:
        message = {
            "version": "v0.9",
            "createSurface": {"surfaceId": "s1", "catalogId": "test"},
        }
        event = A2UIMessageEvent(message)
        assert event.message == message

    def test_reexported_from_package(self) -> None:
        import autogen.beta.a2ui as a2ui

        assert "A2UIMessageEvent" in a2ui.__all__
        assert a2ui.A2UIMessageEvent is A2UIMessageEvent
