# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from autogen.agents.experimental.a2ui.ag_ui_interceptor import create_a2ui_event_interceptor


def _make_response(content: str | None) -> Any:
    """Create a mock ServiceResponse-like object."""

    class MockResponse:
        def __init__(self, content: str | None) -> None:
            self.message: dict[str, Any] | None = {"content": content} if content is not None else None

    return MockResponse(content)


async def _collect_events(interceptor: Any, response: Any) -> list[Any]:
    """Collect all events yielded by the interceptor."""
    events = []
    async for event in interceptor(response):
        events.append(event)
    return events


class TestA2UIEventInterceptor:
    @pytest.mark.asyncio
    async def test_extracts_a2ui_and_yields_event(self) -> None:
        interceptor = create_a2ui_event_interceptor()
        response = _make_response(
            "Here is your UI.\n---a2ui_JSON---\n"
            '[{"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": "test"}}]'
        )

        events = await _collect_events(interceptor, response)

        assert len(events) == 1
        event = events[0]
        assert event.activity_type == "a2ui-surface"
        assert "operations" in event.content
        assert len(event.content["operations"]) == 1
        assert event.content["operations"][0]["createSurface"]["surfaceId"] == "s1"

    @pytest.mark.asyncio
    async def test_strips_a2ui_from_response_text(self) -> None:
        interceptor = create_a2ui_event_interceptor()
        response = _make_response(
            'Hello there.\n---a2ui_JSON---\n[{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]'
        )

        await _collect_events(interceptor, response)

        assert response.message is not None
        assert response.message["content"] == "Hello there."
        assert "---a2ui_JSON---" not in response.message["content"]

    @pytest.mark.asyncio
    async def test_nulls_message_when_text_only_is_empty(self) -> None:
        interceptor = create_a2ui_event_interceptor()
        response = _make_response('---a2ui_JSON---\n[{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]')

        await _collect_events(interceptor, response)

        assert response.message is None

    @pytest.mark.asyncio
    async def test_no_a2ui_passes_through(self) -> None:
        interceptor = create_a2ui_event_interceptor()
        response = _make_response("Just plain text, no UI here.")

        events = await _collect_events(interceptor, response)

        assert len(events) == 0
        assert response.message is not None
        assert response.message["content"] == "Just plain text, no UI here."

    @pytest.mark.asyncio
    async def test_none_message_passes_through(self) -> None:
        interceptor = create_a2ui_event_interceptor()
        response = _make_response(None)

        events = await _collect_events(interceptor, response)

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_invalid_json_skipped(self) -> None:
        interceptor = create_a2ui_event_interceptor()
        response = _make_response("Text.\n---a2ui_JSON---\n{not valid json}")

        events = await _collect_events(interceptor, response)

        assert len(events) == 0  # No event for invalid JSON

    @pytest.mark.asyncio
    async def test_custom_delimiter(self) -> None:
        interceptor = create_a2ui_event_interceptor(delimiter="<<<A2UI>>>")
        response = _make_response('UI below.\n<<<A2UI>>>\n[{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]')

        events = await _collect_events(interceptor, response)

        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_custom_activity_type(self) -> None:
        interceptor = create_a2ui_event_interceptor(activity_type="custom-a2ui")
        response = _make_response('Text.\n---a2ui_JSON---\n[{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]')

        events = await _collect_events(interceptor, response)

        assert events[0].activity_type == "custom-a2ui"

    @pytest.mark.asyncio
    async def test_multiple_operations(self) -> None:
        interceptor = create_a2ui_event_interceptor()
        response = _make_response(
            "Text.\n---a2ui_JSON---\n"
            '[{"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": "test"}}, '
            '{"version": "v0.9", "updateComponents": {"surfaceId": "s1", '
            '"components": [{"id": "root", "component": "Text", "text": "Hi"}]}}]'
        )

        events = await _collect_events(interceptor, response)

        assert len(events) == 1
        assert len(events[0].content["operations"]) == 2

    @pytest.mark.asyncio
    async def test_default_interceptor_import(self) -> None:
        from autogen.agents.experimental.a2ui import a2ui_event_interceptor

        assert callable(a2ui_event_interceptor)
