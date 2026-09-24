# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the AG-UI subscriber's handling of empty model events.

The AG-UI protocol forbids empty deltas (``MinLen(1)`` on
``TextMessageContentEvent.delta`` / ``TextMessageChunkEvent.delta``).
``map_events_to_ag_ui`` must drop empty ``ModelMessageChunk`` and empty
``ModelMessage`` events instead of forwarding them to the AG-UI encoder.
"""

import pytest
from ag_ui.core import UserMessage

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.events import ModelMessageChunk
from ag2.testing import TestConfig

from .utils import (
    assert_no_event_type,
    collect_events,
    create_run_input,
    get_events_of_type,
)

pytestmark = pytest.mark.asyncio


class TestEmptyModelMessageChunk:
    async def test_empty_chunk_does_not_open_text_message(self) -> None:
        # Empty chunk arrives first — must not emit TEXT_MESSAGE_START.
        # The non-empty chunk that follows opens the message normally.
        agent = Agent("test_agent", config=TestConfig(ModelMessageChunk(""), ModelMessageChunk("Hello"), "Hello"))
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="m1", content="hi"))

        events = await collect_events(stream, run_input)

        starts = get_events_of_type(events, "TEXT_MESSAGE_START")
        contents = get_events_of_type(events, "TEXT_MESSAGE_CONTENT")

        # Exactly one start (from the non-empty chunk), one content event.
        assert len(starts) == 1
        assert len(contents) == 1
        assert contents[0]["delta"] == "Hello"

    async def test_empty_chunk_between_real_chunks_dropped(self) -> None:
        agent = Agent(
            "test_agent",
            config=TestConfig(
                ModelMessageChunk("Hello"), ModelMessageChunk(""), ModelMessageChunk(" world"), "Hello world"
            ),
        )
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="m1", content="hi"))

        events = await collect_events(stream, run_input)

        contents = get_events_of_type(events, "TEXT_MESSAGE_CONTENT")
        assert [c["delta"] for c in contents] == ["Hello", " world"]

        # No content event ever carries an empty delta.
        for event in contents:
            assert event["delta"] != ""


class TestEmptyModelMessage:
    async def test_empty_non_streaming_message_emits_no_text_event(self) -> None:
        agent = Agent("test_agent", config=TestConfig(""))
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="m1", content="hi"))

        events = await collect_events(stream, run_input)

        assert_no_event_type(events, "TEXT_MESSAGE_CHUNK")
        assert_no_event_type(events, "TEXT_MESSAGE_CONTENT")
        assert_no_event_type(events, "TEXT_MESSAGE_START")
