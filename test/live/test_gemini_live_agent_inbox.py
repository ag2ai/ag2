# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

pytest.importorskip("google.genai")

from ag2.events import MessageEnqueued
from ag2.stream import MemoryStream
from test.live._announcements import Announcements
from test.live._gemini_helpers import live_agent, speech, turn_complete


def user_texts(turns: list[dict[str, Any]]) -> list[list[str]]:
    return [[part["text"] for part in turn["parts"]] for turn in turns]


@pytest.mark.asyncio
class TestLiveAgentInbox:
    async def test_message_enqueued_before_open_is_delivered_at_open(self) -> None:
        stream = MemoryStream()
        stream.enqueue("left over")
        agent, session = live_agent(stream=stream)

        async with agent.run():
            assert user_texts(session.added_turns()) == [["left over"]]
            assert session.response_requests() == 1

    async def test_enqueue_during_response_is_delivered_at_boundary(self) -> None:
        agent, session = live_agent()

        async with agent.run() as context:
            await session.emit(speech())
            announcements = Announcements()
            with context.stream.where(MessageEnqueued).sub_scope(announcements.on_enqueued):
                context.enqueue("typed")
                await announcements.wait(1)

            assert session.added_turns() == []

            await session.emit(turn_complete())

            assert user_texts(session.added_turns()) == [["typed"]]
            assert session.response_requests() == 1
