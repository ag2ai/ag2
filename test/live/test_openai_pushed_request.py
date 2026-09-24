# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("openai")

from ag2.events import ModelRequest, TextInput
from test.live._helpers import created, done, live_agent


@pytest.mark.asyncio
class TestPushedModelRequest:
    async def test_pushed_model_request_reaches_the_connection(self) -> None:
        agent, conn = live_agent()

        async with agent.run() as context:
            await context.send(ModelRequest([TextInput("first"), TextInput("second")]))

            assert conn.created_items() == [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "first"},
                        {"type": "input_text", "text": "second"},
                    ],
                },
            ]

    async def test_push_to_idle_session_adds_item_then_requests_response(self) -> None:
        agent, conn = live_agent()

        async with agent.run() as context:
            await context.send(ModelRequest([TextInput("hello")]))

            assert [name for name, _ in conn.calls[-2:]] == ["conversation.item.create", "response.create"]

    async def test_push_during_response_is_added_now_and_answered_at_boundary(self) -> None:
        agent, conn = live_agent()

        async with agent.run() as context:
            await conn.emit(created("resp-1"))
            await context.send(ModelRequest([TextInput("hello")]))

            assert len(conn.created_items()) == 1
            assert conn.response_requests() == 0

            await conn.emit(done("resp-1"))

            assert conn.response_requests() == 1

    async def test_empty_request_sends_nothing(self) -> None:
        agent, conn = live_agent()

        async with agent.run() as context:
            calls_before = list(conn.calls)
            await context.send(ModelRequest([]))

            assert conn.calls == calls_before
