# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

import pytest

pytest.importorskip("google.genai")

from google.genai import types as gtypes

from ag2.events import DataInput, ModelRequest, TextInput
from test.live._gemini_helpers import live_agent, speech, turn_complete


@dataclass
class Booking:
    id: UUID
    at: datetime


@pytest.mark.asyncio
class TestPushedModelRequest:
    async def test_pushed_model_request_reaches_the_connection(self) -> None:
        agent, session = live_agent()

        async with agent.run() as context:
            await context.send(ModelRequest([TextInput("first"), TextInput("second")]))

            assert (
                "send_client_content",
                {
                    "turns": [{"role": "user", "parts": [{"text": "first"}, {"text": "second"}]}],
                    "turn_complete": True,
                },
            ) in session.calls

    async def test_push_to_idle_session_is_sent_and_answered_at_once(self) -> None:
        agent, session = live_agent()

        async with agent.run() as context:
            await context.send(ModelRequest([TextInput("hello")]))

            assert session.calls[-1:] == [
                (
                    "send_client_content",
                    {"turns": [{"role": "user", "parts": [{"text": "hello"}]}], "turn_complete": True},
                ),
            ]

    async def test_push_during_response_sends_nothing_until_boundary(self) -> None:
        agent, session = live_agent()

        async with agent.run() as context:
            await session.emit(speech())
            calls_before = list(session.calls)
            await context.send(ModelRequest([TextInput("hello")]))

            # Any client content would cut the response off.
            assert session.calls == calls_before

            await session.emit(turn_complete())

            assert session.calls[-1:] == [
                (
                    "send_client_content",
                    {"turns": [{"role": "user", "parts": [{"text": "hello"}]}], "turn_complete": True},
                ),
            ]

    async def test_pushes_during_one_response_share_one_request(self) -> None:
        agent, session = live_agent()

        async with agent.run() as context:
            await session.emit(speech())
            await context.send(ModelRequest([TextInput("first")]))
            await context.send(ModelRequest([TextInput("second")]))
            await context.send(ModelRequest([TextInput("third")]))

            assert session.response_requests() == 0

            await session.emit(turn_complete())

            assert session.response_requests() == 1
            assert len(session.added_turns()) == 3

    async def test_requested_response_counts_as_active_until_its_turn_completes(self) -> None:
        agent, session = live_agent()

        async with agent.run() as context:
            await context.send(ModelRequest([TextInput("first")]))
            await context.send(ModelRequest([TextInput("second")]))

            assert session.response_requests() == 1

            await session.emit(speech(), turn_complete())

            assert session.response_requests() == 2

    @pytest.mark.skipif(
        not hasattr(gtypes, "InteractionStatus"),
        reason="google-genai before 2.18 reports no interaction_status",
    )
    async def test_deferred_request_waits_while_the_server_reports_more_work(self) -> None:
        agent, session = live_agent()

        async with agent.run() as context:
            await session.emit(speech())
            await context.send(ModelRequest([TextInput("hello")]))
            await session.emit(turn_complete(more_coming=True))

            assert session.response_requests() == 0

            await session.emit(speech(), turn_complete())

            assert session.response_requests() == 1

    async def test_empty_request_sends_nothing(self) -> None:
        agent, session = live_agent()

        async with agent.run() as context:
            calls_before = list(session.calls)
            await context.send(ModelRequest([]))

            assert session.calls == calls_before

    async def test_pushed_data_input_is_sent_with_the_agent_serializer(self) -> None:
        agent, session = live_agent()
        booking = Booking(
            id=UUID("12345678-1234-5678-1234-567812345678"),
            at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        )

        async with agent.run() as context:
            await context.send(ModelRequest([DataInput(booking)]))

            assert session.added_turns() == [
                {
                    "role": "user",
                    "parts": [{"text": '{"id":"12345678-1234-5678-1234-567812345678","at":"2026-01-02T03:04:05Z"}'}],
                },
            ]
