# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from a2a.types import Part, TaskState

from ag2.a2a.events import (
    A2AEvent,
    A2ATaskSnapshot,
    A2ATaskStatusUpdate,
)
from ag2.events import BaseEvent
from ag2.stream import MemoryStream

from ._helpers import make_pair


@pytest.mark.asyncio
class TestA2AEventsReachClientStream:
    async def test_streaming_publishes_a2a_events_to_user_stream(self) -> None:
        pair = make_pair("hello world", streaming=True)
        stream = MemoryStream()
        captured: list[BaseEvent] = []

        @stream.where(A2AEvent).subscribe
        async def collect(ev: BaseEvent) -> None:
            captured.append(ev)

        await pair.client.ask("ping", stream=stream)

        # Every captured event must be one of our typed wrappers.
        assert captured, "expected at least one A2AEvent in the user stream"
        assert all(isinstance(ev, A2AEvent) for ev in captured)

    async def test_streaming_carries_final_text_on_completion_status(self) -> None:
        # ``StatelessScript`` mock emits a complete ``ModelMessage`` rather
        # than per-token ``ModelMessageChunk``s, so the server finalises
        # via ``updater.complete(message=...)`` and the wire surfaces the
        # text on the COMPLETED ``status.message``, not on a separate
        # message payload.
        pair = make_pair("final reply", streaming=True)
        stream = MemoryStream()
        completed: list[A2ATaskStatusUpdate] = []

        @stream.where(A2ATaskStatusUpdate).subscribe
        async def collect(ev: A2ATaskStatusUpdate) -> None:
            if ev.state == TaskState.TASK_STATE_COMPLETED:
                completed.append(ev)

        reply = await pair.client.ask("ping", stream=stream)

        assert reply.response.content == "final reply"
        [final] = completed
        assert list(final.update.status.message.parts) == [Part(text="final reply")]

    async def test_streaming_emits_completed_status_update(self) -> None:
        pair = make_pair("done", streaming=True)
        stream = MemoryStream()
        status_updates: list[A2ATaskStatusUpdate] = []

        @stream.where(A2ATaskStatusUpdate).subscribe
        async def collect(ev: A2ATaskStatusUpdate) -> None:
            status_updates.append(ev)

        await pair.client.ask("ping", stream=stream)

        assert any(s.state == TaskState.TASK_STATE_COMPLETED for s in status_updates)

    async def test_polling_publishes_initial_task_snapshot(self) -> None:
        # Polling drains only the bootstrap ``send_message`` response
        # through ``_drain_stream`` (where typed events are published);
        # subsequent ``get_task`` polls feed ``_absorb_task_artifacts``
        # without re-emitting events. Confirm at least the bootstrap
        # ``A2ATaskSnapshot`` reaches the user stream so observers can
        # latch onto the task lifecycle even in polling mode.
        pair = make_pair("polled reply", streaming=False)
        stream = MemoryStream()
        captured: list[BaseEvent] = []

        @stream.where(A2AEvent).subscribe
        async def collect(ev: BaseEvent) -> None:
            captured.append(ev)

        reply = await pair.client.ask("ping", stream=stream)

        assert reply.response.content == "polled reply"
        assert any(isinstance(ev, A2ATaskSnapshot) for ev in captured)
