# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from a2a.types import TaskState

from ag2.a2a.events import (
    A2AEvent,
    A2ATaskSnapshot,
    A2ATaskStatusUpdate,
    A2ATextArtifact,
)
from ag2.events import BaseEvent
from ag2.stream import MemoryStream

from ._helpers import make_pair


def _collect_a2a_events() -> tuple[MemoryStream, list[BaseEvent]]:
    """A stream plus the list every ``A2AEvent`` published on it lands in."""
    stream = MemoryStream()
    captured: list[BaseEvent] = []

    @stream.where(A2AEvent).subscribe
    async def collect(ev: BaseEvent) -> None:
        captured.append(ev)

    return stream, captured


def _states(events: list[BaseEvent]) -> list[int]:
    return [ev.state for ev in events if isinstance(ev, A2ATaskStatusUpdate)]


@pytest.mark.asyncio
class TestA2AEventsReachClientStream:
    async def test_streaming_publishes_the_whole_task_lifecycle(self) -> None:
        pair = make_pair("hello world", streaming=True)
        stream, captured = _collect_a2a_events()

        await pair.client.ask("ping", stream=stream)

        assert [type(ev) for ev in captured] == [
            A2ATaskSnapshot,
            A2ATaskStatusUpdate,
            A2ATextArtifact,
            A2ATextArtifact,
            A2ATaskStatusUpdate,
        ]
        assert _states(captured) == [
            TaskState.TASK_STATE_WORKING,
            TaskState.TASK_STATE_COMPLETED,
        ]

    async def test_streaming_carries_unstreamed_final_text_in_an_artifact(self) -> None:
        # The script emits a complete ``ModelMessage`` rather than per-token ``ModelMessageChunk``s.
        # Results are artifacts (spec §3.7), so the text arrives as the text artifact, and the
        # COMPLETED status carries no message.
        pair = make_pair("final reply", streaming=True)
        stream, captured = _collect_a2a_events()

        reply = await pair.client.ask("ping", stream=stream)

        assert reply.body == "final reply"
        assert [ev.text for ev in captured if isinstance(ev, A2ATextArtifact)] == ["final reply", ""]
        [final] = [
            ev for ev in captured if isinstance(ev, A2ATaskStatusUpdate) and ev.state == TaskState.TASK_STATE_COMPLETED
        ]
        assert not final.update.status.HasField("message")

    async def test_polling_publishes_only_the_bootstrap_snapshot(self) -> None:
        # Polling drains just the bootstrap send-message response through the
        # event-publishing path; the later get_task polls are absorbed without
        # re-emitting. Subscribers therefore see the task appear, but not its
        # lifecycle — pinning that here so the asymmetry with streaming is a
        # deliberate, visible contract rather than an accident.
        pair = make_pair("polled reply", streaming=False)
        stream, captured = _collect_a2a_events()

        reply = await pair.client.ask("ping", stream=stream)

        assert reply.body == "polled reply"
        assert [type(ev) for ev in captured] == [A2ATaskSnapshot]
