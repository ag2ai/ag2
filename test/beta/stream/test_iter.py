# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import AsyncIterable
from unittest.mock import MagicMock

import pytest

from autogen.beta import Context, MemoryStream
from autogen.beta.events import BaseEvent, ModelMessage, ToolCallEvent


class TestStreamSend:
    @pytest.mark.asyncio
    async def test_send_event_to_iter_subscriber(self, signal: asyncio.Event, mock: MagicMock) -> None:
        stream = MemoryStream()

        stream.subscribe(lambda ev: mock(ev))

        async def listen_stream(events: AsyncIterable[BaseEvent]) -> None:
            async for ev in events:
                mock.iter(ev)
                signal.set()
                break

        with stream.join() as events:
            asyncio.create_task(listen_stream(events))

            event = ToolCallEvent(name="func1", arguments='"test"')
            await stream.send(event, context=Context(stream))

            await asyncio.wait_for(signal.wait(), timeout=1.0)

        mock.assert_called_once_with(event)
        mock.iter.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_iter_subscriber_max_msgs(self, signal: asyncio.Event, mock: MagicMock) -> None:
        stream = MemoryStream()

        async def listen_stream(events: AsyncIterable[BaseEvent]) -> None:
            async for ev in events:
                mock(ev)
            signal.set()

        with stream.join(max_events=2) as events:
            asyncio.create_task(listen_stream(events))

            for _ in range(5):
                # publish more messages than expected
                await stream.send(
                    ToolCallEvent(name="func1", arguments='"test"'),
                    context=Context(stream),
                )

            await asyncio.wait_for(signal.wait(), timeout=1.0)

        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_iter_substream(self, signal: asyncio.Event, mock: MagicMock) -> None:
        stream = MemoryStream()

        async def listen_stream(events: AsyncIterable[BaseEvent]) -> None:
            async for ev in events:
                mock(ev)
                signal.set()
                break

        with stream.where(ToolCallEvent).join() as events:
            asyncio.create_task(listen_stream(events))

            await stream.send(
                ModelMessage("test"),
                context=Context(stream),
            )
            event = ToolCallEvent(name="func1", arguments='"test"')
            await stream.send(event, context=Context(stream))

            await asyncio.wait_for(signal.wait(), timeout=1.0)

        mock.assert_called_once_with(event)
