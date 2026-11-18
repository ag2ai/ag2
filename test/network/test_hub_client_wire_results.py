# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``HubClient`` checks the JSON type of a remote hub's RPC results.

A hub of another version (or a buggy one) may answer an op with the
wrong JSON type. The client raises ``NetworkError`` at the boundary
instead of handing the wrong value to the caller.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_TEXT,
    Envelope,
    Frame,
    Hub,
    HubClient,
    NetworkError,
    RequestFrame,
    ResponseFrame,
)
from ag2.network.hub import task_checkpoint_path


class _AnsweringLinkClient:
    """Answers every ``RequestFrame`` with ``ok=True`` and a fixed result."""

    def __init__(self, result: Any) -> None:
        self.endpoint_id = "answering-ep"
        self._result = result
        self._inbound: asyncio.Queue[Frame | None] = asyncio.Queue()

    async def open(self) -> None:
        pass

    async def send_frame(self, frame: Frame) -> None:
        if isinstance(frame, RequestFrame):
            await self._inbound.put(ResponseFrame(request_id=frame.request_id, ok=True, result=self._result))

    async def frames(self) -> AsyncIterator[Frame]:
        while (frame := await self._inbound.get()) is not None:
            yield frame

    async def close(self) -> None:
        await self._inbound.put(None)


class _AnsweringLink:
    """A ``LinkFactory`` for a remote hub that answers every op with ``result``."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def client(self) -> _AnsweringLinkClient:
        return _AnsweringLinkClient(self._result)


@pytest.mark.asyncio
class TestHubClientWireResults:
    async def test_get_skill_rejects_a_non_string_result(self) -> None:
        hub_client = HubClient(_AnsweringLink({"skill": "md"}))
        try:
            with pytest.raises(NetworkError, match="get_skill returned dict"):
                await hub_client.get_skill("alice")
        finally:
            await hub_client.close()

    async def test_post_envelope_rejects_a_non_string_envelope_id(self) -> None:
        hub_client = HubClient(_AnsweringLink(42))
        envelope = Envelope(
            channel_id="c1",
            sender_id="alice",
            audience=None,
            event_type=EV_TEXT,
            event_data={"text": "hi"},
        )
        try:
            with pytest.raises(NetworkError, match="post_envelope returned int"):
                await hub_client.post_envelope(envelope)
        finally:
            await hub_client.close()

    async def test_read_task_checkpoint_rejects_a_non_object_result(self) -> None:
        hub_client = HubClient(_AnsweringLink(["step", 3]))
        try:
            with pytest.raises(NetworkError, match="read_task_checkpoint returned list"):
                await hub_client.read_task_checkpoint("task-1")
        finally:
            await hub_client.close()


@pytest.mark.asyncio
async def test_hub_rejects_a_checkpoint_file_that_is_not_an_object() -> None:
    store = MemoryKnowledgeStore()
    await store.write(task_checkpoint_path("task-1"), "[1, 2, 3]")
    hub = await Hub.open(store, ttl_sweep_interval=0, expectation_sweep_interval=0)
    try:
        with pytest.raises(ValueError, match="'task-1' is not a JSON object"):
            await hub.read_task_checkpoint("task-1")
    finally:
        await hub.close()
