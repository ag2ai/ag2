# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from a2a.server.tasks import TaskUpdater
from a2a.types import Part

from ag2.events import BaseEvent, ModelMessageChunk
from ag2.stream import MemoryStream

from ._helpers import ScriptedExecutor, make_executor_pair


async def _empty_text_parts(updater: TaskUpdater, release: asyncio.Event) -> None:
    """Streams an artifact whose chunks include empty text parts, as some servers do."""
    await updater.add_artifact(parts=[Part(text="")], artifact_id="art-1")
    await updater.add_artifact(parts=[Part(text="hello")], artifact_id="art-1", append=True)
    await updater.add_artifact(parts=[Part(text="")], artifact_id="art-1", append=True, last_chunk=True)
    await updater.complete()


@pytest.mark.asyncio
async def test_empty_text_parts_do_not_emit_empty_message_chunks() -> None:
    pair = make_executor_pair(ScriptedExecutor(_empty_text_parts), streaming=True)
    chunks: list[str] = []
    stream = MemoryStream()

    @stream.where(ModelMessageChunk).subscribe
    async def collect(event: BaseEvent) -> None:
        assert isinstance(event, ModelMessageChunk)
        chunks.append(event.content)

    reply = await pair.client.ask("ping", stream=stream)

    assert reply.response.content == "hello"
    assert chunks == ["hello"]
