# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Context, MemoryStream
from ag2.events import BaseEvent, ModelMessage
from ag2.history import MemoryStorage


class _SizedMemoryStorage(MemoryStorage):
    def __init__(self) -> None:
        super().__init__()
        self.saved_count = 0

    def __len__(self) -> int:
        return self.saved_count

    async def save_event(self, event: BaseEvent, context: Context) -> None:
        await super().save_event(event, context)
        self.saved_count += 1


@pytest.mark.asyncio
async def test_explicit_empty_storage_is_not_replaced() -> None:
    storage = _SizedMemoryStorage()
    assert len(storage) == 0

    stream = MemoryStream(storage=storage)
    event = ModelMessage("hello")
    await stream.send(event, Context(stream))

    assert list(await storage.get_history(stream.id)) == [event]
    assert len(storage) == 1
