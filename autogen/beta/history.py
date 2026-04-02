# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from collections.abc import Iterable
from typing import Protocol

from .annotations import Context
from .context import StreamId
from .events import BaseEvent


class Storage(Protocol):
    async def save_event(self, event: "BaseEvent", context: "Context") -> None: ...

    async def get_history(self, stream_id: "StreamId") -> Iterable["BaseEvent"]: ...

    async def set_history(self, stream_id: "StreamId", events: Iterable[BaseEvent]) -> None: ...

    async def drop_history(self, stream_id: "StreamId") -> None: ...


class MemoryStorage(Storage):
    def __init__(self) -> None:
        self.__data: defaultdict[StreamId, list[BaseEvent]] = defaultdict(list)

    async def save_event(self, event: "BaseEvent", context: "Context") -> None:
        stream_id = context.stream.id
        if event not in self.__data[stream_id]:
            self.__data[stream_id].append(event)

    async def get_history(self, stream_id: "StreamId") -> Iterable["BaseEvent"]:
        return self.__data[stream_id]

    async def set_history(self, stream_id: "StreamId", events: Iterable["BaseEvent"]) -> None:
        self.__data[stream_id] = list(events)

    async def drop_history(self, stream_id: "StreamId") -> None:
        self.__data.pop(stream_id, None)


class History:
    def __init__(self, stream_id: "StreamId", storage: Storage) -> None:
        self.stream_id = stream_id
        self.storage = storage

    async def get_events(self) -> Iterable["BaseEvent"]:
        return await self.storage.get_history(self.stream_id)

    async def replace(self, events: Iterable["BaseEvent"]) -> None:
        await self.storage.set_history(self.stream_id, events)
