# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

import redis.asyncio as aioredis

from autogen.beta.annotations import Context
from autogen.beta.context import StreamId
from autogen.beta.events import BaseEvent
from autogen.beta.history import Storage

from .serializer import Serializer, deserialize, serialize


class RedisStorage(Storage):
    """Redis-backed storage implementing the Storage protocol from autogen.beta.history."""

    def __init__(
        self,
        redis_url: str,
        prefix: str = "ag2:stream",
        serializer: Serializer = Serializer.JSON,
    ) -> None:
        self._redis = aioredis.from_url(redis_url)
        self._prefix = prefix
        self._serializer = serializer

    def _key(self, stream_id: StreamId) -> str:
        return f"{self._prefix}:{stream_id}"

    async def save_event(self, event: BaseEvent, context: Context) -> None:
        stream_id = context.stream.id
        await self._redis.rpush(self._key(stream_id), serialize(event, self._serializer))

    async def get_history(self, stream_id: StreamId) -> Iterable[BaseEvent]:
        raw = await self._redis.lrange(self._key(stream_id), 0, -1)
        return [deserialize(item, self._serializer) for item in raw]

    async def set_history(self, stream_id: StreamId, events: Iterable[BaseEvent]) -> None:
        key = self._key(stream_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.delete(key)
            for event in events:
                pipe.rpush(key, serialize(event, self._serializer))
            await pipe.execute()

    async def drop_history(self, stream_id: StreamId) -> None:
        await self._redis.delete(self._key(stream_id))

    async def close(self) -> None:
        await self._redis.aclose()
