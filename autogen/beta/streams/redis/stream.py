# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
from uuid import uuid4

import redis.asyncio as aioredis

from autogen.beta.annotations import Context
from autogen.beta.context import StreamId
from autogen.beta.events import BaseEvent
from autogen.beta.stream import MemoryStream

from .serializer import Serializer, deserialize, serialize
from .storage import RedisStorage

# Byte used to separate the origin instance ID from the serialized payload
# in Redis Pub/Sub messages.  Works for both JSON and pickle payloads.
_ORIGIN_SEP = b"\x1e"  # ASCII Record Separator


class RedisStream(MemoryStream):
    """A full-featured stream with Redis-backed pub/sub and persistent event history.

    All events flow through Redis Pub/Sub, ensuring subscribers across processes
    and machines receive every event. History is persisted to Redis.

    Event flow:
        send() → persist to Redis + dispatch locally + publish to Pub/Sub
        listener → receives from Pub/Sub → skips self-originated → dispatches

    Local subscribers see events immediately without a Redis roundtrip.
    Remote listeners receive events via Pub/Sub and dispatch to their own
    local subscribers using a captured base context for dependency injection.

    Args:
        redis_url: Redis connection URL.
        prefix: Key prefix for Redis storage and pub/sub channels.
        id: Stream ID. If None, a new UUID is generated.
        serializer: Serialization format (Serializer.JSON or Serializer.PICKLE).
    """

    def __init__(
        self,
        redis_url: str,
        *,
        prefix: str = "ag2:stream",
        id: StreamId | None = None,
        serializer: Serializer = Serializer.JSON,
    ) -> None:
        storage = RedisStorage(redis_url, prefix=prefix, serializer=serializer)
        super().__init__(storage, id=id)

        # Unsubscribe the auto-registered save_event from MemoryStream.__init__
        # We handle persistence explicitly in send() to avoid double-writes
        storage_sub_id = next(iter(self._subscribers))
        self.unsubscribe(storage_sub_id)

        self._redis_storage = storage
        self._redis_url = redis_url
        self._prefix = prefix
        self._serializer = serializer
        self._channel = f"{prefix}:pubsub:{self.id}"
        self._instance_id = str(uuid4())
        self._listener_task: asyncio.Task | None = None
        self._listener_ready = asyncio.Event()
        self._pubsub_redis = aioredis.from_url(redis_url)
        self._publish_redis = aioredis.from_url(redis_url)
        # Captured on first send() call so the listener can provide
        # proper dependency_provider context and dependencies to
        # subscribers processing events from remote instances.
        self._base_context: Context | None = None

    def _ensure_listener(self) -> None:
        """Start the Redis Pub/Sub listener if not already running."""
        if self._listener_task is None or self._listener_task.done():
            self._listener_ready.clear()
            self._listener_task = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        """Listen for events on the Redis Pub/Sub channel and dispatch to local subscribers."""
        pubsub = self._pubsub_redis.pubsub()
        await pubsub.subscribe(self._channel)
        self._listener_ready.set()
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                origin, payload = self._split_origin(message["data"])
                # Skip events published by this instance — they were already
                # dispatched to local subscribers in send().
                if origin == self._instance_id:
                    continue
                event = deserialize(payload, self._serializer)
                if isinstance(event, BaseEvent):
                    ctx = self._base_context or Context(self)
                    await super().send(event, ctx)
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe(self._channel)
            await pubsub.aclose()

    @staticmethod
    def _split_origin(raw: bytes) -> tuple[str | None, bytes]:
        """Split an origin-tagged message into (origin_id, payload).

        Messages are framed as ``<instance_id> <RS> <payload>``.
        If the separator is missing the entire message is treated as payload.
        """
        parts = raw.split(_ORIGIN_SEP, 1)
        if len(parts) == 2:
            return parts[0].decode(), parts[1]
        return None, raw

    async def send(self, event: BaseEvent, context: Context) -> None:
        """Persist the event, dispatch locally, and publish to Redis for remote listeners."""
        self._ensure_listener()
        await self._listener_ready.wait()
        # Capture the first caller's context so the listener can reuse its
        # dependency_provider and dependencies for remotely-received events.
        if self._base_context is None:
            self._base_context = context
        # Persist once — only the sender writes to history
        await self._redis_storage.save_event(event, context)
        # Dispatch to local subscribers immediately, avoiding a Redis roundtrip
        await super().send(event, context)
        # Publish to Redis with origin tag — remote listeners dispatch,
        # local listener skips (already dispatched above).
        payload = serialize(event, self._serializer)
        tagged = self._instance_id.encode() + _ORIGIN_SEP + payload
        await self._publish_redis.publish(self._channel, tagged)

    async def close(self) -> None:
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task
        await self._pubsub_redis.aclose()
        await self._publish_redis.aclose()
        await self._redis_storage.close()
