# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
from collections import defaultdict
from unittest.mock import patch
from uuid import uuid4

import pytest
import pytest_asyncio

from autogen.beta import Context
from autogen.beta.events import ModelMessage, TextInput, ToolCallEvent
from autogen.beta.streams.redis.serializer import Serializer

pytestmark = [
    pytest.mark.redis,
    pytest.mark.asyncio,
]

REDIS_URL = "redis://mocked"


class MockRedis:
    """In-memory fake of redis.asyncio.Redis for testing."""

    def __init__(self) -> None:
        self._data: dict[str, list[bytes]] = defaultdict(list)
        self._pubsub_channels: dict[str, list[MockPubSub]] = defaultdict(list)

    async def rpush(self, key: str, value: bytes) -> None:
        self._data[key].append(value)

    async def lrange(self, key: str, start: int, end: int) -> list[bytes]:
        if end == -1:
            return list(self._data[key][start:])
        return list(self._data[key][start : end + 1])

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def publish(self, channel: str, message: bytes) -> int:
        subs = self._pubsub_channels.get(channel, [])
        for ps in subs:
            await ps._receive(message)
        return len(subs)

    def pipeline(self, transaction: bool = True) -> "MockPipeline":
        return MockPipeline(self)

    def pubsub(self) -> "MockPubSub":
        return MockPubSub(self)

    async def aclose(self) -> None:
        pass


class MockPipeline:
    def __init__(self, redis: MockRedis) -> None:
        self._redis = redis
        self._ops: list[tuple[str, tuple]] = []

    async def __aenter__(self) -> "MockPipeline":
        return self

    async def __aexit__(self, *args: object) -> None:
        pass

    def delete(self, key: str) -> None:
        self._ops.append(("delete", (key,)))

    def rpush(self, key: str, value: bytes) -> None:
        self._ops.append(("rpush", (key, value)))

    async def execute(self) -> None:
        for op, args in self._ops:
            await getattr(self._redis, op)(*args)
        self._ops.clear()


class MockPubSub:
    def __init__(self, redis: MockRedis) -> None:
        self._redis = redis
        self._channels: list[str] = []
        self._queue: asyncio.Queue[dict] = asyncio.Queue()

    async def subscribe(self, channel: str) -> None:
        self._channels.append(channel)
        self._redis._pubsub_channels[channel].append(self)
        await self._queue.put({"type": "subscribe", "data": 1, "channel": channel})

    async def unsubscribe(self, channel: str) -> None:
        if channel in self._channels:
            self._channels.remove(channel)
            self._redis._pubsub_channels[channel].remove(self)

    async def _receive(self, data: bytes) -> None:
        await self._queue.put({"type": "message", "data": data})

    async def listen(self):  # noqa: ANN201
        while True:
            msg = await self._queue.get()
            yield msg

    async def aclose(self) -> None:
        pass


# Shared fake Redis instance per test — all RedisStream instances in the same
# test connect to the same "server" so cross-instance pub/sub works.
@pytest.fixture()
def mock_redis():
    return MockRedis()


@pytest_asyncio.fixture(params=[Serializer.JSON, Serializer.PICKLE], ids=["json", "pickle"])
async def redis_stream(mock_redis, request):
    from autogen.beta.streams.redis import RedisStream

    serializer = request.param
    streams: list = []

    def _make(**kwargs):
        kwargs.setdefault("prefix", f"ag2:test:{uuid4()}")
        kwargs.setdefault("serializer", serializer)
        with (
            patch("autogen.beta.streams.redis.stream.aioredis", create=True) as mock_aioredis,
            patch("autogen.beta.streams.redis.storage.aioredis", create=True) as mock_storage_aioredis,
        ):
            mock_aioredis.from_url.return_value = mock_redis
            mock_storage_aioredis.from_url.return_value = mock_redis
            s = RedisStream(REDIS_URL, **kwargs)
        streams.append(s)
        return s

    yield _make

    for s in streams:
        await s.close()


@pytest.fixture(params=[Serializer.JSON, Serializer.PICKLE], ids=["json", "pickle"])
def redis_storage(mock_redis, request):
    from autogen.beta.streams.redis import RedisStorage

    serializer = request.param

    def _make(prefix=None):
        with patch("autogen.beta.streams.redis.storage.aioredis", create=True) as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_redis
            return RedisStorage(REDIS_URL, prefix=prefix or f"ag2:test:{uuid4()}", serializer=serializer)

    return _make


@pytest.fixture()
def stream_id():
    return uuid4()


class TestRedisStorage:
    async def test_save_and_get_history(self, redis_storage, stream_id):
        storage = redis_storage()
        from autogen.beta.stream import MemoryStream

        stream = MemoryStream(storage, id=stream_id)
        ctx = Context(stream)

        event = ToolCallEvent(name="func1", arguments="test")
        await storage.save_event(event, ctx)

        history = list(await storage.get_history(stream_id))
        assert len(history) == 1
        assert history[0].name == "func1"
        assert history[0].arguments == "test"

    async def test_set_history(self, redis_storage, stream_id):
        storage = redis_storage()

        events = [
            ToolCallEvent(name="func1", arguments="a"),
            ToolCallEvent(name="func2", arguments="b"),
        ]
        await storage.set_history(stream_id, events)

        history = list(await storage.get_history(stream_id))
        assert len(history) == 2
        assert history[0].name == "func1"
        assert history[1].name == "func2"

    async def test_set_history_replaces(self, redis_storage, stream_id):
        storage = redis_storage()

        await storage.set_history(stream_id, [ToolCallEvent(name="old", arguments="x")])
        await storage.set_history(stream_id, [ToolCallEvent(name="new", arguments="y")])

        history = list(await storage.get_history(stream_id))
        assert len(history) == 1
        assert history[0].name == "new"

    async def test_drop_history(self, redis_storage, stream_id):
        storage = redis_storage()

        await storage.set_history(stream_id, [ToolCallEvent(name="f", arguments="a")])
        await storage.drop_history(stream_id)

        history = list(await storage.get_history(stream_id))
        assert len(history) == 0

    async def test_empty_history(self, redis_storage, stream_id):
        storage = redis_storage()
        history = list(await storage.get_history(stream_id))
        assert history == []


class TestRedisStream:
    async def test_send_event_persists(self, redis_stream):
        stream = redis_stream()
        event = ToolCallEvent(name="func1", arguments="test")
        await stream.send(event, context=Context(stream))

        history = list(await stream.history.get_events())
        assert len(history) == 1
        assert history[0].name == "func1"

    async def test_send_notifies_subscribers(self, redis_stream):
        stream = redis_stream()
        received = []

        stream.subscribe(lambda ev: received.append(ev))
        stream._ensure_listener()

        event = ToolCallEvent(name="func1", arguments="test")
        await stream.send(event, context=Context(stream))

        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].name == "func1"

    async def test_send_multiple_events(self, redis_stream):
        stream = redis_stream()
        ctx = Context(stream)

        event1 = ToolCallEvent(name="func1", arguments="a")
        event2 = ModelMessage("hello")

        await stream.send(event1, context=ctx)
        await stream.send(event2, context=ctx)

        history = list(await stream.history.get_events())
        assert len(history) == 2

    async def test_history_persists_across_instances(self, redis_stream):
        prefix = f"ag2:test:{uuid4()}"
        sid = uuid4()

        stream1 = redis_stream(prefix=prefix, id=sid)
        await stream1.send(ToolCallEvent(name="func1", arguments="a"), context=Context(stream1))

        # New stream instance, same id and prefix — shares the same fake redis
        stream2 = redis_stream(prefix=prefix, id=sid)
        history = list(await stream2.history.get_events())
        assert len(history) == 1
        assert history[0].name == "func1"

    async def test_no_duplicate_persistence(self, redis_stream):
        """Events should be persisted exactly once, not duplicated."""
        prefix = f"ag2:test:{uuid4()}"
        sid = uuid4()

        stream1 = redis_stream(prefix=prefix, id=sid)
        stream2 = redis_stream(prefix=prefix, id=sid)
        stream2._ensure_listener()
        await asyncio.sleep(0.1)

        await stream1.send(ToolCallEvent(name="func1", arguments="a"), context=Context(stream1))
        await asyncio.sleep(0.1)

        history = list(await stream1.history.get_events())
        assert len(history) == 1

    async def test_cross_instance_pubsub(self, redis_stream):
        """Events sent on one instance are received by subscribers on another."""
        prefix = f"ag2:test:{uuid4()}"
        sid = uuid4()

        stream1 = redis_stream(prefix=prefix, id=sid)
        stream2 = redis_stream(prefix=prefix, id=sid)

        received = []
        stream2.subscribe(lambda ev: received.append(ev))
        stream2._ensure_listener()
        await asyncio.sleep(0.1)

        event = ToolCallEvent(name="cross_process", arguments="hello")
        await stream1.send(event, context=Context(stream1))

        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0].name == "cross_process"

    async def test_bidirectional_pubsub(self, redis_stream):
        """Both instances can send and receive from each other."""
        prefix = f"ag2:test:{uuid4()}"
        sid = uuid4()

        stream1 = redis_stream(prefix=prefix, id=sid)
        stream2 = redis_stream(prefix=prefix, id=sid)

        received1 = []
        received2 = []
        stream1.subscribe(lambda ev: received1.append(ev))
        stream2.subscribe(lambda ev: received2.append(ev))
        stream1._ensure_listener()
        stream2._ensure_listener()
        await asyncio.sleep(0.1)

        await stream1.send(TextInput("from stream1"), context=Context(stream1))
        await asyncio.sleep(0.1)

        await stream2.send(ModelMessage("from stream2"), context=Context(stream2))
        await asyncio.sleep(0.1)

        # stream1 receives both (own + stream2's)
        assert len(received1) == 2
        # stream2 receives both (stream1's + own)
        assert len(received2) == 2

        history = list(await stream1.history.get_events())
        assert len(history) == 2

    async def test_where_filter_with_pubsub(self, redis_stream):
        """Filtered subscriptions work correctly with pub/sub delivery."""
        stream = redis_stream()
        tool_events = []

        stream.where(ToolCallEvent).subscribe(lambda ev: tool_events.append(ev))
        stream._ensure_listener()

        await stream.send(ToolCallEvent(name="func1", arguments="a"), context=Context(stream))
        await stream.send(ModelMessage("hello"), context=Context(stream))
        await asyncio.sleep(0.1)

        assert len(tool_events) == 1
        assert tool_events[0].name == "func1"

    async def test_multiple_subscribers_same_instance(self, redis_stream):
        """Multiple subscribers on the same stream all receive events."""
        stream = redis_stream()
        received_a = []
        received_b = []
        received_c = []

        stream.subscribe(lambda ev: received_a.append(ev))
        stream.subscribe(lambda ev: received_b.append(ev))
        stream.subscribe(lambda ev: received_c.append(ev))
        stream._ensure_listener()

        await stream.send(ToolCallEvent(name="func1", arguments="a"), context=Context(stream))
        await asyncio.sleep(0.1)

        assert len(received_a) == 1
        assert len(received_b) == 1
        assert len(received_c) == 1

    async def test_local_dispatch_survives_listener_failure(self, redis_stream):
        """Local subscribers keep receiving events after the pub/sub listener dies."""
        stream = redis_stream()
        received = []
        stream.subscribe(lambda ev: received.append(ev))
        stream._ensure_listener()
        await stream._listener_ready.wait()

        stream._listener_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stream._listener_task
        dead_task = stream._listener_task
        # Block auto-restart so the next send() must rely on local dispatch.
        stream._ensure_listener = lambda: None

        await stream.send(ToolCallEvent(name="post_crash", arguments=""), context=Context(stream))
        await asyncio.sleep(0.05)

        assert len(received) == 1
        assert received[0].name == "post_crash"
        assert stream._listener_task is dead_task and dead_task.done()
