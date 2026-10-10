# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import shutil
import subprocess
import time
from collections.abc import Iterator
from typing import Any

import pytest

redis = pytest.importorskip("redis")

from redis.asyncio import Redis

from ag2.knowledge.redis import RedisKnowledgeStore


@pytest.fixture(scope="module")
def redis_socket(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    executable = shutil.which("redis-server")
    if executable is None:
        pytest.skip("redis-server is required for the knowledge-store integration tests")
    socket = str(tmp_path_factory.mktemp("redis") / "server.sock")
    process = subprocess.Popen(
        [executable, "--port", "0", "--unixsocket", socket, "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with redis.Redis(unix_socket_path=socket) as client:
            deadline = time.monotonic() + 5
            while True:
                try:
                    client.ping()
                    break
                except redis.ConnectionError:
                    if process.poll() is not None or time.monotonic() >= deadline:
                        pytest.fail("the owned Redis server did not start")
                    time.sleep(0.01)
        yield socket
    finally:
        process.terminate()
        process.wait(timeout=5)


class ReadBarrier:
    def __init__(self) -> None:
        self.reads = 0
        self.ready = asyncio.Event()

    async def wait(self) -> None:
        self.reads += 1
        if self.reads == 2:
            self.ready.set()
        await self.ready.wait()


class CoordinatedRedis:
    """Keep two real GET replies in flight to exercise a lost-update interleaving."""

    def __init__(self, client: Any, barrier: ReadBarrier) -> None:
        self.client = client
        self.barrier = barrier

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    async def get(self, name: Any) -> Any:
        value = await self.client.get(name)
        if str(name).endswith("/log.jsonl"):
            await self.barrier.wait()
        return value


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", ["", "snow\u2603\n"])
async def test_independent_connections_append_without_losing_entries(redis_socket: str, initial: str) -> None:
    barrier = ReadBarrier()
    async with (
        Redis(unix_socket_path=redis_socket) as reader,
        Redis(unix_socket_path=redis_socket) as first_client,
        Redis(unix_socket_path=redis_socket) as second_client,
    ):
        await reader.flushdb()
        reader_store = RedisKnowledgeStore(reader)
        if initial:
            await reader_store.write("/log.jsonl", initial)
        first = RedisKnowledgeStore(CoordinatedRedis(first_client, barrier))
        second = RedisKnowledgeStore(CoordinatedRedis(second_client, barrier))
        payloads = ["first\n", "\u96ea\n"]
        offsets = await asyncio.wait_for(
            asyncio.gather(first.append("log.jsonl", payloads[0]), second.append("/log.jsonl", payloads[1])),
            timeout=5,
        )
        content = await reader_store.read("/log.jsonl")
        assert content is not None
        data = content.encode("utf-8")
        assert len(data) == len(initial.encode("utf-8")) + sum(len(p.encode("utf-8")) for p in payloads)
        assert data.startswith(initial.encode("utf-8"))
        for offset, payload in zip(offsets, payloads):
            assert await reader_store.read_range("/log.jsonl", offset, offset + len(payload.encode("utf-8"))) == payload
        assert offsets[0] != offsets[1]
        assert await reader_store.list("/") == ["log.jsonl"]
        assert (await reader_store.list_versions_under("/"))["/log.jsonl"] == 2 + bool(initial)


@pytest.mark.asyncio
async def test_append_keeps_empty_and_utf8_byte_offset_semantics(redis_socket: str) -> None:
    async with Redis(unix_socket_path=redis_socket) as client:
        await client.flushdb()
        store = RedisKnowledgeStore(client)
        assert await store.append("/log.jsonl", "") == 0
        assert await store.exists("/log.jsonl")
        assert await store.append("/log.jsonl", "\u96ea") == 0
        assert await store.append("/log.jsonl", "snow\u2603") == len("\u96ea".encode("utf-8"))
        assert await store.read("/log.jsonl") == "\u96easnow\u2603"
        assert (await store.list_versions_under("/"))["/log.jsonl"] == 3
        await store.delete("/log.jsonl")
        assert not await store.exists("/log.jsonl")
