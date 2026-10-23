# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

redis = pytest.importorskip("redis")

from redis.asyncio import Redis

from ag2.knowledge.redis import RedisKnowledgeStore

pytestmark = [
    pytest.mark.redis,
    pytest.mark.asyncio,
]

_CONNECTIONS = 20


@pytest.fixture(scope="module")
def redis_socket() -> Iterator[str]:
    executable = shutil.which("redis-server")
    if executable is None:
        pytest.skip("redis-server is required for the knowledge-store integration tests")
    # A short directory: unix socket paths are limited to ~100 characters on macOS.
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="ag2-redis-") as directory:
        socket = str(Path(directory) / "server.sock")
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
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.mark.parametrize("initial", ["", "snow☃\n"])
async def test_independent_connections_append_without_losing_entries(redis_socket: str, initial: str) -> None:
    clients = [Redis(unix_socket_path=redis_socket) for _ in range(_CONNECTIONS)]
    try:
        await clients[0].flushdb()
        stores = [RedisKnowledgeStore(client) for client in clients]
        if initial:
            await stores[0].write("/log.jsonl", initial)
        payloads = [f"entry-{i}-雪\n" for i in range(_CONNECTIONS)]

        offsets = await asyncio.gather(*(store.append("log.jsonl", text) for store, text in zip(stores, payloads)))

        for offset, text in zip(offsets, payloads):
            assert await stores[0].read_range("/log.jsonl", offset, offset + len(text.encode("utf-8"))) == text
        content = await stores[0].read("/log.jsonl")
        assert content is not None
        assert content.startswith(initial)
        assert sorted(content[len(initial) :].splitlines(keepends=True)) == sorted(payloads)
        assert (await stores[0].list_versions_under("/"))["/log.jsonl"] == _CONNECTIONS + bool(initial)
    finally:
        for client in clients:
            await client.aclose()


async def test_append_keeps_empty_and_utf8_byte_offset_semantics(redis_socket: str) -> None:
    async with Redis(unix_socket_path=redis_socket) as client:
        await client.flushdb()
        store = RedisKnowledgeStore(client)
        assert await store.append("/log.jsonl", "") == 0
        assert await store.exists("/log.jsonl")
        assert await store.append("/log.jsonl", "雪") == 0
        assert await store.append("/log.jsonl", "snow☃") == len("雪".encode())
        assert await store.read("/log.jsonl") == "雪snow☃"
        assert (await store.list_versions_under("/"))["/log.jsonl"] == 3
        await store.delete("/log.jsonl")
        assert not await store.exists("/log.jsonl")
