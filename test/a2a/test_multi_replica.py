# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from a2a.server.agent_execution import AgentExecutor as A2AAgentExecutorBase
from a2a.server.cluster import DatabaseTaskEventStream, VersionedDatabaseTaskStore
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import TaskState
from dirty_equals import IsPartialDict

from ag2 import Agent
from ag2.a2a import A2AConfig, A2AServer
from ag2.a2a.tasks import get_task, list_tasks
from ag2.a2a.testing import make_test_client_factory, make_test_rest_client_factory
from ag2.events import ToolCallEvent, ToolResultEvent, ToolResultsEvent
from ag2.testing import TestConfig, TrackingConfig

from ._helpers import GatedExecutor, StatelessScript, Switchboard

pytest.importorskip("aiosqlite")

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

URL = "http://test"

Replica = Callable[..., Awaitable[A2AServer]]


@pytest_asyncio.fixture
async def replica(tmp_path: Path) -> AsyncGenerator[Replica]:
    """Build servers that share one SQLite file, one engine apiece — like separate processes."""
    engines: list[AsyncEngine] = []

    async def build(agent: Agent | None = None, executor: A2AAgentExecutorBase | None = None) -> A2AServer:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'a2a.db'}")
        engines.append(engine)
        store, stream = VersionedDatabaseTaskStore(engine), DatabaseTaskEventStream(engine, poll_interval_s=0.01)
        # The SDK creates only the `tasks` table lazily; the version and event tables need this call.
        await store.initialize()
        await stream.initialize()
        return A2AServer(
            agent or Agent("server", config=TestConfig("hi")),
            task_store=store,
            event_stream=stream,
            executor=executor,
        )

    yield build

    for engine in engines:
        await engine.dispose()


def _client_config(server: A2AServer) -> A2AConfig:
    return A2AConfig(card_url=URL, httpx_client_factory=make_test_client_factory(server, url=URL), streaming=False)


@pytest.mark.asyncio
async def test_a_task_created_on_one_replica_is_visible_on_another(replica: Replica) -> None:
    first, second = _client_config(await replica()), _client_config(await replica())

    await Agent("client", config=first).ask("ping")

    [created] = (await list_tasks(first)).tasks
    assert (await get_task(second, created.id)).id == created.id
    assert [task.id for task in (await list_tasks(second)).tasks] == [created.id]


@pytest.mark.parametrize(
    ("prefer", "make_factory"),
    [("jsonrpc", make_test_client_factory), ("rest", make_test_rest_client_factory)],
)
@pytest.mark.asyncio
async def test_every_http_transport_serves_the_shared_store(
    replica: Replica, prefer: str, make_factory: Callable[..., Callable[[], httpx.AsyncClient]]
) -> None:
    # Wiring is per-transport, so one transport working proves nothing about the others.
    first, second = [
        A2AConfig(
            card_url=URL, httpx_client_factory=make_factory(await replica(), url=URL), streaming=False, prefer=prefer
        )
        for _ in range(2)
    ]

    await Agent("client", config=first).ask("ping")

    [created] = (await list_tasks(first)).tasks
    assert (await get_task(second, created.id)).id == created.id


@pytest.mark.asyncio
async def test_a_versioned_store_without_an_event_stream_is_refused(replica: Replica) -> None:
    donor = await replica()

    with pytest.raises(ValueError, match="event_stream"):
        A2AServer(Agent("a", config=TestConfig("hi")), task_store=donor.task_store)


@pytest.mark.asyncio
async def test_an_event_stream_without_a_versioned_store_is_refused(replica: Replica) -> None:
    donor = await replica()

    with pytest.raises(ValueError, match="versioned"):
        A2AServer(Agent("a", config=TestConfig("hi")), task_store=InMemoryTaskStore(), event_stream=donor.event_stream)

    with pytest.raises(ValueError, match="versioned"):
        A2AServer(Agent("a", config=TestConfig("hi")), event_stream=donor.event_stream)


@pytest.mark.asyncio
async def test_the_unclustered_configurations_still_construct(replica: Replica) -> None:
    clustered = await replica()

    A2AServer(Agent("a", config=TestConfig("hi")))
    A2AServer(Agent("a", config=TestConfig("hi")), task_store=InMemoryTaskStore())

    assert clustered.event_stream is not None
    assert A2AServer(Agent("a", config=TestConfig("hi"))).event_stream is None


@pytest.mark.asyncio
async def test_a_task_waiting_on_the_client_resumes_on_another_replica(replica: Replica) -> None:
    tool_call = ToolCallEvent(name="get_weather", arguments='{"city": "Paris"}')
    tracking = [TrackingConfig(StatelessScript(tool_call, after_tool="Weather report ready")) for _ in range(2)]
    first = await replica(Agent("server", config=tracking[0]))
    second = await replica(Agent("server", config=tracking[1]))
    board = Switchboard(first, second, url=URL)

    def get_weather(city: str) -> str:
        # Runs on the client between the two requests: the next one reaches the other replica.
        board.active = 1
        return f"It is sunny in {city}"

    client = Agent(
        "client",
        config=A2AConfig(
            card_url=URL,
            httpx_client_factory=lambda: httpx.AsyncClient(transport=board, base_url=URL),
            streaming=False,
        ),
        tools=[get_weather],
    )

    reply = await client.ask("how is paris?")

    assert reply.response.content == "Weather report ready"
    assert tracking[0].mock.call_count == 1
    tracking[1].mock.assert_called_with(
        ToolResultsEvent([ToolResultEvent.from_call(tool_call, "It is sunny in Paris")])
    )
    [task] = (await list_tasks(_client_config(first))).tasks
    assert task.status.state == TaskState.TASK_STATE_COMPLETED


async def _subscribe_texts(server: A2AServer, task_id: str) -> list[str]:
    """Raw JSON-RPC ``SubscribeToTask``: the wire contract a client on that replica would use."""
    body = {"jsonrpc": "2.0", "id": 1, "method": "SubscribeToTask", "params": {"id": task_id}}
    transport = httpx.ASGITransport(app=server.build_jsonrpc(url=URL))
    events: list[str] = []
    async with (
        httpx.AsyncClient(transport=transport, base_url=URL) as http,
        http.stream("POST", "/", json=body, headers={"A2A-Version": "1.0"}) as response,
    ):
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                events.append(line)
    return events


@pytest.mark.asyncio
async def test_a_subscriber_on_another_replica_sees_a_running_task_finish(replica: Replica) -> None:
    executor = GatedExecutor()
    first, second = await replica(executor=executor), await replica(executor=GatedExecutor())
    running = asyncio.create_task(Agent("client", config=_client_config(first)).ask("work"))

    task_id = await _wait_for_working_task(first)
    subscriber = asyncio.create_task(_subscribe_texts(second, task_id))
    executor.gate.set()

    events = await asyncio.wait_for(subscriber, timeout=10)
    await running
    assert json.loads(events[-1].removeprefix("data:"))["result"] == IsPartialDict({
        "statusUpdate": IsPartialDict({"status": IsPartialDict({"state": "TASK_STATE_COMPLETED"})})
    })


async def _wait_for_working_task(server: A2AServer) -> str:
    config = _client_config(server)
    async with asyncio.timeout(10):
        while True:
            for task in (await list_tasks(config)).tasks:
                if task.status.state == TaskState.TASK_STATE_WORKING:
                    return task.id
            await asyncio.sleep(0.01)
