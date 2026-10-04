# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from a2a.server.agent_execution import AgentExecutor as A2AAgentExecutorBase
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
from ._http_server import serve_over_http

pytest.importorskip("aiosqlite")

from a2a.server.cluster import DatabaseTaskEventStream, VersionedDatabaseTaskStore
from sqlalchemy.ext.asyncio import create_async_engine

URL = "http://test"


@asynccontextmanager
async def _replica(
    tmp_path: Path, agent: Agent | None = None, executor: A2AAgentExecutorBase | None = None
) -> AsyncGenerator[A2AServer]:
    """A server on the SQLite file in ``tmp_path`` with an engine of its own — like a separate process."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'a2a.db'}")
    try:
        store, stream = VersionedDatabaseTaskStore(engine), DatabaseTaskEventStream(engine, poll_interval_s=0.01)
        # The SDK creates only the `tasks` table lazily; the version and event tables need this call.
        await store.initialize()
        await stream.initialize()
        yield A2AServer(
            agent or Agent("server", config=TestConfig("hi")),
            task_store=store,
            event_stream=stream,
            executor=executor,
        )
    finally:
        await engine.dispose()


def _client_config(server: A2AServer) -> A2AConfig:
    return A2AConfig(card_url=URL, httpx_client_factory=make_test_client_factory(server, url=URL), streaming=False)


@pytest.mark.asyncio
async def test_a_task_created_on_one_replica_is_visible_on_another(tmp_path: Path) -> None:
    async with _replica(tmp_path) as one, _replica(tmp_path) as two:
        first, second = _client_config(one), _client_config(two)

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
    tmp_path: Path, prefer: str, make_factory: Callable[..., Callable[[], httpx.AsyncClient]]
) -> None:
    # Wiring is per-transport, so one transport working proves nothing about the others.
    async with _replica(tmp_path) as one, _replica(tmp_path) as two:
        first, second = [
            A2AConfig(card_url=URL, httpx_client_factory=make_factory(server, url=URL), streaming=False, prefer=prefer)
            for server in (one, two)
        ]

        await Agent("client", config=first).ask("ping")

        [created] = (await list_tasks(first)).tasks
        assert (await get_task(second, created.id)).id == created.id


@pytest.mark.asyncio
async def test_a_versioned_store_without_an_event_stream_is_refused(tmp_path: Path) -> None:
    async with _replica(tmp_path) as donor:
        with pytest.raises(ValueError, match="event_stream"):
            A2AServer(Agent("a", config=TestConfig("hi")), task_store=donor.task_store)


@pytest.mark.asyncio
async def test_an_event_stream_without_a_versioned_store_is_refused(tmp_path: Path) -> None:
    async with _replica(tmp_path) as donor:
        with pytest.raises(ValueError, match="versioned"):
            A2AServer(
                Agent("a", config=TestConfig("hi")), task_store=InMemoryTaskStore(), event_stream=donor.event_stream
            )

        with pytest.raises(ValueError, match="versioned"):
            A2AServer(Agent("a", config=TestConfig("hi")), event_stream=donor.event_stream)


@pytest.mark.asyncio
async def test_the_unclustered_configurations_still_construct(tmp_path: Path) -> None:
    default = A2AServer(Agent("a", config=TestConfig("hi")))
    in_memory = A2AServer(Agent("a", config=TestConfig("hi")), task_store=InMemoryTaskStore())

    async with _replica(tmp_path) as clustered:
        assert clustered.event_stream is not None
    assert default.event_stream is None
    assert in_memory.event_stream is None


@pytest.mark.asyncio
async def test_a_task_waiting_on_the_client_resumes_on_another_replica(tmp_path: Path) -> None:
    tool_call = ToolCallEvent(name="get_weather", arguments='{"city": "Paris"}')
    tracking = [TrackingConfig(StatelessScript(tool_call, after_tool="Weather report ready")) for _ in range(2)]

    async with (
        _replica(tmp_path, Agent("server", config=tracking[0])) as first,
        _replica(tmp_path, Agent("server", config=tracking[1])) as second,
    ):
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


@pytest.mark.asyncio
async def test_a_subscriber_on_another_replica_sees_a_running_task_finish(tmp_path: Path) -> None:
    executor = GatedExecutor()
    async with (
        _replica(tmp_path, executor=executor) as first,
        _replica(tmp_path, executor=GatedExecutor()) as second,
    ):
        running = asyncio.create_task(Agent("client", config=_client_config(first)).ask("work"))
        await executor.working.wait()

        # A real socket, because an in-process transport hands the body over only once it is complete.
        body = {"jsonrpc": "2.0", "id": 1, "method": "SubscribeToTask", "params": {"id": executor.task_id}}
        events: list[str] = []
        async with (
            serve_over_http(lambda url: second.build_jsonrpc(url=url)) as url,
            httpx.AsyncClient(base_url=url) as http,
            http.stream("POST", "/", json=body, headers={"A2A-Version": "1.0"}) as response,
        ):
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    events.append(line)
                    # The first event proves the subscription is live; only now may the task finish.
                    executor.gate.set()

        await running
        assert json.loads(events[-1].removeprefix("data:"))["result"] == IsPartialDict({
            "statusUpdate": IsPartialDict({"status": IsPartialDict({"state": "TASK_STATE_COMPLETED"})})
        })
