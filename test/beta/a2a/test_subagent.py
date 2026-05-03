# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest
from a2a.server.context import ServerCallContext
from a2a.server.tasks import InMemoryTaskStore, TaskStore
from a2a.types import ListTasksRequest, ListTasksResponse, Task

from autogen.beta import Agent
from autogen.beta.a2a import A2AConfig, A2AServer
from autogen.beta.events import ToolCallEvent
from autogen.beta.testing import TestConfig


class _RecordingTaskStore(TaskStore):
    def __init__(self) -> None:
        self._inner = InMemoryTaskStore()
        self.saved: list[Task] = []

    async def save(self, task: Task, context: ServerCallContext) -> None:
        self.saved.append(task)
        await self._inner.save(task, context)

    async def get(self, task_id: str, context: ServerCallContext) -> Task | None:
        return await self._inner.get(task_id, context)

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        await self._inner.delete(task_id, context)

    async def list(self, params: ListTasksRequest, context: ServerCallContext) -> ListTasksResponse:
        return await self._inner.list(params, context)


@pytest.mark.asyncio
async def test_subagent_tool_shares_context_id_across_calls() -> None:
    store = _RecordingTaskStore()
    remote_agent = Agent("remote", "p", config=TestConfig("r1", "r2", "r3"))
    server = A2AServer(remote_agent, url="http://test", task_store=store)
    transport = httpx.ASGITransport(app=server.build_asgi())

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        remote = Agent("remote", config=A2AConfig("http://test", client_factory=lambda: http))
        orchestrator = Agent(
            "orchestrator",
            "p",
            config=TestConfig(
                [
                    ToolCallEvent(name="task_remote", arguments='{"objective": "first"}'),
                    ToolCallEvent(name="task_remote", arguments='{"objective": "second"}'),
                ],
                ToolCallEvent(name="task_remote", arguments='{"objective": "third"}'),
                "done",
            ),
            tools=[remote.as_tool(description="ask remote")],
        )

        reply = await orchestrator.ask("go")

    assert reply.body == "done"

    persisted = {task.id: task for task in store.saved}
    assert len(persisted) == 3
    assert len({task.context_id for task in persisted.values()}) == 1
