# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import httpx
import pytest
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import Task, TaskState, TaskStatus

from autogen.beta import Agent
from autogen.beta.a2a import A2AConfig, A2AServer
from autogen.beta.a2a.errors import (
    A2AAuthRequiredError,
    A2ATaskFailedError,
    A2ATaskRejectedError,
)
from autogen.beta.events import ToolCallEvent
from autogen.beta.testing import TestConfig


def _bootstrap_task(context: RequestContext) -> Task:
    msg = context.message
    assert msg is not None
    return context.current_task or Task(
        id=msg.task_id or uuid4().hex,
        context_id=msg.context_id or uuid4().hex,
        status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
        history=[msg],
    )


class _StatusExecutor(AgentExecutor):
    def __init__(self, action: str) -> None:
        self._action = action

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = _bootstrap_task(context)
        if context.current_task is None:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await getattr(updater, self._action)()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError()


@pytest.mark.asyncio
async def test_failing_tool_raises_a2a_task_failed(serve) -> None:
    server_agent = Agent("specialist", "p", config=TestConfig(ToolCallEvent(name="boom", arguments="{}")))

    @server_agent.tool
    def boom() -> str:
        raise RuntimeError("kaboom")

    config = serve(server_agent)

    with pytest.raises(A2ATaskFailedError) as exc_info:
        await Agent("client", "p", config=config).ask("hi")

    assert exc_info.value.task.status.state == TaskState.TASK_STATE_FAILED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "error_cls", "expected_state"),
    [
        ("reject", A2ATaskRejectedError, TaskState.TASK_STATE_REJECTED),
        ("requires_auth", A2AAuthRequiredError, TaskState.TASK_STATE_AUTH_REQUIRED),
    ],
)
async def test_executor_terminal_state_raises(
    action: str,
    error_cls: type[Exception],
    expected_state: TaskState,
) -> None:
    agent = Agent("specialist", "p", config=TestConfig("never-called"))
    server = A2AServer(agent, url="http://test")
    handler = DefaultRequestHandler(
        agent_executor=_StatusExecutor(action),
        task_store=InMemoryTaskStore(),
        agent_card=server.card,
    )
    asgi = server.build_asgi(request_handler=handler)
    transport = httpx.ASGITransport(app=asgi)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        client_agent = Agent("client", "p", config=A2AConfig("http://test", client_factory=lambda: http))

        with pytest.raises(error_cls) as exc_info:
            await client_agent.ask("hi")

    assert exc_info.value.task.status.state == expected_state  # type: ignore[attr-defined]
