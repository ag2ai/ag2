# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest
from a2a.server.agent_execution import AgentExecutor as A2AAgentExecutorBase
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Task, TaskState, TaskStatus

from ag2.a2a.errors import (
    A2ATaskAuthRequiredError,
    A2ATaskFailedError,
    A2ATaskRejectedError,
)

from ._helpers import make_executor_pair


def _bootstrap_task(request_context: RequestContext, event_queue: EventQueue) -> tuple[str, str, TaskUpdater]:
    msg = request_context.message
    assert msg is not None
    task_id = msg.task_id or uuid4().hex
    context_id = msg.context_id or uuid4().hex
    return task_id, context_id, TaskUpdater(event_queue, task_id, context_id)


class _FailedExecutor(A2AAgentExecutorBase):
    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        task_id, context_id, updater = _bootstrap_task(request_context, event_queue)
        await event_queue.enqueue_event(
            Task(id=task_id, context_id=context_id, status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED)),
        )
        await updater.start_work()
        await updater.failed()

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        return None


class _RejectedExecutor(A2AAgentExecutorBase):
    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        task_id, context_id, updater = _bootstrap_task(request_context, event_queue)
        await event_queue.enqueue_event(
            Task(id=task_id, context_id=context_id, status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED)),
        )
        await updater.start_work()
        await updater.reject()

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        return None


class _AuthRequiredExecutor(A2AAgentExecutorBase):
    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        task_id, context_id, updater = _bootstrap_task(request_context, event_queue)
        await event_queue.enqueue_event(
            Task(id=task_id, context_id=context_id, status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED)),
        )
        await updater.start_work()
        await updater.requires_auth()

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        return None


@pytest.mark.asyncio
class TestTerminalStates:
    async def test_failed_task_raises_failed_error_streaming(self) -> None:
        pair = make_executor_pair(_FailedExecutor(), streaming=True)
        with pytest.raises(A2ATaskFailedError):
            await pair.client.ask("ping")

    async def test_failed_task_raises_failed_error_polling(self) -> None:
        pair = make_executor_pair(_FailedExecutor(), streaming=False)
        with pytest.raises(A2ATaskFailedError):
            await pair.client.ask("ping")

    async def test_rejected_task_raises_rejected_error_streaming(self) -> None:
        pair = make_executor_pair(_RejectedExecutor(), streaming=True)
        with pytest.raises(A2ATaskRejectedError):
            await pair.client.ask("ping")

    async def test_rejected_task_raises_rejected_error_polling(self) -> None:
        pair = make_executor_pair(_RejectedExecutor(), streaming=False)
        with pytest.raises(A2ATaskRejectedError):
            await pair.client.ask("ping")

    async def test_auth_required_raises_auth_error_streaming(self) -> None:
        pair = make_executor_pair(_AuthRequiredExecutor(), streaming=True)
        with pytest.raises(A2ATaskAuthRequiredError):
            await pair.client.ask("ping")

    async def test_auth_required_raises_auth_error_polling(self) -> None:
        pair = make_executor_pair(_AuthRequiredExecutor(), streaming=False)
        with pytest.raises(A2ATaskAuthRequiredError):
            await pair.client.ask("ping")
