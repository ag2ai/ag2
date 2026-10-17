# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
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
    A2ATaskTerminalError,
)

from ._helpers import make_executor_pair

_Finish = Callable[[TaskUpdater], Awaitable[None]]


@dataclass(slots=True)
class _TerminalExecutor(A2AAgentExecutorBase):
    """Drives a fresh task straight to one terminal state.

    ``finish`` is the ``TaskUpdater`` transition that ends the task — the
    only thing that differs between the states under test, so the rest of
    the choreography stays in one place.
    """

    finish: _Finish

    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        msg = request_context.message
        assert msg is not None
        task_id = msg.task_id or uuid4().hex
        context_id = msg.context_id or uuid4().hex
        await event_queue.enqueue_event(
            Task(id=task_id, context_id=context_id, status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED)),
        )
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()
        await self.finish(updater)

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False], ids=["streaming", "polling"])
@pytest.mark.parametrize(
    ("finish", "error", "state"),
    [
        pytest.param(
            lambda updater: updater.failed(),
            A2ATaskFailedError,
            TaskState.TASK_STATE_FAILED,
            id="failed",
        ),
        pytest.param(
            lambda updater: updater.reject(),
            A2ATaskRejectedError,
            TaskState.TASK_STATE_REJECTED,
            id="rejected",
        ),
        pytest.param(
            lambda updater: updater.requires_auth(),
            A2ATaskAuthRequiredError,
            TaskState.TASK_STATE_AUTH_REQUIRED,
            id="auth-required",
        ),
    ],
)
async def test_terminal_state_raises_and_carries_the_task(
    finish: _Finish,
    error: type[A2ATaskTerminalError],
    state: int,
    streaming: bool,
) -> None:
    pair = make_executor_pair(_TerminalExecutor(finish), streaming=streaming)

    with pytest.raises(error) as exc_info:
        await pair.client.ask("ping")

    # The error carries the terminal Task, so callers can inspect why it ended.
    assert exc_info.value.task.status.state == state
