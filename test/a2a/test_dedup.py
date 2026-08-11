# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest
from a2a.server.agent_execution import AgentExecutor as A2AAgentExecutorBase
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, Task, TaskState, TaskStatus

from ._helpers import make_executor_pair


class _MultiTurnTextExecutor(A2AAgentExecutorBase):
    async def execute(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        msg = request_context.message
        assert msg is not None
        task_id = msg.task_id or uuid4().hex
        context_id = msg.context_id or uuid4().hex
        updater = TaskUpdater(event_queue, task_id, context_id)

        if request_context.current_task is None:
            await event_queue.enqueue_event(
                Task(id=task_id, context_id=context_id, status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED)),
            )
            await updater.start_work()
            await updater.add_artifact(parts=[Part(text="partial-")], artifact_id="art-1", last_chunk=True)
            await updater.requires_input(message=updater.new_agent_message(parts=[Part(text="continue?")]))
            return

        user_reply = "".join(p.text for p in msg.parts if p.text)
        await updater.complete(message=updater.new_agent_message(parts=[Part(text=f"final-{user_reply}")]))

    async def cancel(self, request_context: RequestContext, event_queue: EventQueue) -> None:
        return None


@pytest.mark.asyncio
async def test_text_accumulates_across_input_required_polling() -> None:
    async def hitl() -> str:
        return "yes"

    pair = make_executor_pair(_MultiTurnTextExecutor(), streaming=False, hitl_hook=hitl)

    reply = await pair.client.ask("start")

    assert reply.response.content == "partial-final-yes"


@pytest.mark.asyncio
async def test_text_accumulates_across_input_required_streaming() -> None:
    async def hitl() -> str:
        return "yes"

    pair = make_executor_pair(_MultiTurnTextExecutor(), streaming=True, hitl_hook=hitl)

    reply = await pair.client.ask("start")

    assert reply.response.content == "partial-final-yes"
