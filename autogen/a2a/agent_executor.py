# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import TaskArtifactUpdateEvent, TaskState, TaskStatus, TaskStatusUpdateEvent
from a2a.utils import new_task
from a2a.utils.message import new_agent_text_message

from autogen import ConversableAgent
from autogen.doc_utils import export_module
from autogen.remote.agent_service import AgentService

from .utils import request_message_from_a2a, response_message_to_a2a


@export_module("autogen.a2a")
class AutogenAgentExecutor(AgentExecutor):
    def __init__(self, agent: ConversableAgent) -> None:
        self.agent = AgentService(agent)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        assert context.message

        task = context.current_task
        if not task:
            task = new_task(context.message)
            task.status.timestamp = datetime.now(timezone.utc).isoformat()
            # publish the task status submitted event
            await event_queue.enqueue_event(task)

        try:
            result = await self.agent(request_message_from_a2a(context.message))

        except Exception as e:
            # publish the task status failed event
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task.id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=new_agent_text_message(
                            str(e),
                            task_id=task.id,
                            context_id=context.context_id,
                        ),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                    context_id=context.context_id,
                    final=True,
                )
            )
            return

        artifact, messages = response_message_to_a2a(result, context.context_id, task.id)

        # publish local chat history events
        for message in messages:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task.id,
                    status=TaskStatus(
                        state=TaskState.working,
                        message=message,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    ),
                    context_id=context.context_id,
                    final=False,
                )
            )

        # publish the task result event
        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id=task.id,
                last_chunk=True,
                context_id=context.context_id,
                artifact=artifact,
            )
        )

        # publish the task status completed event
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task.id,
                status=TaskStatus(
                    state=TaskState.completed,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                context_id=context.context_id,
                final=True,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass
