# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    DataPart,
    Part,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_parts_message, new_task
from a2a.utils.message import get_data_parts

from autogen import ConversableAgent
from autogen.doc_utils import export_module
from autogen.remote.agent_service import AgentService
from autogen.remote.protocol import RequestMessage


@export_module("autogen.remote.a2a")
class AutogenAgentExecutor(AgentExecutor):
    def __init__(self, agent: ConversableAgent) -> None:
        self.agent = AgentService(agent)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        assert context.message
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        message = RequestMessage.model_validate(get_data_parts(context.message.parts)[0])
        result = await self.agent(message)

        if result:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    status=TaskStatus(
                        state=TaskState.completed,
                        message=new_agent_parts_message(
                            parts=[Part(root=DataPart(data=result.model_dump()))],
                            task_id=task.id,
                            context_id=task.context_id,
                        ),
                    ),
                    final=True,
                    context_id=task.context_id,
                    task_id=task.id,
                )
            )

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                status=TaskStatus(
                    state=TaskState.completed,
                    message=new_agent_parts_message(
                        [],
                        task_id=task.id,
                        context_id=task.context_id,
                    ),
                ),
                final=True,
                context_id=task.context_id,
                task_id=task.id,
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise Exception("cancel not supported")
