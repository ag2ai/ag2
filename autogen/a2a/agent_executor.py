# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import InternalError, Task, TaskState, TaskStatus
from a2a.utils.errors import ServerError

from autogen import ConversableAgent
from autogen.agentchat.remote import AgentService, ServiceResponse
from autogen.doc_utils import export_module

from .utils import make_artifact, make_input_required_message, make_working_message, request_message_from_a2a


@export_module("autogen.a2a")
class AutogenAgentExecutor(AgentExecutor):
    """An agent executor that bridges Autogen ConversableAgents with A2A protocols.

    This class wraps an Autogen ConversableAgent to enable it to be executed within
    the A2A framework, handling message processing, task management, and event publishing.
    """

    def __init__(self, agent: ConversableAgent) -> None:
        self.agent = AgentService(agent)
        self.__agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        assert context.message

        task = context.current_task
        if not task:
            request = context.message
            # build task object manually to allow empty messages
            task = Task(
                status=TaskStatus(
                    state=TaskState.submitted,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                id=request.task_id or str(uuid4()),
                context_id=request.context_id or str(uuid4()),
                history=[request],
            )
            # publish the task status submitted event
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        final_message: ServiceResponse | None = None
        try:
            async for response in self.agent(request_message_from_a2a(context.message)):
                if response.input_required:
                    await updater.requires_input(
                        message=make_input_required_message(
                            context_id=task.context_id,
                            task_id=task.id,
                            text=response.input_required,
                            context=response.context,
                        ),
                        final=True,
                    )
                    return

                else:
                    await updater.update_status(
                        message=make_working_message(
                            message=response.message,
                            context_id=task.context_id,
                            task_id=task.id,
                        ),
                        state=TaskState.working,
                    )

                final_message = response

        except Exception as e:
            raise ServerError(error=InternalError()) from e

        if final_message:
            artifact = make_artifact(
                message=final_message.message,
                context=final_message.context,
                cost=self.__agent.get_actual_usage(),
            )

            await updater.add_artifact(
                artifact_id=artifact.artifact_id,
                name=artifact.name,
                parts=artifact.parts,
                metadata=artifact.metadata,
                extensions=artifact.extensions,
            )

            await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass
