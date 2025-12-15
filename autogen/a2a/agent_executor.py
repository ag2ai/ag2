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
from autogen.doc_utils import export_module
from autogen.remote.agent_service import AgentService

from .utils import request_message_from_a2a, response_message_to_a2a


@export_module("autogen.a2a")
class AutogenAgentExecutor(AgentExecutor):
    """An agent executor that bridges Autogen ConversableAgents with A2A protocols.

    This class wraps an Autogen ConversableAgent to enable it to be executed within
    the A2A framework, handling message processing, task management, and event publishing.
    """

    def __init__(self, agent: ConversableAgent) -> None:
        self.agent = AgentService(agent)

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

        try:
            result = await self.agent(request_message_from_a2a(context.message))

        except Exception as e:
            raise ServerError(error=InternalError()) from e

        artifact, messages, input_required_msg = response_message_to_a2a(result, context.context_id, task.id)

        # publish local chat history events
        for message in messages:
            await updater.update_status(
                state=TaskState.working,
                message=message,
            )

        # publish input required event
        if input_required_msg:
            await updater.requires_input(message=input_required_msg, final=True)
            return

        # publish the task final result event
        if artifact:
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
