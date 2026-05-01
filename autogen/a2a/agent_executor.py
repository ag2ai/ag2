# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from uuid import uuid4

from a2a.compat.v0_3.conversions import (
    to_compat_message,
    to_core_message,
    to_core_part,
    to_core_task,
)
from a2a.compat.v0_3.types import Task, TaskState, TaskStatus
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState as ProtoTaskState

from autogen import ConversableAgent
from autogen.agentchat.remote import AgentService
from autogen.doc_utils import export_module

from .utils import (
    copy_artifact,
    make_artifact,
    make_input_required_message,
    request_message_from_a2a,
)


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

        # ``context.message`` arrives as the proto type from a2a-sdk 1.0; our
        # internal mappers (utils.py) operate on the Pydantic compat types.
        compat_message = to_compat_message(context.message)

        task = context.current_task
        if task is None:
            # Build the initial task in compat-types so we can reuse our mappers,
            # then convert to proto for the EventQueue.
            compat_task = Task(
                status=TaskStatus(
                    state=TaskState.submitted,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ),
                id=compat_message.task_id or str(uuid4()),
                context_id=compat_message.context_id or str(uuid4()),
                history=[compat_message],
            )
            await event_queue.enqueue_event(to_core_task(compat_task))
            task_id = compat_task.id
            context_id = compat_task.context_id
        else:
            task_id = task.id
            context_id = task.context_id

        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.update_status(state=ProtoTaskState.TASK_STATE_WORKING)

        artifact = make_artifact(message=None)

        streaming_started = False
        async for response in self.agent(request_message_from_a2a(compat_message)):
            if response.input_required:
                input_msg = make_input_required_message(
                    context_id=context_id,
                    task_id=task_id,
                    text=response.input_required,
                    context=response.context,
                )
                await updater.requires_input(message=to_core_message(input_msg))
                return

            if response.streaming_text:
                artifact = copy_artifact(
                    artifact=artifact,
                    message={"content": response.streaming_text},
                    context=response.context,
                )

                await updater.add_artifact(
                    parts=[to_core_part(p) for p in artifact.parts],
                    artifact_id=artifact.artifact_id,
                    name=artifact.name,
                    append=streaming_started,
                    last_chunk=False,
                )

                streaming_started = True

            elif response.message:
                artifact = copy_artifact(
                    artifact=artifact,
                    message=response.message,
                    context=response.context,
                )

        await updater.add_artifact(
            artifact_id=artifact.artifact_id,
            name=artifact.name,
            parts=[to_core_part(p) for p in artifact.parts],
            metadata=artifact.metadata,
            extensions=artifact.extensions,
            append=streaming_started,
            last_chunk=True,
        )

        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass
