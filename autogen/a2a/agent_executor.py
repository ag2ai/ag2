# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from uuid import uuid4

from a2a.compat.v0_3 import conversions as _v03_conversions
from a2a.compat.v0_3.types import Task, TaskState, TaskStatus
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.utils.errors import InternalError

from autogen import ConversableAgent
from autogen.agentchat.remote import AgentService

from .utils import (
    copy_artifact,
    make_artifact,
    make_input_required_message,
    request_message_from_a2a,
    to_core_message,
    to_core_parts,
)


class AutogenAgentExecutor(AgentExecutor):
    """An agent executor that bridges Autogen ConversableAgents with A2A protocols.

    This class wraps an Autogen ConversableAgent to enable it to be executed within
    the A2A framework, handling message processing, task management, and event publishing.
    """

    def __init__(self, agent: ConversableAgent) -> None:
        self.agent = AgentService(agent)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        assert context.message
        # The 1.0 SDK gives us protobuf objects on the context; the rest of this
        # module operates on v0.3-pydantic, so translate at the entry boundary.
        request = _v03_conversions.to_compat_message(context.message)

        if context.current_task is None:
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
            # publish the task status submitted event (proto on the wire)
            await event_queue.enqueue_event(_v03_conversions.to_core_task(task))
        else:
            task = _v03_conversions.to_compat_task(context.current_task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        artifact = make_artifact(message=None)

        streaming_started = False
        try:
            async for response in self.agent(request_message_from_a2a(request)):
                if response.input_required:
                    await updater.requires_input(
                        message=to_core_message(
                            make_input_required_message(
                                context_id=task.context_id,
                                task_id=task.id,
                                text=response.input_required,
                                context=response.context,
                            )
                        ),
                    )
                    return

                if response.streaming_text:
                    artifact = copy_artifact(
                        artifact=artifact,
                        message={"content": response.streaming_text},
                        context=response.context,
                    )

                    await updater.add_artifact(
                        parts=to_core_parts(artifact.parts),
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

        except Exception as e:
            raise InternalError(repr(e)) from e

        await updater.add_artifact(
            artifact_id=artifact.artifact_id,
            name=artifact.name,
            parts=to_core_parts(artifact.parts),
            metadata=artifact.metadata,
            extensions=artifact.extensions,
            append=streaming_started,
            last_chunk=True,
        )

        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass
