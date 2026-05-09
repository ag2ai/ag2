# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from a2a.types import (
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)

from autogen.beta.events import BaseEvent, Field, ToolCallEvent


class A2AEvent(BaseEvent):
    """Base marker for every A2A wire-event surfaced into the AG2 stream.

    Subclasses wrap the four ``StreamResponse.payload`` oneof variants
    from a2a-sdk v1.x. Filtering on this base type catches all of them
    (e.g. ``stream.where(A2AEvent)`` for a transport-agnostic firehose).

    Marked ``__transient__`` so these wrappers are never persisted into
    ``stream.history`` — they are wire-format echoes, not AG2 conversation
    events; if they leaked into history, ``Agent._execute`` would feed
    them back into the LLM as bogus messages on the next turn.
    """

    __transient__ = True


class A2ATaskSnapshot(A2AEvent):
    """Wraps a full ``Task`` snapshot (``StreamResponse.payload="task"``).

    Emitted on the client when the server returns a complete task object
    (typically the bootstrap event before status/artifact updates start).
    """

    task: Task = Field(repr=False)


class A2AMessage(A2AEvent):
    """Wraps a standalone ``Message`` (``StreamResponse.payload="message"``).

    Emitted when the agent sends a message that is not part of an
    artifact stream — e.g. the final agent reply attached to a completed
    task or a one-shot non-task response.
    """

    message: Message = Field(repr=False)


class A2ATaskStatusUpdate(A2AEvent):
    """Wraps a ``TaskStatusUpdateEvent`` (``payload="status_update"``).

    ``state`` is duplicated as a top-level field so subscribers can
    filter on lifecycle transitions without unwrapping the protobuf
    (``stream.where(A2ATaskStatusUpdate.state == TaskState.TASK_STATE_COMPLETED)``).
    """

    update: TaskStatusUpdateEvent = Field(repr=False)
    state: TaskState


class A2ATaskArtifactUpdate(A2AEvent):
    """Wraps a ``TaskArtifactUpdateEvent`` (``payload="artifact_update"``).

    ``append`` and ``last_chunk`` are surfaced at the top level so
    subscribers can build chunk-aware logic (e.g. only persist on
    ``last_chunk=True``) without unwrapping the protobuf.
    """

    update: TaskArtifactUpdateEvent = Field(repr=False)
    append: bool = False
    last_chunk: bool = False


class A2ATextArtifact(A2ATaskArtifactUpdate):
    """Typed view over a text-only artifact chunk.

    Carries the flattened text alongside the raw protobuf so subscribers
    that only want the streaming text don't have to walk
    ``update.artifact.parts``. Server-side this is the channel used to
    forward ``ModelMessageChunk`` events to the A2A client.
    """

    text: str


class A2AToolCallArtifact(A2ATaskArtifactUpdate):
    """Typed view over a ``tool-call+json`` artifact (AG2 client-tools extension).

    The server emits one of these per pending client-side tool invocation
    so the client can execute the tool locally and continue the task.
    ``call`` is the already-parsed AG2 ``ToolCallEvent`` — subscribers
    don't have to decode the JSON ``DataPart`` themselves.
    """

    call: ToolCallEvent
