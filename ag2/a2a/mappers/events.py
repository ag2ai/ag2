# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from a2a.types import (
    Artifact,
    Message,
    Part,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)

from ag2.events import ClientToolCallEvent, ModelMessageChunk, ToolCallEvent

from ..events import (
    A2AEvent,
    A2AMessage,
    A2ATaskArtifactUpdate,
    A2ATaskSnapshot,
    A2ATaskStatusUpdate,
    A2ATextArtifact,
    A2AToolCallArtifact,
)
from ..extension import MIME_TOOL_CALL
from .parts import data_part, is_data_part_with_mime, part_data_to_python, struct_from_dict
from .tools import call_to_payload, payload_to_call


def chunk_to_text_artifact(
    event: ModelMessageChunk,
    *,
    artifact_id: str,
    task_id: str,
    context_id: str,
    append: bool = True,
    last_chunk: bool = False,
    name: str | None = None,
    description: str | None = None,
    artifact_metadata: Mapping[str, Any] | None = None,
    update_metadata: Mapping[str, Any] | None = None,
) -> A2ATextArtifact:
    """Map a streaming text chunk to an A2A text-artifact event.

    The artifact is structured to play well with A2A append-streaming:
    every chunk reuses the same ``artifact_id`` with ``append=True`` so
    the client side can concatenate them. Caller flips ``last_chunk``
    on the final piece.

    ``name`` / ``description`` / ``artifact_metadata`` populate the
    optional ``Artifact`` fields; ``update_metadata`` rides on the
    enclosing ``TaskArtifactUpdateEvent``.
    """
    artifact = _build_artifact(
        artifact_id=artifact_id,
        parts=[Part(text=event.content)],
        name=name,
        description=description,
        artifact_metadata=artifact_metadata,
    )
    update = _build_artifact_update(
        task_id=task_id,
        context_id=context_id,
        artifact=artifact,
        append=append,
        last_chunk=last_chunk,
        update_metadata=update_metadata,
    )
    return A2ATextArtifact(
        update=update,
        append=append,
        last_chunk=last_chunk,
        text=event.content,
    )


def client_call_to_artifact(
    event: ClientToolCallEvent,
    *,
    task_id: str,
    context_id: str,
    name: str | None = None,
    description: str | None = None,
    artifact_metadata: Mapping[str, Any] | None = None,
    update_metadata: Mapping[str, Any] | None = None,
) -> A2AToolCallArtifact:
    """Map a pending client-side tool invocation to a tool-call artifact.

    Each call gets a fresh artifact keyed by the call id with
    ``last_chunk=True`` — a tool-call payload is delivered atomically,
    not streamed in chunks. ``name`` defaults to the tool name so
    downstream consumers can filter by it without decoding the payload.
    """
    call = ToolCallEvent(id=event.id, name=event.name, arguments=event.arguments)
    artifact = _build_artifact(
        artifact_id=event.id,
        parts=[data_part(call_to_payload(call), media_type=MIME_TOOL_CALL)],
        name=name if name is not None else event.name,
        description=description,
        artifact_metadata=artifact_metadata,
    )
    update = _build_artifact_update(
        task_id=task_id,
        context_id=context_id,
        artifact=artifact,
        append=False,
        last_chunk=True,
        update_metadata=update_metadata,
    )
    return A2AToolCallArtifact(
        update=update,
        append=False,
        last_chunk=True,
        call=call,
    )


def task_state_to_status_update(
    state: TaskState,
    *,
    task_id: str,
    context_id: str,
    message: Message | None = None,
    timestamp: datetime | None = None,
) -> A2ATaskStatusUpdate:
    """Build an ``A2ATaskStatusUpdate`` for a lifecycle transition.

    Used by the executor to surface ``start_work``/``complete``/``failed``
    transitions through the same A2AEvent layer that artifact updates
    use, instead of going through ``TaskUpdater`` directly.

    ``timestamp`` populates ``TaskStatus.timestamp``; pass ``None`` to
    let the wire stay empty (callers that want a real transition time
    should supply a ``datetime.now(tz=UTC)``).
    """
    status_kwargs: dict[str, Any] = {"state": state}
    if message is not None:
        status_kwargs["message"] = message
    if timestamp is not None:
        status_kwargs["timestamp"] = timestamp
    status = TaskStatus(**status_kwargs)
    update = TaskStatusUpdateEvent(task_id=task_id, context_id=context_id, status=status)
    return A2ATaskStatusUpdate(update=update, state=state)


def _build_artifact(
    *,
    artifact_id: str,
    parts: list[Part],
    name: str | None,
    description: str | None,
    artifact_metadata: Mapping[str, Any] | None,
) -> Artifact:
    kwargs: dict[str, Any] = {"artifact_id": artifact_id, "parts": parts}
    if name:
        kwargs["name"] = name
    if description:
        kwargs["description"] = description
    if artifact_metadata:
        kwargs["metadata"] = struct_from_dict(dict(artifact_metadata))
    return Artifact(**kwargs)


def _build_artifact_update(
    *,
    task_id: str,
    context_id: str,
    artifact: Artifact,
    append: bool,
    last_chunk: bool,
    update_metadata: Mapping[str, Any] | None,
) -> TaskArtifactUpdateEvent:
    kwargs: dict[str, Any] = {
        "task_id": task_id,
        "context_id": context_id,
        "artifact": artifact,
        "append": append,
        "last_chunk": last_chunk,
    }
    if update_metadata:
        kwargs["metadata"] = struct_from_dict(dict(update_metadata))
    return TaskArtifactUpdateEvent(**kwargs)


def a2a_event_to_sdk(
    event: A2AEvent,
) -> Task | Message | TaskStatusUpdateEvent | TaskArtifactUpdateEvent:
    """Unwrap an A2A event back to the bare SDK protobuf for ``event_queue``.

    ``EventQueue.enqueue_event`` accepts the bare protobuf objects (not
    ``StreamResponse``); the SDK request handler does the StreamResponse
    wrapping itself. We unwrap here so callers can keep working with
    typed AG2 events all the way up to the queue boundary.
    """
    if isinstance(event, A2ATaskArtifactUpdate):
        return event.update
    if isinstance(event, A2ATaskStatusUpdate):
        return event.update
    if isinstance(event, A2AMessage):
        return event.message
    if isinstance(event, A2ATaskSnapshot):
        return event.task
    raise TypeError(f"Cannot unwrap {type(event).__name__} to A2A SDK type")


def parse_stream_response(response: StreamResponse) -> A2AEvent:
    """Decode a wire ``StreamResponse`` into a typed A2A event.

    Picks typed subclasses where possible — text-only artifacts become
    ``A2ATextArtifact``, ``tool-call+json`` artifacts become
    ``A2AToolCallArtifact``. Unrecognised artifact shapes fall through
    to the base ``A2ATaskArtifactUpdate``.
    """
    payload = response.WhichOneof("payload")
    if payload == "task":
        return A2ATaskSnapshot(task=response.task)
    if payload == "message":
        return A2AMessage(message=response.message)
    if payload == "status_update":
        update = response.status_update
        return A2ATaskStatusUpdate(update=update, state=update.status.state)
    if payload == "artifact_update":
        return parse_artifact_update(response.artifact_update)
    raise ValueError(f"Unexpected StreamResponse payload: {payload!r}")


def parse_artifact_update(update: TaskArtifactUpdateEvent) -> A2ATaskArtifactUpdate:
    """Pick the most specific ``A2ATaskArtifactUpdate`` subclass for an update.

    Returns ``A2ATextArtifact`` for text-only artifacts, ``A2AToolCallArtifact``
    for AG2 ``tool-call+json`` extension parts, and the base
    ``A2ATaskArtifactUpdate`` for any other shape.
    """
    artifact = update.artifact
    parts = list(artifact.parts)
    text_only = parts and all(bool(p.text) for p in parts)
    if text_only:
        return A2ATextArtifact(
            update=update,
            append=update.append,
            last_chunk=update.last_chunk,
            text="".join(p.text for p in parts),
        )

    if len(parts) == 1 and is_data_part_with_mime(parts[0], MIME_TOOL_CALL):
        return A2AToolCallArtifact(
            update=update,
            append=update.append,
            last_chunk=update.last_chunk,
            call=payload_to_call(part_data_to_python(parts[0])),
        )

    return A2ATaskArtifactUpdate(
        update=update,
        append=update.append,
        last_chunk=update.last_chunk,
    )


def parse_task_artifact(
    artifact: Artifact,
    *,
    task_id: str,
    context_id: str,
) -> A2ATaskArtifactUpdate:
    """Wrap a polling-snapshot ``Artifact`` as a typed update event.

    A2A polling re-reads ``task.artifacts`` (a cumulative list of
    finalised artifacts) instead of an incremental
    ``TaskArtifactUpdateEvent`` stream. Synthesise the streaming wire
    shape so the same downstream code path can handle both transports.
    Each polled artifact is treated as a final, non-appended chunk —
    polling never sees mid-stream chunks.
    """
    update = TaskArtifactUpdateEvent(
        task_id=task_id,
        context_id=context_id,
        artifact=artifact,
        append=False,
        last_chunk=True,
    )
    return parse_artifact_update(update)
