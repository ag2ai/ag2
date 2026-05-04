# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator, Iterator
from typing import Any, cast
from uuid import uuid4

from a2a.client import Client
from a2a.compat.v0_3 import conversions as _v03_conversions
from a2a.compat.v0_3.types import (
    AgentCard,
    Artifact,
    DataPart,
    Message,
    MessageSendParams,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.compat.v0_3.types import (
    SendMessageRequest as _CompatSendMessageRequest,
)
from a2a.compat.v0_3.types import (
    TaskResubscriptionRequest as _CompatResubscriptionRequest,
)
from a2a.types import TaskState as _CoreTaskState
from a2a.types.a2a_pb2 import (
    AgentCard as _CoreAgentCard,
)
from a2a.types.a2a_pb2 import (
    GetTaskRequest as _CoreGetTaskRequest,
)
from a2a.types.a2a_pb2 import (
    Message as _CoreMessage,
)
from a2a.types.a2a_pb2 import (
    Part as _CorePart,
)
from a2a.types.a2a_pb2 import (
    StreamResponse as _CoreStreamResponse,
)

from autogen.agentchat.remote import RequestMessage, ResponseMessage
from autogen.events.client_events import StreamEvent


def get_message_text(message: Message, delimiter: str = "\n") -> str:
    """Join text from all TextPart parts in a Message (compat-shim)."""
    return delimiter.join(p.root.text for p in message.parts if isinstance(p.root, TextPart))


def new_artifact(name: str, parts: list[Part], description: str | None = None) -> Artifact:
    """Construct an Artifact with a fresh id (compat-shim)."""
    return Artifact(artifact_id=uuid4().hex, name=name, parts=parts, description=description)


def new_agent_text_message(
    text: str,
    *,
    context_id: str | None = None,
    task_id: str | None = None,
) -> Message:
    """Construct a Message from agent with a single TextPart (compat-shim)."""
    return Message(
        role=Role.agent,
        parts=[Part(root=TextPart(text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
        task_id=task_id,
    )


AG2_METADATA_KEY_PREFIX = "ag2_"
CLIENT_TOOLS_KEY = f"{AG2_METADATA_KEY_PREFIX}client_tools"
CONTEXT_KEY = f"{AG2_METADATA_KEY_PREFIX}context_update"

RESULT_ARTIFACT_NAME = "result"

from autogen.a2a.constants import A2UI_MIME_TYPE


def request_message_to_a2a(
    request_message: RequestMessage,
    context_id: str,
    extra_parts: list[Part] | None = None,
) -> Message:
    metadata: dict[str, Any] = {}
    if request_message.client_tools:
        metadata[CLIENT_TOOLS_KEY] = request_message.client_tools
    if request_message.context:
        metadata[CONTEXT_KEY] = request_message.context

    parts = [message_to_part(message) for message in request_message.messages]
    if extra_parts:
        parts.extend(extra_parts)

    return Message(
        role=Role.user,
        parts=parts,
        message_id=uuid4().hex,
        context_id=context_id,
        metadata=metadata,
    )


def request_message_from_a2a(message: Message) -> RequestMessage:
    metadata = message.metadata or {}
    return RequestMessage(
        messages=[message_from_part(part) for part in message.parts],
        context=metadata.get(CONTEXT_KEY),
        client_tools=metadata.get(CLIENT_TOOLS_KEY, []),
    )


def response_message_from_a2a_task(task: Task) -> ResponseMessage | None:
    history = [message_from_part(p) for m in (task.history or []) for p in m.parts]

    if task.status.state is TaskState.input_required:
        message: str | None = None
        context: dict[str, Any] | None = None

        if task.history:
            input_message = task.history[-1]
            message = get_message_text(input_message)

            if input_message.metadata:
                context = input_message.metadata.get(CONTEXT_KEY)

            if "role" not in history[-1]:
                history[-1]["role"] = "assistant"

        return ResponseMessage(
            messages=history,
            input_required=message or "Please provide input:",
            context=context,
        )

    response = response_message_from_a2a_artifacts(task.artifacts)
    if response:
        response.messages = history + response.messages
    return response


def response_message_from_a2a_artifacts(artifacts: list[Artifact] | None) -> ResponseMessage | None:
    if not artifacts:
        return None

    if len(artifacts) > 1:
        raise NotImplementedError("Multiple artifacts are not supported")

    artifact = artifacts[-1]

    if not artifact.parts:
        return None

    # Check if there are any data parts mixed with text parts
    has_data = any(isinstance(p.root, DataPart) for p in artifact.parts)

    if not has_data:
        # Text-only (metadata like name, role)
        return ResponseMessage(
            messages=[message_from_part(p) for p in artifact.parts],
            context=(artifact.metadata or {}).get(CONTEXT_KEY),
        )

    # Check if any data parts are A2UI — if so, merge text + data into single dict
    has_a2ui = any(
        isinstance(p.root, DataPart) and p.root.metadata and p.root.metadata.get("mimeType") == A2UI_MIME_TYPE
        for p in artifact.parts
    )

    text_content: list[str] = []
    data_messages: list[dict[str, Any]] = []
    for p in artifact.parts:
        if isinstance(p.root, TextPart):
            text_content.append(p.root.text)
        else:
            data_messages.append(message_from_part(p))

    if has_a2ui:
        # Merge text + A2UI data into a single dict so client gets both
        combined: dict[str, Any] = {}
        if text_content:
            combined["content"] = "\n".join(text_content)
        for dm in data_messages:
            combined.update(dm)
        return ResponseMessage(
            messages=[combined],
            context=(artifact.metadata or {}).get(CONTEXT_KEY),
        )

    # Non-A2UI: keep text and data as separate messages
    messages: list[dict[str, Any]] = []
    if text_content:
        messages.append({"content": "\n".join(text_content)})
    messages.extend(data_messages)

    return ResponseMessage(
        messages=messages,
        context=(artifact.metadata or {}).get(CONTEXT_KEY),
    )


def update_artifact_to_streaming(event: TaskArtifactUpdateEvent) -> Iterator[StreamEvent]:
    if event.last_chunk is False:  # respect None
        for part in event.artifact.parts:
            root = part.root
            if isinstance(root, TextPart):
                text = root.text
            elif isinstance(root, DataPart):
                text = root.data.get("content", "")
            yield StreamEvent(content=text)


def response_message_from_a2a_message(message: Message) -> ResponseMessage | None:
    text_parts: list[Part] = []
    data_parts: list[Part] = []
    for part in message.parts:
        if isinstance(part.root, TextPart):
            text_parts.append(part)
        elif isinstance(part.root, DataPart):
            data_parts.append(part)
        else:
            raise NotImplementedError(f"Unsupported part type: {type(part.root)}")

    has_a2ui = any(
        isinstance(p.root, DataPart) and p.root.metadata and p.root.metadata.get("mimeType") == A2UI_MIME_TYPE
        for p in data_parts
    )

    messages: list[dict[str, Any]] = []

    if has_a2ui:
        # Merge text + A2UI data into a single dict
        combined: dict[str, Any] = {}
        if text_parts:
            combined["content"] = "\n".join(cast(TextPart, t.root).text for t in text_parts)
        for dp in data_parts:
            combined.update(message_from_part(dp))
        messages.append(combined)
    else:
        # Non-A2UI: keep text and data as separate messages
        if len(text_parts) == 1:
            messages.append(message_from_part(text_parts[0]))
        elif len(text_parts) > 1:
            messages.append({"content": "\n".join(cast(TextPart, t.root).text for t in text_parts)})
        for dp in data_parts:
            messages.append(message_from_part(dp))

    return ResponseMessage(
        messages=messages,
        context=(message.metadata or {}).get(CONTEXT_KEY),
    )


def make_artifact(
    message: dict[str, Any] | None,
    context: dict[str, Any] | None = None,
    name: str = RESULT_ARTIFACT_NAME,
) -> Artifact:
    artifact = new_artifact(
        name=name,
        parts=[message_to_part(message)] if message else [],
        description=None,
    )

    if context:
        artifact.metadata = {CONTEXT_KEY: context}

    return artifact


def copy_artifact(
    artifact: Artifact,
    message: dict[str, Any] | None,
    context: dict[str, Any] | None = None,
) -> Artifact:
    updated_artifact = Artifact(
        artifact_id=artifact.artifact_id,
        description=artifact.description,
        parts=[message_to_part(message)] if message else [],
        name=artifact.name,
        metadata=artifact.metadata,
        extensions=artifact.extensions,
    )

    old_metadata = artifact.metadata or {}
    context = old_metadata.get(CONTEXT_KEY, {}) | (context or {})
    if context:
        old_metadata[CONTEXT_KEY] = context
        updated_artifact.metadata = old_metadata

    return updated_artifact


def make_input_required_message(
    text: str,
    context_id: str,
    task_id: str,
    context: dict[str, Any] | None = None,
) -> Message:
    message = new_agent_text_message(
        text=text,
        context_id=context_id,
        task_id=task_id,
    )
    if context:
        message.metadata = {CONTEXT_KEY: context}
    return message


def message_to_part(message: dict[str, Any]) -> Part:
    message = message.copy()
    text = message.pop("content", "") or ""
    return Part(
        root=TextPart(
            text=text,
            metadata=message or None,
        )
    )


def message_from_part(part: Part) -> dict[str, Any]:
    root = part.root

    if isinstance(root, TextPart):
        return {
            **(root.metadata or {}),
            "content": root.text,
        }

    elif isinstance(root, DataPart):
        if (  # pydantic-ai specific
            set(root.data.keys()) == {RESULT_ARTIFACT_NAME}
            and root.metadata
            and "json_schema" in root.metadata
            and isinstance(data := root.data[RESULT_ARTIFACT_NAME], dict)
        ):
            return data

        # Preserve DataPart data as-is (structured dict)
        return dict(root.data)

    else:
        raise NotImplementedError(f"Unsupported part type: {type(part.root)}")


ClientStreamEvent = Message | tuple[Task, TaskStatusUpdateEvent | TaskArtifactUpdateEvent | None]
"""What our compat send_message/subscribe iterators yield: either a standalone
Message (message-only flow) or a (Task, optional update event) tuple
(task-lifecycle flow). Mirrors the v0.3 ClientEvent shape so client.py keeps
working unchanged."""


def stream_chunk_to_compat(
    chunk: _CoreStreamResponse,
    last_task: Task | None,
) -> tuple[ClientStreamEvent | None, Task | None]:
    """Translate one v1.0 protobuf StreamResponse into a v0.3-shaped event.

    Returns ``(event, new_last_task)``. The caller threads ``new_last_task``
    back in on the next call so that streamed status/artifact updates inherit
    the most recent Task snapshot (state for completion checks, accumulated
    artifacts for the final response builder).
    """
    if chunk.HasField("message"):
        return _v03_conversions.to_compat_message(chunk.message), last_task

    if chunk.HasField("task"):
        task = _v03_conversions.to_compat_task(chunk.task)
        return (task, None), task

    if chunk.HasField("status_update"):
        status_ev: TaskStatusUpdateEvent = _v03_conversions.to_compat_task_status_update_event(chunk.status_update)
        if last_task is not None:
            task = last_task.model_copy(update={"status": status_ev.status})
        else:
            task = Task(id=status_ev.task_id, context_id=status_ev.context_id, status=status_ev.status, history=[])
        return (task, status_ev), task

    if chunk.HasField("artifact_update"):
        artifact_ev: TaskArtifactUpdateEvent = _v03_conversions.to_compat_task_artifact_update_event(
            chunk.artifact_update
        )
        # Accumulate streamed artifacts on the carried Task snapshot so the
        # final response builder sees the complete artifact list when the task
        # reaches a terminal state.
        if last_task is not None:
            artifacts = list(last_task.artifacts or [])
            incoming = artifact_ev.artifact
            for i, art in enumerate(artifacts):
                if art.artifact_id == incoming.artifact_id:
                    artifacts[i] = (
                        art.model_copy(update={"parts": list(art.parts) + list(incoming.parts)})
                        if artifact_ev.append
                        else incoming
                    )
                    break
            else:
                artifacts.append(incoming)
            task = last_task.model_copy(update={"artifacts": artifacts})
        else:
            task = Task(
                id=artifact_ev.task_id,
                context_id=artifact_ev.context_id,
                status=None,  # type: ignore[arg-type]
                artifacts=[artifact_ev.artifact],
                history=[],
            )
        return (task, artifact_ev), task

    return None, last_task


async def compat_send_message(
    client: Client,
    message: Message,
) -> AsyncIterator[ClientStreamEvent]:
    core_req = _v03_conversions.to_core_send_message_request(
        _CompatSendMessageRequest(id=uuid4().hex, params=MessageSendParams(message=message))
    )
    last_task: Task | None = None
    async for chunk in client.send_message(core_req):
        event, last_task = stream_chunk_to_compat(chunk, last_task)
        if event is not None:
            yield event


async def compat_subscribe_to_task(
    client: Client,
    task_id: str,
) -> AsyncIterator[ClientStreamEvent]:
    core_req = _v03_conversions.to_core_subscribe_to_task_request(
        _CompatResubscriptionRequest(id=uuid4().hex, params=TaskIdParams(id=task_id))
    )
    last_task: Task | None = None
    async for chunk in client.subscribe(core_req):
        event, last_task = stream_chunk_to_compat(chunk, last_task)
        if event is not None:
            yield event


async def compat_get_task(client: Client, task_id: str) -> Task:
    """Fetch a task via the v1.0 Client, returning a v0.3-shaped pydantic Task."""
    core_task = await client.get_task(_CoreGetTaskRequest(id=task_id))
    return _v03_conversions.to_compat_task(core_task)


def to_core_agent_card(card: AgentCard) -> _CoreAgentCard:
    """Convert a v0.3 pydantic AgentCard into a v1.0 protobuf AgentCard."""
    return _v03_conversions.to_core_agent_card(card)


def to_compat_agent_card(card: _CoreAgentCard) -> AgentCard:
    """Convert a v1.0 protobuf AgentCard into a v0.3 pydantic AgentCard."""
    return _v03_conversions.to_compat_agent_card(card)


_TASK_STATE_TO_CORE: dict[TaskState, int] = {
    TaskState.submitted: _CoreTaskState.TASK_STATE_SUBMITTED,
    TaskState.working: _CoreTaskState.TASK_STATE_WORKING,
    TaskState.completed: _CoreTaskState.TASK_STATE_COMPLETED,
    TaskState.failed: _CoreTaskState.TASK_STATE_FAILED,
    TaskState.canceled: _CoreTaskState.TASK_STATE_CANCELED,
    TaskState.input_required: _CoreTaskState.TASK_STATE_INPUT_REQUIRED,
    TaskState.rejected: _CoreTaskState.TASK_STATE_REJECTED,
    TaskState.auth_required: _CoreTaskState.TASK_STATE_AUTH_REQUIRED,
}


def to_core_task_state(state: TaskState) -> int:
    """Map a v0.3 pydantic TaskState enum to its v1.0 protobuf int value.

    ``TaskUpdater.update_status`` annotates ``state`` as the protobuf enum
    class itself, but at runtime it accepts the underlying int value — that's
    what we return.
    """
    return _TASK_STATE_TO_CORE[state]


def to_core_message(message: Message) -> _CoreMessage:
    """Convert a v0.3 pydantic Message into a v1.0 protobuf Message."""
    return _v03_conversions.to_core_message(message)


def to_core_parts(parts: list[Part]) -> list[_CorePart]:
    """Convert a list of v0.3 pydantic Parts into v1.0 protobuf Parts."""
    return [_v03_conversions.to_core_part(p) for p in parts]
