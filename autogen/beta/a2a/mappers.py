# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from base64 import b64decode, b64encode
from collections.abc import Iterable, Iterator
from enum import Enum
from typing import Any, cast
from uuid import uuid4

from a2a.types import (
    Artifact,
    DataPart,
    FilePart,
    FileWithBytes,
    FileWithUri,
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TextPart,
)

from autogen.beta.events import (
    BinaryInput,
    DataInput,
    FileIdInput,
    ModelMessageChunk,
    ModelRequest,
    TextInput,
    UrlInput,
)
from autogen.beta.events.input_events import BinaryType, Input

from .utils import AG2_BETA_METADATA_KEY_PREFIX

__all__ = (
    "a2a_message_to_inputs",
    "a2a_parts_to_inputs",
    "artifact_text",
    "followup_user_message",
    "hitl_replay_queue",
    "initial_inputs",
    "input_required_message",
    "inputs_to_a2a_parts",
    "model_request_to_a2a_message",
    "task_artifact_update_to_chunks",
    "text_from_message",
    "text_from_parts",
    "text_parts",
    "user_messages",
)


class _PartKind(str, Enum):
    """Internal marker for the original beta `Input` subtype, recorded in
    `Part.metadata` so we can round-trip back into the right class."""

    TEXT = "text"
    BINARY = "binary"
    URL = "url"
    FILE_ID = "file_id"
    DATA_DICT = "data_dict"
    DATA_VALUE = "data_value"


_PART_KIND_KEY = AG2_BETA_METADATA_KEY_PREFIX + "kind"
_BINARY_TYPE_KEY = AG2_BETA_METADATA_KEY_PREFIX + "binary_type"
_FILENAME_KEY = AG2_BETA_METADATA_KEY_PREFIX + "filename"
_VENDOR_METADATA_KEY = AG2_BETA_METADATA_KEY_PREFIX + "vendor_metadata"


def inputs_to_a2a_parts(inputs: Iterable[Input]) -> list[Part]:
    """Convert beta `Input` events into A2A wire `Part`s."""
    return [_input_to_part(inp) for inp in inputs]


def a2a_parts_to_inputs(parts: Iterable[Part]) -> list[Input]:
    """Convert A2A wire `Part`s back into beta `Input` events."""
    return [_part_to_input(p) for p in parts]


def model_request_to_a2a_message(
    req: ModelRequest,
    *,
    context_id: str | None,
    task_id: str | None = None,
) -> Message:
    """Build an A2A user `Message` from a beta `ModelRequest`."""
    return Message(
        role=Role.user,
        parts=inputs_to_a2a_parts(req.parts),
        message_id=uuid4().hex,
        context_id=context_id,
        task_id=task_id,
    )


def a2a_message_to_inputs(msg: Message) -> list[Input]:
    """Extract beta `Input`s from an A2A `Message`."""
    return a2a_parts_to_inputs(msg.parts)


def task_artifact_update_to_chunks(event: TaskArtifactUpdateEvent) -> Iterator[ModelMessageChunk]:
    """Yield `ModelMessageChunk`s for any text content in an artifact update."""
    for text in _iter_artifact_text(event.artifact):
        if text:
            yield ModelMessageChunk(text)


def artifact_text(artifact: Artifact) -> str:
    """Concatenate all text content from an artifact."""
    return "".join(_iter_artifact_text(artifact))


def text_from_message(message: Message) -> str:
    """Concatenate every `TextPart` in an A2A message."""
    return text_from_parts(message.parts)


def text_from_parts(parts: Iterable[Part]) -> str:
    """Concatenate every `TextPart` in a sequence of parts (other parts ignored)."""
    return "".join(p.root.text for p in parts if isinstance(p.root, TextPart))


def text_parts(text: str) -> list[Part]:
    """Wrap a string into a single-element parts list."""
    return [Part(root=TextPart(text=text))]


def user_messages(task: Task) -> list[Message]:
    """Return the user-role messages from a task's history, in order."""
    return [m for m in task.history or () if m.role == Role.user]


def initial_inputs(task: Task) -> list[Input]:
    """Extract the agent's initial inputs (first user message) from a task."""
    msgs = user_messages(task)
    if not msgs:
        return []
    return a2a_message_to_inputs(msgs[0])


def hitl_replay_queue(task: Task) -> list[str]:
    """Extract the queue of human inputs already provided in this task.

    Each entry is the text body of a follow-up user message that arrived after
    the initial request. Order is preserved — they will be replayed into the
    agent's `context.input(...)` calls in the same order.
    """
    msgs = user_messages(task)
    return [text_from_message(m) for m in msgs[1:]]


def input_required_message(text: str, *, context_id: str, task_id: str) -> Message:
    """Construct an agent message that asks the client for human input."""
    return Message(
        role=Role.agent,
        parts=text_parts(text),
        message_id=uuid4().hex,
        context_id=context_id,
        task_id=task_id,
    )


def followup_user_message(text: str, *, context_id: str, task_id: str) -> Message:
    """Construct a follow-up user message that resumes a suspended task."""
    return Message(
        role=Role.user,
        parts=text_parts(text),
        message_id=uuid4().hex,
        context_id=context_id,
        task_id=task_id,
    )


def _input_to_part(inp: Input) -> Part:
    if isinstance(inp, TextInput):
        return Part(
            root=TextPart(
                text=inp.content,
                metadata=_metadata({_PART_KIND_KEY: _PartKind.TEXT.value}, inp.metadata),
            )
        )
    if isinstance(inp, BinaryInput):
        return Part(
            root=FilePart(
                file=FileWithBytes(
                    bytes=b64encode(inp.data).decode("ascii"),
                    mime_type=str(inp.media_type),
                    name=None,
                ),
                metadata=_metadata(
                    {
                        _PART_KIND_KEY: _PartKind.BINARY.value,
                        _BINARY_TYPE_KEY: inp.kind.value,
                        _VENDOR_METADATA_KEY: dict(inp.vendor_metadata) or None,
                    },
                    inp.metadata,
                ),
            )
        )
    if isinstance(inp, FileIdInput):
        return Part(
            root=DataPart(
                data={"file_id": inp.file_id, "filename": inp.filename},
                metadata=_metadata(
                    {_PART_KIND_KEY: _PartKind.FILE_ID.value, _FILENAME_KEY: inp.filename},
                    inp.metadata,
                ),
            )
        )
    if isinstance(inp, UrlInput):
        return Part(
            root=FilePart(
                file=FileWithUri(uri=inp.url, mime_type=None, name=None),
                metadata=_metadata(
                    {_PART_KIND_KEY: _PartKind.URL.value, _BINARY_TYPE_KEY: inp.kind.value},
                    inp.metadata,
                ),
            )
        )
    if isinstance(inp, DataInput):
        if isinstance(inp.data, dict):
            return Part(
                root=DataPart(
                    data=cast(dict[str, Any], inp.data),
                    metadata=_metadata({_PART_KIND_KEY: _PartKind.DATA_DICT.value}, inp.metadata),
                )
            )
        return Part(
            root=DataPart(
                data={"value": inp.data},
                metadata=_metadata({_PART_KIND_KEY: _PartKind.DATA_VALUE.value}, inp.metadata),
            )
        )
    raise TypeError(f"Unsupported Input type: {type(inp).__name__}")


def _part_to_input(part: Part) -> Input:
    root = part.root
    raw_metadata = dict(root.metadata or {})
    kind_marker = raw_metadata.get(_PART_KIND_KEY)
    user_metadata = _strip_internal(raw_metadata)

    if isinstance(root, TextPart):
        return TextInput(root.text, metadata=user_metadata)

    if isinstance(root, FilePart):
        binary_type = _coerce_binary_type(raw_metadata.get(_BINARY_TYPE_KEY))
        if isinstance(root.file, FileWithBytes):
            return BinaryInput(
                b64decode(root.file.bytes),
                media_type=root.file.mime_type or "application/octet-stream",
                kind=binary_type,
                vendor_metadata=dict(raw_metadata.get(_VENDOR_METADATA_KEY) or {}),
                metadata=user_metadata,
            )
        if isinstance(root.file, FileWithUri):
            return UrlInput(root.file.uri, kind=binary_type, metadata=user_metadata)
        raise TypeError(f"Unsupported FilePart.file type: {type(root.file).__name__}")

    if isinstance(root, DataPart):
        if kind_marker == _PartKind.FILE_ID.value:
            return FileIdInput(
                str(root.data.get("file_id")),
                filename=cast(str | None, root.data.get("filename")),
                metadata=user_metadata,
            )
        if kind_marker == _PartKind.DATA_VALUE.value:
            return DataInput(root.data.get("value"), metadata=user_metadata)
        return DataInput(root.data, metadata=user_metadata)

    raise TypeError(f"Unsupported Part root type: {type(root).__name__}")


def _iter_artifact_text(artifact: Artifact) -> Iterator[str]:
    for part in artifact.parts:
        root = part.root
        if isinstance(root, TextPart):
            yield root.text
        elif isinstance(root, DataPart):
            # Cross-compat with legacy `autogen/a2a/`: that executor emits
            # streaming chunks as `DataPart(data={"content": text})`.
            # We don't produce this shape ourselves but accept it on read.
            content = root.data.get("content")
            if isinstance(content, str):
                yield content


def _metadata(internal: dict[str, Any], user: dict[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {k: v for k, v in internal.items() if v is not None}
    if user:
        merged.update(user)
    return merged or None


def _strip_internal(metadata: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metadata.items() if not k.startswith(AG2_BETA_METADATA_KEY_PREFIX)}


def _coerce_binary_type(value: Any) -> BinaryType:
    if isinstance(value, BinaryType):
        return value
    if isinstance(value, str):
        try:
            return BinaryType(value)
        except ValueError:
            return BinaryType.BINARY
    return BinaryType.BINARY
