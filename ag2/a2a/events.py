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

from ag2.events import BaseEvent, Field, ToolCallEvent


class A2AEvent(BaseEvent):
    """Base marker for every A2A wire-event surfaced into the AG2 stream.

    Subclasses wrap the four ``StreamResponse.payload`` oneof variants
    from a2a-sdk v1.x. Filter on this base for a transport-agnostic
    firehose. Marked ``__transient__`` so wire-format echoes don't
    persist into ``stream.history`` and get fed back to the LLM.
    """

    __transient__ = True


class A2ATaskSnapshot(A2AEvent):
    """Full ``Task`` snapshot (``StreamResponse.payload="task"``)."""

    task: Task = Field(repr=False)


class A2AMessage(A2AEvent):
    """Standalone ``Message`` (``StreamResponse.payload="message"``)."""

    message: Message = Field(repr=False)


class A2ATaskStatusUpdate(A2AEvent):
    """``TaskStatusUpdateEvent`` (``payload="status_update"``); ``state`` is duplicated for filtering."""

    update: TaskStatusUpdateEvent = Field(repr=False)
    state: TaskState


class A2ATaskArtifactUpdate(A2AEvent):
    """``TaskArtifactUpdateEvent`` (``payload="artifact_update"``); ``append``/``last_chunk`` exposed for chunk-aware logic."""

    update: TaskArtifactUpdateEvent = Field(repr=False)
    append: bool = False
    last_chunk: bool = False


class A2ATextArtifact(A2ATaskArtifactUpdate):
    """Typed view over a text-only artifact chunk — flat ``text`` alongside the raw protobuf."""

    text: str


class A2AToolCallArtifact(A2ATaskArtifactUpdate):
    """Typed view over a ``tool-call+json`` artifact — pre-parsed ``call`` so subscribers skip JSON-decode."""

    call: ToolCallEvent
