# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Callable

from acp import schema

from ag2 import MemoryStream
from ag2.acp.types import SessionUpdate
from ag2.context import ConversationContext
from ag2.events import BaseEvent, ToolCallEvent

Responder = Callable[[ToolCallEvent], BaseEvent]


class RecordingRun:
    """A live run for the bridge to act in: a real stream, answered by ``respond``.

    Records every event sent on the stream. ``respond=None`` leaves a tool call
    unanswered, as a run whose tool never completes.
    """

    def __init__(self, respond: Responder | None = None) -> None:
        self.stream = MemoryStream()
        self.context = ConversationContext(stream=self.stream)
        self.sent: list[BaseEvent] = []
        self.first_send = asyncio.Event()
        self._respond = respond
        self.stream.subscribe(self._on_event, sync_to_thread=False)

    async def _on_event(self, event: BaseEvent) -> None:
        self.sent.append(event)
        self.first_send.set()
        # A turn that failed sends a ToolResultEvent on its way out, to close
        # the call off in history; only the call itself is answered.
        if self._respond is not None and isinstance(event, ToolCallEvent):
            await self.context.send(self._respond(event))


def chunk_text(update: SessionUpdate) -> str:
    """The text of a message chunk the test expects to carry text."""
    assert isinstance(update, (schema.UserMessageChunk, schema.AgentMessageChunk, schema.AgentThoughtChunk)), update
    assert isinstance(update.content, schema.TextContentBlock), update.content
    return update.content.text


def tool_call_text(update: schema.ToolCallProgress) -> str:
    """The text of a tool call update's first content item, which the test expects to be text."""
    assert update.content is not None
    item = update.content[0]
    assert isinstance(item, schema.ContentToolCallContent), item
    assert isinstance(item.content, schema.TextContentBlock), item.content
    return item.content.text
