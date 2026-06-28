# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Pure translation between ACP SDK models and AG2 beta events.

Functions here take the ``acp.schema`` model objects directly (no ``model_dump``
indirection) and dispatch with :func:`isinstance`, so the mapping is checked
against the real SDK types rather than stringly-typed dicts.
"""

import base64
import json

from acp import schema

from ag2.events import BaseEvent, ModelMessageChunk, ModelReasoning
from ag2.events.tool_events import BuiltinToolCallEvent, BuiltinToolResultEvent, ToolResult
from ag2.events.types import BinaryResult, Usage

from .events import ACPAvailableCommands, ACPModeChange, ACPPlan, ACPPlanEntry
from .types import ContentBlock, SessionUpdate, ToolCallContent


def block_text(block: ContentBlock | None) -> str:
    """The text of a single content block, or ``""`` if it carries no text."""
    return block.text if isinstance(block, schema.TextContentBlock) else ""


def block_to_files(block: ContentBlock | None) -> list[BinaryResult]:
    """Decode a single ``image``/``audio`` content block into binary results."""
    if isinstance(block, (schema.ImageContentBlock, schema.AudioContentBlock)):
        return [BinaryResult(data=base64.b64decode(block.data), metadata={"mimeType": block.mime_type})]
    return []


def content_blocks_to_text(blocks: list[ContentBlock] | None) -> str:
    """Concatenate the text of any ``text`` content blocks; ignore the rest."""
    return "".join(block_text(b) for b in (blocks or ()))


def content_blocks_to_files(blocks: list[ContentBlock] | None) -> list[BinaryResult]:
    """Decode ``image``/``audio`` content blocks into binary results."""
    files: list[BinaryResult] = []
    for b in blocks or ():
        files.extend(block_to_files(b))
    return files


def _tool_content_text(content: list[ToolCallContent] | None) -> str:
    """Extract text from a tool call's ``content`` list (``ContentToolCallContent``)."""
    return "".join(
        block_text(item.content) for item in (content or ()) if isinstance(item, schema.ContentToolCallContent)
    )


def map_usage(usage: schema.Usage | None) -> Usage:
    """Map an ACP ``Usage`` onto AG2's :class:`Usage` (absent -> empty)."""
    if usage is None:
        return Usage()
    return Usage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cache_read_input_tokens=usage.cached_read_tokens,
        cache_creation_input_tokens=usage.cached_write_tokens,
        thinking_tokens=usage.thought_tokens,
    )


def map_session_update(update: SessionUpdate) -> BaseEvent | None:
    """Translate one ACP ``session/update`` into an AG2 event.

    Returns ``None`` for variants with no meaningful AG2 representation
    (``user_message_chunk``, ``usage_update``, ``session_info_update``,
    ``config_option_update``). ``usage_update`` is handled out-of-band by the
    client via :func:`map_usage`.
    """
    if isinstance(update, schema.AgentMessageChunk):
        return ModelMessageChunk(block_text(update.content))

    if isinstance(update, schema.AgentThoughtChunk):
        return ModelReasoning(block_text(update.content))

    if isinstance(update, schema.ToolCallStart):
        return BuiltinToolCallEvent(
            id=update.tool_call_id,
            name=update.title or "tool",
            arguments=json.dumps(update.raw_input or {}),
        )

    if isinstance(update, schema.ToolCallProgress):
        text = _tool_content_text(update.content) or (update.status or "")
        return BuiltinToolResultEvent(
            parent_id=update.tool_call_id,
            name=update.title,
            result=ToolResult(text),
        )

    if isinstance(update, schema.AgentPlanUpdate):
        return ACPPlan(
            entries=[ACPPlanEntry(content=e.content, status=e.status, priority=e.priority) for e in update.entries]
        )

    if isinstance(update, schema.CurrentModeUpdate):
        return ACPModeChange(mode_id=update.current_mode_id)

    if isinstance(update, schema.AvailableCommandsUpdate):
        return ACPAvailableCommands(commands=[c.name for c in update.available_commands])

    return None
