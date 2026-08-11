# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Pure translation between ACP SDK models and AG2 events.

Functions here take the ``acp.schema`` model objects directly (no ``model_dump``
indirection) and dispatch with :func:`isinstance`, so the mapping is checked
against the real SDK types rather than stringly-typed dicts.

Both directions live here:

* **ACP -> AG2** (:func:`map_session_update`, :func:`content_blocks_to_text`, …)
  serves the *client* side — AG2 driving an external CLI agent.
* **AG2 -> ACP** (:func:`prompt_to_inputs`, :func:`event_to_session_update`)
  serves the *serving* side — :class:`~ag2.acp.agent.ACPAgent` exposing an AG2
  agent to an ACP Client.
"""

import base64
import json
import logging
from collections.abc import Sequence
from typing import Any, cast

import acp
from acp import schema

from ag2.events import (
    AudioInput,
    BaseEvent,
    DataInput,
    DocumentInput,
    ImageInput,
    Input,
    ModelMessageChunk,
    ModelReasoning,
    TextInput,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
)
from ag2.events.tool_events import BuiltinToolCallEvent, BuiltinToolResultEvent, ToolResult
from ag2.events.types import BinaryResult, Usage
from ag2.types import SendableMessage

from .events import ACPAvailableCommands, ACPModeChange, ACPPlan, ACPPlanEntry
from .types import ContentBlock, SessionUpdate, ToolCallContent

logger = logging.getLogger(__name__)

# AG2 tool events carry no ACP ``kind`` (read / edit / execute / …), and guessing
# one from a tool's name would be wrong more often than useful.
DEFAULT_TOOL_KIND: "schema.ToolKind" = "other"


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
        return _plan(update.entries)

    # Incremental plan updates: only the ``items`` payload carries plan entries;
    # ``file``/``markdown`` reference a plan by URI or raw markdown and have no
    # ACPPlan equivalent, so they fall through to the debug log below.
    if isinstance(update, schema.AgentPlanContentUpdate) and isinstance(update.plan, schema.PlanUpdateItems):
        return _plan(update.plan.entries)

    if isinstance(update, schema.CurrentModeUpdate):
        return ACPModeChange(mode_id=update.current_mode_id)

    if isinstance(update, schema.AvailableCommandsUpdate):
        return ACPAvailableCommands(commands=[c.name for c in update.available_commands])

    # ``usage_update`` is consumed out-of-band by map_usage; anything else reaching
    # here is a session update this version has no event for — say so rather than
    # dropping it silently, so a protocol addition is visible in a debug log.
    if not isinstance(update, schema.UsageUpdate):
        logger.debug("no AG2 event for ACP session update %r; ignored", getattr(update, "session_update", update))
    return None


def _plan(entries: "Sequence[schema.PlanEntry]") -> ACPPlan:
    return ACPPlan(entries=[ACPPlanEntry(content=e.content, status=e.status, priority=e.priority) for e in entries])


def block_to_input(block: ContentBlock) -> Input | None:
    """Translate one ACP prompt content block into an AG2 input.

    Returns ``None`` for a block this version has no AG2 representation for —
    the caller decides whether to drop it or fail.

    Resource *links* are deliberately not fetched: a URI named by a Client is
    context, not authorization to dereference it. The link is passed through as
    text so the model can see it was referenced.
    """
    if isinstance(block, schema.TextContentBlock):
        return TextInput(block.text)

    if isinstance(block, schema.ImageContentBlock):
        if block.data:
            return ImageInput(data=base64.b64decode(block.data), media_type=cast(Any, block.mime_type))
        return ImageInput(block.uri) if block.uri else None

    if isinstance(block, schema.AudioContentBlock):
        return AudioInput(data=base64.b64decode(block.data), media_type=cast(Any, block.mime_type))

    if isinstance(block, schema.EmbeddedResourceContentBlock):
        resource = block.resource
        if isinstance(resource, schema.TextResourceContents):
            return TextInput(resource.text)
        # A blob the Client inlined — its bytes travelled with the prompt, so
        # using them reads nothing from the host filesystem.
        return DocumentInput(
            data=base64.b64decode(resource.blob),
            media_type=cast(Any, resource.mime_type or "application/octet-stream"),
        )

    if isinstance(block, schema.ResourceContentBlock):
        label = block.name or block.title or block.uri
        return TextInput(f"[resource] {label} ({block.uri})")

    return None


def prompt_to_inputs(blocks: "Sequence[ContentBlock]") -> list[Input]:
    """Translate an ACP ``session/prompt`` payload into AG2 inputs, in order.

    Blocks with no AG2 equivalent are logged and skipped rather than failing the
    turn — a Client sending one richer block should not lose the rest of its
    prompt.
    """
    inputs: list[Input] = []
    for block in blocks:
        mapped = block_to_input(block)
        if mapped is None:
            logger.debug("no AG2 input for ACP content block %r; skipped", getattr(block, "type", block))
            continue
        inputs.append(mapped)
    return inputs


def event_to_session_update(event: BaseEvent, *, stream_thoughts: bool = False) -> SessionUpdate | None:
    """Translate one AG2 event into an ACP ``session/update``.

    Returns ``None`` for events with no ACP equivalent, so the caller can drop
    them without a branch per event type.

    ``stream_thoughts`` gates :class:`ModelReasoning`. Reasoning is internal by
    default and an ACP Client may be an external audience, so exposing it is an
    explicit opt-in rather than a side effect of connecting.

    The final :class:`~ag2.events.ModelResponse` is intentionally *not* mapped:
    its text was already delivered chunk by chunk, and re-sending it would show
    the reply twice.
    """
    if isinstance(event, ModelMessageChunk):
        return acp.update_agent_message_text(event.content)

    if isinstance(event, ModelReasoning):
        return acp.update_agent_thought_text(event.content) if stream_thoughts else None

    # ToolErrorEvent subclasses ToolResultEvent, so it must be tested first.
    if isinstance(event, ToolErrorEvent):
        return acp.update_tool_call(
            event.parent_id,
            title=event.name,
            status="failed",
            content=[acp.tool_content(acp.text_block(str(event.error)))],
        )

    if isinstance(event, ToolResultEvent):
        return acp.update_tool_call(
            event.parent_id,
            title=event.name,
            status="completed",
            content=[acp.tool_content(acp.text_block(tool_result_text(event.result)))],
        )

    if isinstance(event, ToolCallEvent):
        return acp.start_tool_call(
            event.id,
            title=event.name,
            kind=DEFAULT_TOOL_KIND,
            status="in_progress",
            raw_input=event.serialized_arguments,
        )

    return None


def tool_result_text(result: ToolResult) -> str:
    """Flatten a tool result's parts into the text ACP carries in ``content``.

    A tool returning a non-string (an ``int``, a ``dict``) is coerced by AG2 into
    a :class:`DataInput`, so rendering only :class:`TextInput` would reduce every
    such result to a placeholder. Binary parts *do* get a placeholder — their
    bytes have no faithful text form and must not be smuggled into a text block.
    """
    pieces: list[str] = []
    for part in result.parts:
        if isinstance(part, TextInput):
            pieces.append(part.content)
        elif isinstance(part, DataInput):
            pieces.append(_render_data(part.data))
        else:
            pieces.append(f"[{getattr(part, 'kind', 'binary')}]")
    return "".join(pieces)


def _render_data(data: SendableMessage) -> str:
    """Render a ``DataInput`` payload as text, preferring JSON for structures."""
    if isinstance(data, str):
        return data
    try:
        return json.dumps(data)
    except (TypeError, ValueError):
        return str(data)
