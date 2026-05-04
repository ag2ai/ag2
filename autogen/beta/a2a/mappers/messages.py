# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from a2a.types import Message, Part, Role

from autogen.beta.events import Input, ToolCallEvent, ToolResultEvent
from autogen.beta.tools.final.function_tool import FunctionToolSchema

from ..extension import (
    CONTEXT_UPDATE_METADATA_KEY,
    EXTENSION_URI,
    MIME_TOOL_CALL,
    MIME_TOOL_RESULT,
    MIME_TOOL_SCHEMAS,
)
from .parts import (
    data_part,
    input_to_part,
    is_data_part_with_mime,
    part_data_to_python,
    part_to_input,
    struct_from_dict,
    struct_to_dict,
)
from .tools import (
    call_to_payload,
    payload_to_call,
    payload_to_results,
    payload_to_schemas,
    results_to_payload,
    schemas_to_payload,
)


@dataclass(slots=True)
class ParsedMessage:
    """Result of decoding an incoming A2A ``Message``.

    Each list contains *only* parts of the corresponding kind. Callers
    pick whichever they need.

    ``context_update`` carries ``context.variables`` deltas piggy-backed
    on ``Message.metadata`` (server -> client on a finalize, client ->
    server on a user turn).

    ``reference_task_ids`` mirrors ``Message.reference_task_ids`` —
    spec-level field used by callers to link a message to additional
    task ids (typically for inter-task coordination).
    """

    inputs: list[Input] = field(default_factory=list)
    tool_schemas: list[FunctionToolSchema] = field(default_factory=list)
    tool_calls: list[ToolCallEvent] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    context_update: dict[str, Any] = field(default_factory=dict)
    reference_task_ids: list[str] = field(default_factory=list)


def build_user_message(
    inputs: Iterable[Input],
    *,
    tool_schemas: Sequence[FunctionToolSchema] = (),
    task_id: str | None = None,
    context_id: str | None = None,
    message_id: str | None = None,
    advertise_extension: bool = False,
    context_update: Mapping[str, Any] | None = None,
    reference_task_ids: Sequence[str] = (),
    extra_parts: Sequence[Part] = (),
) -> Message:
    """Build a ``Message`` from an AG2 client (``role=ROLE_USER``).

    ``tool_schemas`` is included as a ``tool-schemas+json`` data Part —
    this is how the client tells the server which tools live on the
    calling side. ``context_update``, when provided, is attached to
    ``Message.metadata`` so the server can sync into its
    ``context.variables``. ``extra_parts`` are appended as-is, useful for
    extension data that doesn't have a dedicated builder argument.
    """
    parts: list[Part] = [input_to_part(inp) for inp in inputs]
    if tool_schemas:
        parts.append(data_part(schemas_to_payload(tool_schemas), media_type=MIME_TOOL_SCHEMAS))
    parts.extend(extra_parts)
    return _build_message(
        parts,
        role=Role.ROLE_USER,
        task_id=task_id,
        context_id=context_id,
        message_id=message_id,
        advertise_extension=advertise_extension,
        context_update=context_update,
        reference_task_ids=reference_task_ids,
    )


def build_tool_result_message(
    results: Iterable[ToolResultEvent],
    *,
    task_id: str,
    context_id: str | None = None,
    message_id: str | None = None,
    context_update: Mapping[str, Any] | None = None,
    reference_task_ids: Sequence[str] = (),
) -> Message:
    """Build a ``Message`` carrying tool results back to the server.

    Used on the second leg of a client-side tool round-trip — after the
    AG2 outer loop locally executed the tools the server requested.
    """
    parts = [data_part(results_to_payload(results), media_type=MIME_TOOL_RESULT)]
    return _build_message(
        parts,
        role=Role.ROLE_USER,
        task_id=task_id,
        context_id=context_id,
        message_id=message_id,
        advertise_extension=True,
        context_update=context_update,
        reference_task_ids=reference_task_ids,
    )


def build_input_response_message(
    text: str,
    *,
    task_id: str,
    context_id: str | None = None,
    message_id: str | None = None,
    context_update: Mapping[str, Any] | None = None,
    reference_task_ids: Sequence[str] = (),
) -> Message:
    """Build a continuation ``Message`` carrying a HITL response back.

    Sent after the server transitioned the task to
    ``TASK_STATE_INPUT_REQUIRED``: we wrap the user's reply as a single
    text Part and reuse the existing ``task_id`` so the server can
    resume the same task.
    """
    return _build_message(
        [Part(text=text)],
        role=Role.ROLE_USER,
        task_id=task_id,
        context_id=context_id,
        message_id=message_id,
        advertise_extension=False,
        context_update=context_update,
        reference_task_ids=reference_task_ids,
    )


def build_agent_message(
    *,
    text: str = "",
    tool_calls: Iterable[ToolCallEvent] = (),
    additional_parts: Sequence[Part] = (),
    task_id: str | None = None,
    context_id: str | None = None,
    message_id: str | None = None,
    context_update: Mapping[str, Any] | None = None,
    reference_task_ids: Sequence[str] = (),
) -> Message:
    """Build a ``Message`` produced by the agent (``role=ROLE_AGENT``).

    Used by the server when emitting a final agent message that contains
    text and/or pending client-side tool invocations. ``context_update``
    rides on ``Message.metadata`` for variables sync into the client.
    """
    parts: list[Part] = []
    if text:
        parts.append(Part(text=text))
    for call in tool_calls:
        parts.append(data_part(call_to_payload(call), media_type=MIME_TOOL_CALL))
    parts.extend(additional_parts)
    return _build_message(
        parts,
        role=Role.ROLE_AGENT,
        task_id=task_id,
        context_id=context_id,
        message_id=message_id,
        advertise_extension=bool(tool_calls),
        context_update=context_update,
        reference_task_ids=reference_task_ids,
    )


def parse_message(msg: Message) -> ParsedMessage:
    """Decode an incoming ``Message`` into AG2-shaped buckets.

    Walks the parts in order; extension data parts are routed to the
    matching bucket, everything else becomes an ``Input``. Picks up
    ``context_update`` from ``Message.metadata`` if present.
    """
    parsed = ParsedMessage()
    for part in msg.parts:
        if is_data_part_with_mime(part, MIME_TOOL_SCHEMAS):
            parsed.tool_schemas.extend(payload_to_schemas(part_data_to_python(part)))
            continue
        if is_data_part_with_mime(part, MIME_TOOL_CALL):
            parsed.tool_calls.append(payload_to_call(part_data_to_python(part)))
            continue
        if is_data_part_with_mime(part, MIME_TOOL_RESULT):
            parsed.tool_results.extend(payload_to_results(part_data_to_python(part)))
            continue
        parsed.inputs.append(part_to_input(part))
    parsed.context_update = extract_context_update(msg)
    parsed.reference_task_ids = list(msg.reference_task_ids)
    return parsed


def extract_context_update(msg: Message) -> dict[str, Any]:
    """Extract the ``ag2.context_update`` payload from ``Message.metadata``.

    Returns an empty dict when nothing is set — callers can ``.update``
    a real ``context.variables`` with the result unconditionally.
    """
    if not msg.HasField("metadata"):
        return {}
    metadata = struct_to_dict(msg.metadata)
    payload = metadata.get(CONTEXT_UPDATE_METADATA_KEY)
    if isinstance(payload, dict):
        return payload
    return {}


def _build_message(
    parts: Sequence[Part],
    *,
    role: int,
    task_id: str | None,
    context_id: str | None,
    message_id: str | None,
    advertise_extension: bool,
    context_update: Mapping[str, Any] | None,
    reference_task_ids: Sequence[str] = (),
) -> Message:
    kwargs: dict[str, Any] = {
        "role": role,
        "parts": list(parts),
        "message_id": message_id or uuid4().hex,
    }
    if task_id:
        kwargs["task_id"] = task_id
    if context_id:
        kwargs["context_id"] = context_id
    if advertise_extension:
        kwargs["extensions"] = [EXTENSION_URI]
    if context_update:
        kwargs["metadata"] = struct_from_dict({CONTEXT_UPDATE_METADATA_KEY: dict(context_update)})
    if reference_task_ids:
        kwargs["reference_task_ids"] = list(reference_task_ids)
    return Message(**kwargs)
