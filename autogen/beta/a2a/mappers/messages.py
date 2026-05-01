# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from a2a.types import Message, Part, Role

from autogen.beta.events import ModelRequest
from autogen.beta.events.input_events import Input

from ._proto import dict_to_struct, struct_to_dict
from .parts import a2a_parts_to_inputs, inputs_to_a2a_parts


def model_request_to_a2a_message(
    req: ModelRequest,
    *,
    context_id: str | None,
    task_id: str | None = None,
    extensions: Iterable[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Message:
    """Build an A2A user ``Message`` from a beta ``ModelRequest``."""
    return Message(
        role=Role.ROLE_USER,
        parts=inputs_to_a2a_parts(req.parts),
        message_id=uuid4().hex,
        context_id=context_id or "",
        task_id=task_id or "",
        extensions=list(extensions) if extensions else [],
        metadata=dict_to_struct(metadata),
    )


def a2a_message_to_inputs(msg: Message) -> list[Input]:
    """Extract beta ``Input``s from an A2A ``Message``."""
    return a2a_parts_to_inputs(msg.parts)


def text_from_message(message: Message) -> str:
    """Concatenate every text part body in a message."""
    return text_from_parts(message.parts)


def text_from_parts(parts: Iterable[Part]) -> str:
    """Concatenate every text part body in a sequence of parts (other parts ignored)."""
    return "".join(p.text for p in parts if p.WhichOneof("content") == "text")


def text_parts(text: str) -> list[Part]:
    """Wrap a string into a single-element parts list."""
    return [Part(text=text)]


def message_metadata(message: Message) -> dict[str, Any]:
    """Decode the ``Struct`` metadata of a Message back to a Python dict."""
    return struct_to_dict(message.metadata) if message.HasField("metadata") else {}


def input_required_message(
    text: str,
    *,
    context_id: str,
    task_id: str,
    metadata: dict[str, Any] | None = None,
) -> Message:
    """Construct an agent message asking the client for human input or a client-side tool result."""
    return Message(
        role=Role.ROLE_AGENT,
        parts=text_parts(text),
        message_id=uuid4().hex,
        context_id=context_id,
        task_id=task_id,
        metadata=dict_to_struct(metadata),
    )


def followup_user_message(
    text: str,
    *,
    context_id: str,
    task_id: str,
    metadata: dict[str, Any] | None = None,
) -> Message:
    """Construct a follow-up user message that resumes a suspended task."""
    return Message(
        role=Role.ROLE_USER,
        parts=text_parts(text),
        message_id=uuid4().hex,
        context_id=context_id,
        task_id=task_id,
        metadata=dict_to_struct(metadata),
    )
