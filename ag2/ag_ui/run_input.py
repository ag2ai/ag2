# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Reading a run input the way AG-UI 1.0's processing model says to.

Unrecognised material is not an error, and a malformed known value is. What a
newer client sends that this server does not describe — a property, a message
role, a content part kind or source kind, a resume status — is stripped with a
warning, and the run is served.
"""

import json
import logging
from typing import Literal, TypeAlias, get_args

from ag_ui.core import ResumeStatus, Role, RunAgentInput
from pydantic import BaseModel, JsonValue

logger = logging.getLogger("ag2.ag_ui")

_PartType: TypeAlias = Literal["text", "image", "audio", "video", "document"]
"""The `type` of each `ContentPart` member."""

_SourceType: TypeAlias = Literal["data", "url", "file"]
"""The `type` of each `PartSource` member."""

_PART_ROLES: tuple[Role, ...] = ("user", "tool")
"""The messages whose content may be a list of parts."""


def read_run_input(body: str | bytes) -> RunAgentInput:
    """Parse a request body as the run input it carries.

    Material this server does not recognise is removed, with one `ag2.ag_ui`
    warning per removal naming its path. A body that is not JSON, or that
    carries a known field with a value the schema rejects, raises
    `ValueError` (`pydantic.ValidationError` for the latter).
    """
    raw: JsonValue = json.loads(body)
    if isinstance(raw, dict):
        _strip_unknown_members(raw)
    return strip_unrecognised(RunAgentInput.model_validate(raw))


def strip_unrecognised(incoming: RunAgentInput) -> RunAgentInput:
    """Remove the properties `incoming` carries that the protocol does not describe, in place.

    The SDK's models keep unknown properties for whoever reads them next; this
    is the stage that must not. Open objects — `state`, `forwardedProps`,
    `metadata`, a tool's `parameters` — are values rather than models, and are
    kept whole.
    """
    _strip_extras(incoming, "")
    return incoming


def _strip_unknown_members(raw: dict[str, JsonValue]) -> None:
    # Before validation, since a union member the SDK cannot place fails it.
    # Only a member of an object shape naming a kind this server lacks is taken
    # out: anything else malformed is left for validation to refuse.
    if isinstance(resume := raw.get("resume"), list):
        raw["resume"] = _known_entries(resume)
    if isinstance(messages := raw.get("messages"), list):
        raw["messages"] = _known_messages(messages)


def _known_messages(messages: list[JsonValue]) -> list[JsonValue]:
    kept: list[JsonValue] = []
    for index, message in enumerate(messages):
        role = _tag(message, "role")
        if role is not None and role not in get_args(Role):
            _warn(f"/messages/{index}", f"a message of role {role!r}")
            continue
        if isinstance(message, dict) and role in _PART_ROLES and isinstance(content := message.get("content"), list):
            message["content"] = _known_parts(content, f"/messages/{index}/content")
        kept.append(message)
    return kept


def _known_entries(entries: list[JsonValue]) -> list[JsonValue]:
    # `status` is required, so an entry whose status is unknown goes whole. The
    # interrupt it answered is then uncovered, which the exchange refuses.
    kept: list[JsonValue] = []
    for index, entry in enumerate(entries):
        status = _tag(entry, "status")
        if status is not None and status not in get_args(ResumeStatus):
            _warn(f"/resume/{index}", f"a resume entry of status {status!r}")
            continue
        kept.append(entry)
    return kept


def _known_parts(parts: list[JsonValue], path: str) -> list[JsonValue]:
    kept: list[JsonValue] = []
    for index, part in enumerate(parts):
        kind = _tag(part, "type")
        if kind is not None and kind not in get_args(_PartType):
            _warn(f"{path}/{index}", f"a content part of type {kind!r}")
            continue
        # A part left without its source would be malformed, so the part goes whole.
        source_kind = _tag(part.get("source") if isinstance(part, dict) else None, "type")
        if source_kind is not None and source_kind not in get_args(_SourceType):
            _warn(f"{path}/{index}", f"a {kind} part whose source is of type {source_kind!r}")
            continue
        kept.append(part)
    return kept


def _tag(value: JsonValue, discriminator: str) -> str | None:
    """The kind `value` names, when it is an object naming one with a string."""
    tag = value.get(discriminator) if isinstance(value, dict) else None
    return tag if isinstance(tag, str) else None


def _strip_extras(model: BaseModel, path: str) -> None:
    extra = model.__pydantic_extra__ or {}
    for key in list(extra):
        _warn(f"{path}/{key}", "a property")
        del extra[key]
    for name, info in type(model).model_fields.items():
        field_path = f"{path}/{info.alias or name}"
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            _strip_extras(value, field_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, BaseModel):
                    _strip_extras(item, f"{field_path}/{index}")


def _warn(path: str, what: str) -> None:
    logger.warning("stripping %s at %s from an AG-UI run input: this server does not recognise it", what, path)
