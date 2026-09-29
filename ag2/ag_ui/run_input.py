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
from typing import Any

from ag_ui.core import RunAgentInput
from pydantic import BaseModel

logger = logging.getLogger("ag2.ag_ui")

# The union members 1.0 describes, by the discriminator that names them. Kept by
# hand: a later minor adds members, and that is exactly what this reads as
# unrecognised.
_ROLES = frozenset({"developer", "system", "assistant", "user", "tool", "activity", "reasoning"})
_PART_TYPES = frozenset({"text", "image", "audio", "video", "document"})
_SOURCE_TYPES = frozenset({"data", "url", "file"})
_RESUME_STATUSES = frozenset({"resolved", "cancelled"})

# The messages whose content may be a list of parts.
_PART_ROLES = frozenset({"user", "tool"})


def read_run_input(body: str | bytes) -> RunAgentInput:
    """Parse a request body as the run input it carries.

    Material this server does not recognise is removed, with one `ag2.ag_ui`
    warning per removal naming its path. A body that is not JSON, or that
    carries a known field with a value the schema rejects, raises
    `ValueError` (`pydantic.ValidationError` for the latter).
    """
    raw = json.loads(body)
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


def _strip_unknown_members(raw: dict[str, Any]) -> None:
    # Before validation, since a union member the SDK cannot place fails it.
    # Only a member of an object shape naming a kind this server lacks is taken
    # out: anything else malformed is left for validation to refuse.
    if isinstance(resume := raw.get("resume"), list):
        raw["resume"] = _known_entries(resume)
    messages = raw.get("messages")
    if not isinstance(messages, list):
        return
    kept = []
    for index, message in enumerate(messages):
        role = message.get("role") if isinstance(message, dict) else None
        if isinstance(role, str) and role not in _ROLES:
            _warn(f"/messages/{index}", f"a message of role {role!r}")
            continue
        if role in _PART_ROLES and isinstance(content := message.get("content"), list):
            message["content"] = _known_parts(content, f"/messages/{index}/content")
        kept.append(message)
    raw["messages"] = kept


def _known_entries(entries: list[Any]) -> list[Any]:
    # `status` is required, so an entry whose status is unknown goes whole. The
    # interrupt it answered is then uncovered, which the exchange refuses.
    kept = []
    for index, entry in enumerate(entries):
        status = entry.get("status") if isinstance(entry, dict) else None
        if isinstance(status, str) and status not in _RESUME_STATUSES:
            _warn(f"/resume/{index}", f"a resume entry of status {status!r}")
            continue
        kept.append(entry)
    return kept


def _known_parts(parts: list[Any], path: str) -> list[Any]:
    kept = []
    for index, part in enumerate(parts):
        kind = part.get("type") if isinstance(part, dict) else None
        if isinstance(kind, str) and kind not in _PART_TYPES:
            _warn(f"{path}/{index}", f"a content part of type {kind!r}")
            continue
        # A part left without its source would be malformed, so the part goes whole.
        source = part.get("source") if isinstance(part, dict) else None
        source_kind = source.get("type") if isinstance(source, dict) else None
        if isinstance(source_kind, str) and source_kind not in _SOURCE_TYPES:
            _warn(f"{path}/{index}", f"a {kind} part whose source is of type {source_kind!r}")
            continue
        kept.append(part)
    return kept


def _strip_extras(value: object, path: str) -> None:
    if isinstance(value, BaseModel):
        extra = value.__pydantic_extra__ or {}
        for key in list(extra):
            _warn(f"{path}/{key}", "a property")
            del extra[key]
        for name, info in type(value).model_fields.items():
            _strip_extras(getattr(value, name), f"{path}/{info.alias or name}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _strip_extras(item, f"{path}/{index}")


def _warn(path: str, what: str) -> None:
    logger.warning("stripping %s at %s from an AG-UI run input: this server does not recognise it", what, path)
