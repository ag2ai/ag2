# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Literal

from ._types import JsonObject, JsonValue


@dataclass(slots=True)
class A2UIIncomingAction:
    """A client→server ``action`` envelope content.

    Mirrors ``client_to_server.json#/properties/action``.
    """

    name: str
    surface_id: str
    source_component_id: str
    timestamp: str
    context: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(slots=True)
class A2UIIncomingError:
    """A client→server ``error`` envelope content.

    Mirrors ``client_to_server.json#/properties/error``. ``path`` is set
    only for ``VALIDATION_FAILED`` errors per the spec.
    """

    code: str
    surface_id: str
    message: str
    path: str | None = None


@dataclass(slots=True)
class A2UIIncomingParseResult:
    """Result of parsing one client→server envelope."""

    kind: Literal["action", "error", "unknown"]
    action: A2UIIncomingAction | None = None
    error: A2UIIncomingError | None = None
    raw: JsonObject | None = None
    parse_error: str | None = None


def parse_incoming_message(data: Any) -> A2UIIncomingParseResult:
    """Classify a single client→server A2UI envelope.

    Accepts a dict shaped like ``{"version": "v0.9", "action": {...}}`` or
    ``{"version": "v0.9", "error": {...}}`` and returns a typed result.
    Does NOT validate the envelope against the JSON schema — use
    ``A2UISchemaManager.client_to_server_schema`` for strict validation.
    """
    if not isinstance(data, dict):
        return A2UIIncomingParseResult(kind="unknown", parse_error="envelope is not a JSON object")

    action_obj = data.get("action")
    if isinstance(action_obj, dict):
        return A2UIIncomingParseResult(
            kind="action",
            action=A2UIIncomingAction(
                name=str(action_obj.get("name", "")),
                surface_id=str(action_obj.get("surfaceId", "")),
                source_component_id=str(action_obj.get("sourceComponentId", "")),
                timestamp=str(action_obj.get("timestamp", "")),
                context=dict(action_obj.get("context") or {}),
            ),
            raw=data,
        )

    error_obj = data.get("error")
    if isinstance(error_obj, dict):
        raw_path = error_obj.get("path")
        return A2UIIncomingParseResult(
            kind="error",
            error=A2UIIncomingError(
                code=str(error_obj.get("code", "")),
                surface_id=str(error_obj.get("surfaceId", "")),
                message=str(error_obj.get("message", "")),
                path=str(raw_path) if isinstance(raw_path, str) else None,
            ),
            raw=data,
        )

    return A2UIIncomingParseResult(
        kind="unknown",
        raw=data,
        parse_error="envelope has neither 'action' nor 'error' key",
    )
