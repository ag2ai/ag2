# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from ._types import JsonObject, JsonValue
from .actions import A2UIAction
from .constants import A2UI_JSON_CLOSE_TAG, A2UI_JSON_OPEN_TAG


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


# Cap on client-supplied text spliced into an LLM prompt, bounding the blast
# radius of a malicious or runaway payload.
_MAX_PROMPT_FIELD_LEN = 4000

# Framing markers a hostile client could echo to forge A2UI output or to
# impersonate a conversation role; neutralized by ``sanitize_for_prompt``.
_INJECTION_MARKERS = (A2UI_JSON_OPEN_TAG, A2UI_JSON_CLOSE_TAG)

_ROLE_MARKER_RE = re.compile(r"(?im)^[ \t]*(system|assistant|user|developer|tool)[ \t]*:")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _neutralize_role_marker(match: re.Match[str]) -> str:
    # Replace the ASCII colon with a full-width colon so a line like
    # ``system: ignore previous`` can't read as a role turn while staying legible.
    return match.group(0).replace(":", "：")


def sanitize_for_prompt(text: str) -> str:
    """Best-effort neutralization of client text spliced into an LLM prompt.

    Client-supplied strings (action name, error message, free-text context)
    flow into the agent's prompt when an A2UI ``action``/``error`` envelope is
    rewritten as a corrective instruction. A hostile client could try to smuggle
    prompt-injection — forge the ``<a2ui-json>`` framing, impersonate a
    ``system:``/``assistant:`` turn, or bloat the prompt. This strips control
    characters, defuses those markers, and caps length.

    This is defense-in-depth, **not** a guarantee: it does not make the prompt
    injection-proof. Treat all client text as untrusted and pair this with
    LLM-side and tool-side authorization checks.
    """
    if not text:
        return ""
    cleaned = _CONTROL_CHARS_RE.sub(" ", text)
    for marker in _INJECTION_MARKERS:
        # Swap the angle brackets for look-alike glyphs so the tag can't re-open framing.
        cleaned = cleaned.replace(marker, marker.replace("<", "‹").replace(">", "›"))
    cleaned = _ROLE_MARKER_RE.sub(_neutralize_role_marker, cleaned)
    if len(cleaned) > _MAX_PROMPT_FIELD_LEN:
        cleaned = cleaned[:_MAX_PROMPT_FIELD_LEN] + "…[truncated]"
    return cleaned


def action_to_prompt(action: A2UIIncomingAction, action_def: A2UIAction | None) -> str | None:
    """Rewrite a client A2UI ``action`` as an LLM instruction.

    ``action_def`` is the matching :class:`A2UIAction` the caller resolved from
    the agent's registry (e.g. ``agent.get_action(action.name)``). Returns
    ``None`` when the action cannot be safely mapped — no name, or no matching
    registration — so callers drop the part instead of leaking raw, unvetted
    client data into the prompt. All interpolated client values pass through
    :func:`sanitize_for_prompt`; the registered action's own fields
    (``tool_name``/``description``) are developer-provided and trusted.
    """
    if not action.name or action_def is None:
        return None

    name = sanitize_for_prompt(action.name)
    ctx_json = sanitize_for_prompt(json.dumps(action.context))
    origin_bits: list[str] = []
    if action.surface_id:
        origin_bits.append(f"surface={sanitize_for_prompt(action.surface_id)}")
    if action.source_component_id:
        origin_bits.append(f"component={sanitize_for_prompt(action.source_component_id)}")
    if action.timestamp:
        origin_bits.append(f"at={sanitize_for_prompt(action.timestamp)}")
    origin = f" ({', '.join(origin_bits)})" if origin_bits else ""

    if action_def.tool_name:
        return (
            f"The user clicked the '{name}' button{origin}. "
            f"Call the tool '{action_def.tool_name}' with arguments: {ctx_json}. "
            "Do not respond with text only."
        )
    desc = f" {action_def.description}" if action_def.description else ""
    return f"The user clicked the '{name}' button{origin}.{desc} Context: {ctx_json}"


def error_to_prompt(err: A2UIIncomingError) -> str:
    """Rewrite a client-reported A2UI ``error`` as a corrective LLM instruction."""
    path_hint = sanitize_for_prompt(err.path) if err.path else "(unknown)"
    code = sanitize_for_prompt(err.code) if err.code else "(none)"
    surface = sanitize_for_prompt(err.surface_id)
    message = sanitize_for_prompt(err.message)
    return (
        f"The client reported an A2UI error on surface '{surface}'. "
        f"Code: {code}. Path: {path_hint}. Message: {message}. "
        "Please regenerate the UI with this issue corrected."
    )


__all__ = (
    "A2UIIncomingAction",
    "A2UIIncomingError",
    "A2UIIncomingParseResult",
    "action_to_prompt",
    "error_to_prompt",
    "parse_incoming_message",
    "sanitize_for_prompt",
)
