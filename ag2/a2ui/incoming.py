# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import assert_never

from ._formats import is_date_time
from ._messages import CLIENT_TO_SERVER_KEYS
from ._types import A2UIVersion, JsonObject, JsonValue
from ._versions import A2UIProtocol, get_protocol
from .actions import A2UIEventAction
from .constants import A2UI_JSON_CLOSE_TAG, A2UI_JSON_OPEN_TAG

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class A2UIIncomingAction:
    """A client→server ``action`` envelope content.

    Mirrors ``client_to_server.json#/properties/action`` (``renderer_to_agent.json``
    in v1.0). ``user_message`` and ``metadata`` are v1.0 additions; an action is
    fire-and-forget — a renderer that wants a reply sends a ``callAgentFunction``.
    """

    name: str
    surface_id: str
    source_component_id: str
    timestamp: str
    context: dict[str, JsonValue] = field(default_factory=dict)
    user_message: str | None = None
    metadata: dict[str, JsonValue] | None = None

    @property
    def arguments(self) -> dict[str, JsonValue]:
        """The keyword arguments a server action handler is run with."""
        return self.context


@dataclass(slots=True, frozen=True)
class A2UIIncomingFunctionCall:
    """A client→server ``callAgentFunction`` envelope content (v1.0).

    The renderer asks the agent to run the function ``call`` with ``args``. The
    agent must answer with an ``agentFunctionResponse`` carrying the same
    ``function_call_id`` — so, unlike the other envelopes, a parsed call always has
    a usable id and function name (a malformed one parses as unknown instead).
    ``args`` preserves the wire value so a non-object can be rejected without
    invoking the handler; only an omitted ``args`` defaults to an empty object.
    ``catalog_id_valid`` distinguishes an omitted catalog from an explicitly
    malformed override, including JSON null, so malformed input cannot select a
    handler by silently falling back to the active catalog.

    An omitted ``catalog_id`` is tolerated on purpose: the call resolves against
    the agent's catalog. Upstream (v1.0 ``a2ui_protocol.md``, "callAgentFunction")
    puts the MUST to send ``catalogId`` on the renderer, and
    ``renderer_to_agent.json`` does not require the field, so the agent does not
    reject a call that leaves it out.
    """

    function_call_id: str
    surface_id: str
    call: str
    args: JsonValue = field(default_factory=dict)
    catalog_id: str | None = None
    catalog_id_valid: bool = True

    @property
    def arguments(self) -> dict[str, JsonValue]:
        """The keyword arguments a server action handler is run with."""
        if not isinstance(self.args, dict):
            raise ValueError("Function arguments must be a JSON object")
        return self.args


@dataclass(slots=True, frozen=True)
class A2UIIncomingSurfaceError:
    """A client→server ``error`` reporting a surface validation/render failure.

    Mirrors the surface arm of ``client_to_server.json#/properties/error``.
    ``path`` (a JSON pointer) is set only for ``VALIDATION_FAILED`` errors.
    """

    code: str
    surface_id: str
    message: str
    path: str | None = None


@dataclass(slots=True, frozen=True)
class A2UIIncomingFunctionError:
    """A client→server ``error`` reporting a failed server-initiated function call (v1.0).

    The generic-error arm of ``renderer_to_agent.json#/properties/error``: it
    carries a ``function_call_id`` instead of a surface. Splitting this from
    :class:`A2UIIncomingSurfaceError` makes the spec's "surface XOR functionCall"
    rule structural — neither class can hold the other's correlation field.
    """

    code: str
    function_call_id: str
    message: str


# A client→server ``error`` envelope. Tagged union over the two mutually
# exclusive arms; branch with ``isinstance``.
A2UIIncomingError = A2UIIncomingSurfaceError | A2UIIncomingFunctionError


@dataclass(slots=True, frozen=True)
class A2UIFunctionError:
    """The ``error`` arm of a ``FunctionResponse`` (v1.0): why a function call failed."""

    code: str
    message: str


@dataclass(slots=True, frozen=True)
class A2UIIncomingFunctionResponse:
    """A client→server ``rendererFunctionResponse`` envelope content (v1.0).

    The client's reply to a server-initiated ``callRendererFunction``. Mirrors
    ``common_types.json#/$defs/FunctionResponse``; ``function_call_id`` is copied
    verbatim from the originating call. A response carries a ``value`` or an
    ``error`` — ``error`` is set exactly when the call failed.
    """

    function_call_id: str
    value: JsonValue = None
    error: A2UIFunctionError | None = None


@dataclass(slots=True, frozen=True)
class A2UIIncomingActionResult:
    """A successfully parsed client→server ``action`` envelope."""

    action: A2UIIncomingAction


@dataclass(slots=True, frozen=True)
class A2UIIncomingFunctionCallResult:
    """A successfully parsed client→server ``callAgentFunction`` envelope (v1.0)."""

    function_call: A2UIIncomingFunctionCall


@dataclass(slots=True, frozen=True)
class A2UIIncomingErrorResult:
    """A successfully parsed client→server ``error`` envelope."""

    error: A2UIIncomingError


@dataclass(slots=True, frozen=True)
class A2UIIncomingFunctionResponseResult:
    """A successfully parsed client→server ``rendererFunctionResponse`` envelope (v1.0)."""

    function_response: A2UIIncomingFunctionResponse


@dataclass(slots=True, frozen=True)
class A2UIIncomingInvalidFunctionCallResult:
    """A malformed ``callAgentFunction`` (v1.0) that still carries a ``functionCallId`` to answer.

    The renderer is waiting on that id, and "if parameter validation fails on the server, the agent MUST return an
    ``agentFunctionResponse`` message containing an ``error`` payload" (v1.0 a2ui_protocol.md, "callAgentFunction"),
    so the server answers it with ``INVALID_FUNCTION_CALL`` instead of dropping it. ``reason`` says what is wrong.
    """

    function_call_id: str
    reason: str


@dataclass(slots=True, frozen=True)
class A2UIIncomingUnknownResult:
    """An envelope that could not be classified; ``parse_error`` says why."""

    parse_error: str


# Result of parsing one client→server envelope. A discriminated union over the
# mutually exclusive outcomes — each arm carries only its own payload, so
# the old "kind says action but action is None" states are unrepresentable.
# Branch with ``isinstance`` (mirrors the ``A2UIIncomingError`` union above).
A2UIIncomingParseResult = (
    A2UIIncomingActionResult
    | A2UIIncomingFunctionCallResult
    | A2UIIncomingInvalidFunctionCallResult
    | A2UIIncomingErrorResult
    | A2UIIncomingFunctionResponseResult
    | A2UIIncomingUnknownResult
)


def _optional_str(value: JsonValue) -> str | None:
    return value if isinstance(value, str) else None


def _parse_function_call(obj: dict[str, JsonValue]) -> A2UIIncomingParseResult:
    function_call_id = obj.get("functionCallId")
    call_obj = obj.get("callFunction")
    name = call_obj.get("@call") if isinstance(call_obj, dict) else None
    if not isinstance(function_call_id, str):
        return A2UIIncomingUnknownResult(parse_error="'callAgentFunction' has no 'functionCallId' to correlate a reply")
    if not isinstance(call_obj, dict) or not isinstance(name, str):
        return A2UIIncomingUnknownResult(parse_error="'callAgentFunction' has no 'callFunction' with an '@call' name")
    surface_id = obj.get("surfaceId")
    if not isinstance(surface_id, str):
        return A2UIIncomingUnknownResult(parse_error="'callAgentFunction' has no string 'surfaceId'")
    raw_args = call_obj.get("args", {})
    return A2UIIncomingFunctionCallResult(
        function_call=A2UIIncomingFunctionCall(
            function_call_id=function_call_id,
            surface_id=surface_id,
            call=name,
            # Preserve malformed input so it cannot silently run a handler with defaults.
            args=dict(raw_args) if isinstance(raw_args, dict) else raw_args,
            catalog_id=_optional_str(call_obj.get("catalogId")),
            catalog_id_valid="catalogId" not in call_obj or isinstance(call_obj["catalogId"], str),
        ),
    )


def _parse_function_response(obj: dict[str, JsonValue]) -> A2UIIncomingParseResult:
    function_call_id = obj.get("functionCallId")
    if not isinstance(function_call_id, str):
        return A2UIIncomingUnknownResult(
            parse_error="'rendererFunctionResponse' has no 'functionCallId' to correlate with a call",
        )
    if ("value" in obj) == ("error" in obj):
        return A2UIIncomingUnknownResult(
            parse_error="'rendererFunctionResponse' must carry exactly one of 'value' or 'error'",
        )
    error: A2UIFunctionError | None = None
    if "error" in obj:
        error_obj = obj["error"]
        code = error_obj.get("code") if isinstance(error_obj, dict) else None
        message = error_obj.get("message") if isinstance(error_obj, dict) else None
        if not isinstance(code, str) or not isinstance(message, str):
            return A2UIIncomingUnknownResult(
                parse_error="'rendererFunctionResponse' error must be an object with string 'code' and 'message'",
            )
        error = A2UIFunctionError(code=code, message=message)
    return A2UIIncomingFunctionResponseResult(
        function_response=A2UIIncomingFunctionResponse(
            function_call_id=function_call_id,
            value=obj.get("value"),
            error=error,
        ),
    )


def parse_incoming_message(data: JsonValue) -> A2UIIncomingParseResult:
    """Classify a single client→server A2UI envelope.

    Accepts a dict shaped like ``{"version": ..., "action": {...}}``,
    ``{"version": ..., "callAgentFunction": {...}}``,
    ``{"version": ..., "rendererFunctionResponse": {...}}``, or
    ``{"version": ..., "error": {...}}`` and returns a typed result. An envelope
    with more than one of those keys is unknown: upstream requires exactly one.
    Parsing is structural and version-neutral — v1.0-only arms
    (``callAgentFunction``, ``rendererFunctionResponse``, ``action.userMessage``,
    ``error.functionCallId``) decode when present; :func:`decode_envelopes` drops
    the envelopes that do not conform to the served protocol (see :func:`envelope_problem`).
    """
    if not isinstance(data, dict):
        return A2UIIncomingUnknownResult(parse_error="envelope is not a JSON object")

    present = [key for key in CLIENT_TO_SERVER_KEYS if key in data]
    if len(present) > 1:
        return A2UIIncomingUnknownResult(
            parse_error=f"envelope must carry exactly one of {', '.join(map(repr, CLIENT_TO_SERVER_KEYS))}; "
            f"it carries {', '.join(map(repr, present))}",
        )

    action_obj = data.get("action")
    if isinstance(action_obj, dict):
        raw_context = action_obj.get("context")
        raw_metadata = action_obj.get("metadata")
        return A2UIIncomingActionResult(
            action=A2UIIncomingAction(
                name=str(action_obj.get("name", "")),
                surface_id=str(action_obj.get("surfaceId", "")),
                source_component_id=str(action_obj.get("sourceComponentId", "")),
                timestamp=str(action_obj.get("timestamp", "")),
                context=dict(raw_context) if isinstance(raw_context, dict) else {},
                user_message=_optional_str(action_obj.get("userMessage")),
                metadata=dict(raw_metadata) if isinstance(raw_metadata, dict) else None,
            ),
        )

    function_call_obj = data.get("callAgentFunction")
    if isinstance(function_call_obj, dict):
        return _parse_function_call(function_call_obj)

    function_response_obj = data.get("rendererFunctionResponse")
    if isinstance(function_response_obj, dict):
        return _parse_function_response(function_response_obj)

    error_obj = data.get("error")
    if isinstance(error_obj, dict):
        code = str(error_obj.get("code", ""))
        message = str(error_obj.get("message", ""))
        raw_fc_id = error_obj.get("functionCallId")
        error: A2UIIncomingError
        if isinstance(raw_fc_id, str):
            # functionCallId present → the generic (function) error arm.
            error = A2UIIncomingFunctionError(code=code, function_call_id=raw_fc_id, message=message)
        else:
            raw_path = error_obj.get("path")
            error = A2UIIncomingSurfaceError(
                code=code,
                surface_id=str(error_obj.get("surfaceId", "")),
                message=message,
                path=str(raw_path) if isinstance(raw_path, str) else None,
            )
        return A2UIIncomingErrorResult(error=error)

    return A2UIIncomingUnknownResult(
        parse_error="envelope has none of 'action', 'callAgentFunction', 'rendererFunctionResponse', or 'error' keys",
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


def action_to_prompt(action: A2UIIncomingAction) -> str | None:
    """Rewrite a client A2UI ``action`` as a generic LLM instruction.

    Used only for clicks with **no** registered server action: buttons are
    rendered dynamically by the LLM, so the click is rewritten generically and
    the LLM continues the conversation (a registered action is handled on the
    server and never reaches this path). Returns ``None`` only for a nameless
    (malformed) action. All client-supplied values pass through
    :func:`sanitize_for_prompt`.
    """
    if not action.name:
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

    user_message = f" User message: {sanitize_for_prompt(action.user_message)}" if action.user_message else ""

    return f"The user clicked the '{name}' button{origin}. Context: {ctx_json}{user_message}"


def error_to_prompt(err: A2UIIncomingError) -> str:
    """Rewrite a client-reported A2UI ``error`` as a corrective LLM instruction.

    Handles both error arms: a surface-scoped validation/render error, and the
    v1.0 generic error reporting a failed server-initiated ``callRendererFunction``
    (carrying a ``functionCallId`` instead of a surface).
    """
    code = sanitize_for_prompt(err.code) if err.code else "(none)"
    message = sanitize_for_prompt(err.message)

    if isinstance(err, A2UIIncomingFunctionError):
        fc_id = sanitize_for_prompt(err.function_call_id)
        return (
            f"The client reported an A2UI error for function call '{fc_id}'. "
            f"Code: {code}. Message: {message}. "
            "Adjust your approach and continue without relying on that function's result."
        )

    path_hint = sanitize_for_prompt(err.path) if err.path else "(unknown)"
    surface = sanitize_for_prompt(err.surface_id)
    return (
        f"The client reported an A2UI error on surface '{surface}'. "
        f"Code: {code}. Path: {path_hint}. Message: {message}. "
        "Please regenerate the UI with this issue corrected."
    )


def function_response_to_prompt(fr: A2UIIncomingFunctionResponse) -> str:
    """Rewrite a client ``rendererFunctionResponse`` (v1.0) as a continuation instruction.

    The client has run a server-initiated ``callRendererFunction`` and returned
    its result or its failure. All client-supplied values pass through
    :func:`sanitize_for_prompt`.
    """
    fc_id = sanitize_for_prompt(fr.function_call_id)
    if fr.error is not None:
        code = sanitize_for_prompt(fr.error.code) if fr.error.code else "(none)"
        return (
            f"The client's function call (id={fc_id}) failed. Code: {code}. "
            f"Message: {sanitize_for_prompt(fr.error.message)}. "
            "Adjust your approach and continue without relying on that function's result."
        )
    try:
        value_json = sanitize_for_prompt(json.dumps(fr.value))
    except (TypeError, ValueError):
        logger.warning(
            "rendererFunctionResponse value for call '%s' is not JSON-serializable; using a placeholder.", fc_id
        )
        value_json = "<non-serializable>"
    return (
        f"The client executed the function call (id={fc_id}) and returned: {value_json}. Use this result to continue."
    )


_FUNCTION_RPC_ENVELOPES = frozenset({"callAgentFunction", "rendererFunctionResponse"})
"""The envelope kinds that exist only with function RPC (v1.0)."""

_SURFACE_ERROR_CODES_V1 = frozenset({"VALIDATION_FAILED", "UNALLOWED_PARENT", "UNALLOWED_CHILD"})
"""v1.0 error codes that point at a component on a surface, so they carry its ``path`` (``renderer_to_agent.json``)."""


def _missing_strings(obj: JsonObject, *keys: str) -> list[str]:
    return [key for key in keys if not isinstance(obj.get(key), str)]


def _action_problem(action: JsonObject, protocol: A2UIProtocol) -> str | None:
    missing = _missing_strings(action, "name", "surfaceId", "sourceComponentId", "timestamp")
    if missing:
        return f"'action' needs string {', '.join(map(repr, missing))}"
    if not isinstance(action.get("context"), dict):
        return "'action' needs a 'context' object"
    if not is_date_time(str(action["timestamp"])):
        return f"'action.timestamp' {action['timestamp']!r} is not an RFC 3339 date-time"
    if protocol.function_rpc:
        if "userMessage" in action and not isinstance(action["userMessage"], str):
            return "'action.userMessage' must be a string"
        if "metadata" in action:
            return _action_metadata_problem(action["metadata"])
    return None


def _action_metadata_problem(metadata: JsonValue) -> str | None:
    """``action.metadata`` (v1.0) holds only ``extensions``, whose keys "MUST be Unicode identifiers (UAX #31)"."""
    if not isinstance(metadata, dict):
        return "'action.metadata' must be an object"
    extra = sorted(metadata.keys() - {"extensions"})
    if extra:
        return f"'action.metadata' has unknown properties {', '.join(map(repr, extra))}"
    extensions = metadata.get("extensions", {})
    if not isinstance(extensions, dict):
        return "'action.metadata.extensions' must be an object"
    # ``str.isidentifier`` is defined by XID_Start (plus ``_``) and XID_Continue, the sets UAX #31 names.
    invalid = sorted(key for key in extensions if not key.isidentifier())
    if invalid:
        return f"'action.metadata.extensions' keys must be UAX #31 identifiers: {', '.join(map(repr, invalid))}"
    return None


def _function_call_problem(call: JsonObject) -> str | None:
    extra = sorted(call.keys() - {"surfaceId", "functionCallId", "callFunction"})
    if extra:
        return f"'callAgentFunction' has unknown properties {', '.join(map(repr, extra))}"
    if not isinstance(call.get("functionCallId"), str):
        return "'callAgentFunction' needs a string 'functionCallId'"
    if not isinstance(call.get("surfaceId"), str) or not call["surfaceId"]:
        return "'callAgentFunction' needs a non-empty string 'surfaceId'"
    function = call.get("callFunction")
    if not isinstance(function, dict):
        return "'callAgentFunction' needs a 'callFunction' object"
    name = function.get("@call")
    if not isinstance(name, str) or not name:
        return "'callAgentFunction.callFunction' needs a non-empty string '@call'"
    if "catalogId" in function and not isinstance(function["catalogId"], str):
        return f"'catalogId' of function {name!r} must be a string"
    if "args" in function and not isinstance(function["args"], dict):
        return f"'args' of function {name!r} must be an object"
    return None


def _function_response_problem(response: JsonObject) -> str | None:
    if not isinstance(response.get("functionCallId"), str):
        return "'rendererFunctionResponse' needs a string 'functionCallId'"
    extra = sorted(response.keys() - {"functionCallId", "value", "error"})
    if extra:
        return f"'rendererFunctionResponse' has unknown properties {', '.join(map(repr, extra))}"
    if ("value" in response) == ("error" in response):
        return "'rendererFunctionResponse' must carry exactly one of 'value' or 'error'"
    error = response.get("error")
    if "error" in response and (
        not isinstance(error, dict) or _missing_strings(error, "code", "message") or error.keys() - {"code", "message"}
    ):
        return "'rendererFunctionResponse.error' must be an object with exactly string 'code' and 'message'"
    return None


def _error_problem(error: JsonObject, protocol: A2UIProtocol) -> str | None:
    code = error.get("code")
    surface_codes = _SURFACE_ERROR_CODES_V1 if protocol.function_rpc else frozenset({"VALIDATION_FAILED"})
    if code in surface_codes:
        missing = _missing_strings(error, "code", "path", "message", "surfaceId")
        extra = sorted(error.keys() - {"code", "path", "message", "surfaceId"})
        if missing or extra:
            return f"a {code} 'error' carries exactly string 'code', 'path', 'message' and 'surfaceId'"
        return None
    if "code" not in error:
        return "'error' needs a 'code'"
    if not isinstance(error.get("message"), str):
        return "'error' needs a string 'message'"
    if not protocol.function_rpc:
        return None if isinstance(error.get("surfaceId"), str) else "'error' needs a string 'surfaceId'"
    if not isinstance(code, str):
        return "'error' needs a string 'code'"
    has_surface, has_call = "surfaceId" in error, "functionCallId" in error
    if has_surface == has_call:
        return "'error' must carry exactly one of 'surfaceId' or 'functionCallId'"
    if _missing_strings(error, "surfaceId" if has_surface else "functionCallId"):
        return "'error.surfaceId' / 'error.functionCallId' must be a string"
    return None


def envelope_problem(envelope: JsonValue, protocol: A2UIProtocol) -> str | None:
    """Why ``envelope`` is not a renderer→agent message of ``protocol``, or ``None`` when it is one.

    Checks what the vendored ``client_to_server.json`` (``renderer_to_agent.json`` in v1.0) requires: exactly
    ``version`` and one payload, the version's ``version`` value, the payload kinds the version has, and each
    payload's required fields, types and arms. The function a ``callAgentFunction`` names is not checked against a
    catalog here: an ``@a2ui_action`` handler is not a catalog function.
    """
    if not isinstance(envelope, dict):
        return "envelope is not a JSON object"
    present = [key for key in CLIENT_TO_SERVER_KEYS if key in envelope]
    if len(present) != 1:
        found = ", ".join(map(repr, present)) or "none"
        return f"envelope must carry exactly one of {', '.join(map(repr, CLIENT_TO_SERVER_KEYS))}; it carries {found}"
    kind = present[0]
    if kind in _FUNCTION_RPC_ENVELOPES and not protocol.function_rpc:
        return f"'{kind}' exists only in protocol v1.0"
    extra = sorted(envelope.keys() - {"version", kind})
    if extra:
        return f"envelope has properties besides 'version' and '{kind}': {', '.join(map(repr, extra))}"
    version = envelope.get("version")
    if version not in protocol.accepted_wire_versions:
        accepted = " or ".join(map(repr, protocol.accepted_wire_versions))
        return f"envelope 'version' is {version!r}; this server speaks {accepted}"
    payload = envelope[kind]
    if not isinstance(payload, dict):
        return f"'{kind}' is not an object"
    if kind == "action":
        return _action_problem(payload, protocol)
    if kind == "callAgentFunction":
        return _function_call_problem(payload)
    if kind == "rendererFunctionResponse":
        return _function_response_problem(payload)
    return _error_problem(payload, protocol)


def _answerable_call_id(envelope: JsonValue, protocol: A2UIProtocol) -> str | None:
    """The ``functionCallId`` a malformed v1.0 ``callAgentFunction`` can still be answered on, if it has one."""
    if not protocol.function_rpc or not isinstance(envelope, dict):
        return None
    if envelope.get("version") not in protocol.accepted_wire_versions:
        return None
    call = envelope.get("callAgentFunction")
    call_id = call.get("functionCallId") if isinstance(call, dict) else None
    # An empty id cannot be correlated, as the official SDK also refuses to answer one.
    return call_id if isinstance(call_id, str) and call_id else None


def decode_envelopes(envelopes: Iterable[JsonValue], *, protocol_version: A2UIVersion) -> list[A2UIIncomingParseResult]:
    """Decode client→server envelopes once per request, dropping (with a warning) those the protocol does not allow.

    Both the agent's prompts (:func:`incoming_prompts`) and the
    :class:`~ag2.a2ui.A2UIClientEvent`\\s observers see come from this decode.

    Each envelope must conform to the served protocol (see :func:`envelope_problem`). A malformed v1.0
    ``callAgentFunction`` with a ``functionCallId`` is kept as an
    :class:`A2UIIncomingInvalidFunctionCallResult`, which the server answers with ``INVALID_FUNCTION_CALL``;
    any other malformed envelope is dropped with a warning.

    Args:
        envelopes: The raw envelopes the request carried.
        protocol_version: The served protocol version.
    """
    protocol = get_protocol(protocol_version)
    results: list[A2UIIncomingParseResult] = []
    for envelope in envelopes:
        problem = envelope_problem(envelope, protocol)
        if problem is not None:
            if (call_id := _answerable_call_id(envelope, protocol)) is not None:
                results.append(A2UIIncomingInvalidFunctionCallResult(function_call_id=call_id, reason=problem))
                continue
            logger.warning("Skipping A2UI envelope that does not conform to protocol %s: %s", protocol.version, problem)
            continue
        result = parse_incoming_message(envelope)
        if isinstance(result, A2UIIncomingUnknownResult):
            logger.warning("Skipping A2UI envelope of unrecognized kind: %s", result.parse_error)
            continue
        results.append(result)
    return results


def incoming_prompts(
    interactions: Iterable[A2UIIncomingParseResult],
    resolve_action: Callable[[str], A2UIEventAction | None],
) -> Iterator[str]:
    """Rewrite decoded client→server interactions into prompt strings for the agent.

    Shared by every transport so the security-sensitive ``sanitize_for_prompt``
    handling lives in one place. The server answers some interactions itself, so
    they yield no prompt: a click on a **registered** action (``resolve_action``
    returns its declaration) runs via ``run_server_action``, and every
    ``callAgentFunction``, malformed or not, is answered via ``run_client_interactions``. A nameless
    ``action`` is dropped with a warning.
    """
    for result in interactions:
        if isinstance(result, A2UIIncomingActionResult):
            if resolve_action(result.action.name) is not None:
                continue
            prompt = action_to_prompt(result.action)
            if prompt is None:
                logger.warning("Dropping A2UI action with no name.")
                continue
            yield prompt
        elif isinstance(result, (A2UIIncomingFunctionCallResult, A2UIIncomingInvalidFunctionCallResult)):
            continue
        elif isinstance(result, A2UIIncomingFunctionResponseResult):
            yield function_response_to_prompt(result.function_response)
        elif isinstance(result, A2UIIncomingErrorResult):
            yield error_to_prompt(result.error)
        elif isinstance(result, A2UIIncomingUnknownResult):
            continue
        else:
            assert_never(result)


__all__ = (
    "A2UIFunctionError",
    "A2UIIncomingAction",
    "A2UIIncomingActionResult",
    "A2UIIncomingError",
    "A2UIIncomingErrorResult",
    "A2UIIncomingFunctionCall",
    "A2UIIncomingFunctionCallResult",
    "A2UIIncomingFunctionError",
    "A2UIIncomingFunctionResponse",
    "A2UIIncomingFunctionResponseResult",
    "A2UIIncomingInvalidFunctionCallResult",
    "A2UIIncomingParseResult",
    "A2UIIncomingSurfaceError",
    "A2UIIncomingUnknownResult",
    "action_to_prompt",
    "decode_envelopes",
    "envelope_problem",
    "error_to_prompt",
    "function_response_to_prompt",
    "incoming_prompts",
    "parse_incoming_message",
    "sanitize_for_prompt",
)
