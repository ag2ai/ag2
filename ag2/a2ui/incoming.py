# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

from typing_extensions import assert_never

from ._types import JsonValue
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
    if not isinstance(function_call_id, str) or not function_call_id:
        return A2UIIncomingUnknownResult(parse_error="'callAgentFunction' has no 'functionCallId' to correlate a reply")
    if not isinstance(call_obj, dict) or not isinstance(name, str) or not name:
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
    if not isinstance(function_call_id, str) or not function_call_id:
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
    ``{"version": ..., "error": {...}}`` and returns a typed result. Parsing is
    structural and version-neutral — v1.0-only arms (``callAgentFunction``,
    ``rendererFunctionResponse``, ``action.userMessage``, ``error.functionCallId``)
    decode when present and are simply absent for v0.9 clients. Does NOT validate
    the envelope against the JSON schema — use
    ``A2UISchemaManager.client_to_server_schema`` for strict validation.
    """
    if not isinstance(data, dict):
        return A2UIIncomingUnknownResult(parse_error="envelope is not a JSON object")

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


def function_response_to_prompt(fr: A2UIIncomingFunctionResponse, function_name: str | None = None) -> str:
    """Rewrite a client ``rendererFunctionResponse`` (v1.0) as a continuation instruction.

    The client has run a server-initiated ``callRendererFunction`` and returned
    its result or its failure. ``function_name`` (when the server remembers the call)
    names the function. All client-supplied values pass through
    :func:`sanitize_for_prompt`.
    """
    fc_id = sanitize_for_prompt(fr.function_call_id) if fr.function_call_id else "(unknown)"
    if function_name:
        fc_id = f"{fc_id}, function={sanitize_for_prompt(function_name)}"
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


def parse_incoming_interactions(envelopes: Iterable[JsonValue]) -> list[A2UIIncomingParseResult]:
    """Parse client→server envelopes into typed results, dropping unclassifiable ones.

    Used by the transports to surface each incoming interaction as an
    :class:`~ag2.a2ui.A2UIClientEvent`. Unlike :func:`iter_incoming_prompts`
    (which rewrites envelopes into prompt strings), this keeps the structured
    result — ``action`` / ``callAgentFunction`` / ``rendererFunctionResponse`` / ``error`` — for observers.
    ``unknown`` (unclassifiable) envelopes are skipped.
    """
    results: list[A2UIIncomingParseResult] = []
    for envelope in envelopes:
        result = parse_incoming_message(envelope)
        if isinstance(result, A2UIIncomingUnknownResult):
            continue
        results.append(result)
    return results


def iter_incoming_prompts(
    envelopes: Iterable[JsonValue],
    resolve_action: Callable[[str], A2UIEventAction | None],
    resolve_function: Callable[[str], str | None] | None = None,
) -> Iterator[str]:
    """Rewrite client→server A2UI envelopes into corrective prompt strings.

    Shared by the A2A and REST transports so the (security-sensitive) envelope
    classification + ``sanitize_for_prompt`` handling lives in one place; each
    transport only wraps the yielded strings in its own input type
    (``TextInput`` / ``Part(text=...)``). Clicks on a **registered** action
    (``resolve_action`` returns its declaration) are skipped here — they run on
    the server via ``run_server_action`` instead. So is every ``callAgentFunction``:
    the agent is never asked to answer one, the server does, via
    ``run_agent_function``. Envelopes that cannot be safely mapped are dropped with
    a warning: a nameless ``action`` or an unrecognized or malformed kind.
    ``resolve_function`` maps a ``functionCallId`` to the name of the
    ``callRendererFunction`` it answers, so the continuation prompt can name it.
    """
    for envelope in envelopes:
        result = parse_incoming_message(envelope)
        if isinstance(result, A2UIIncomingActionResult):
            if resolve_action(result.action.name) is not None:
                # Registered action → handled on the server (its handler runs via
                # run_server_action); never rewritten into a prompt for the agent.
                continue
            prompt = action_to_prompt(result.action)
            if prompt is None:
                logger.warning("Dropping A2UI action with no name.")
                continue
            yield prompt
        elif isinstance(result, A2UIIncomingFunctionCallResult):
            # Answered on the server (run_agent_function), never by the LLM.
            continue
        elif isinstance(result, A2UIIncomingFunctionResponseResult):
            response = result.function_response
            name = resolve_function(response.function_call_id) if resolve_function else None
            yield function_response_to_prompt(response, name)
        elif isinstance(result, A2UIIncomingErrorResult):
            yield error_to_prompt(result.error)
        elif isinstance(result, A2UIIncomingUnknownResult):
            logger.warning("Skipping A2UI envelope of unrecognized kind: %s", result.parse_error)
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
    "A2UIIncomingParseResult",
    "A2UIIncomingSurfaceError",
    "A2UIIncomingUnknownResult",
    "action_to_prompt",
    "error_to_prompt",
    "function_response_to_prompt",
    "iter_incoming_prompts",
    "parse_incoming_interactions",
    "parse_incoming_message",
    "sanitize_for_prompt",
)
