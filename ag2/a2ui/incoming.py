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
class ActionResponseRequest:
    """The v1.0 correlation handle when a client action awaits an ``actionResponse``.

    Present only when the client set ``wantResponse`` **and** supplied an
    ``actionId``. Making the id non-optional here is what keeps the illegal
    "wants a response but gave no id to correlate it" state unrepresentable: an
    :class:`A2UIIncomingAction` either carries a fully-formed request or ``None``.
    """

    action_id: str


@dataclass(slots=True, frozen=True)
class A2UIIncomingAction:
    """A client→server ``action`` envelope content.

    Mirrors ``client_to_server.json#/properties/action``. ``response_request``
    is the v1.0 addition: when set, the client expects the server to reply with
    an ``actionResponse`` carrying the same ``action_id``. It is ``None`` for
    v0.9 clients (no response expected).
    """

    name: str
    surface_id: str
    source_component_id: str
    timestamp: str
    context: dict[str, JsonValue] = field(default_factory=dict)
    response_request: ActionResponseRequest | None = None


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
    """A client→server ``error`` answering a server-initiated ``callFunction`` (v1.0).

    The generic-error arm of ``client_to_server.json#/properties/error``: it
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
class A2UIIncomingFunctionResponse:
    """A client→server ``functionResponse`` envelope content (v1.0).

    The client's successful reply to a server-initiated ``callFunction``.
    Mirrors ``client_to_server.json#/properties/functionResponse``;
    ``function_call_id`` is copied verbatim from the originating call.
    """

    function_call_id: str
    call: str
    value: JsonValue = None


@dataclass(slots=True, frozen=True)
class A2UIIncomingActionResult:
    """A successfully parsed client→server ``action`` envelope."""

    action: A2UIIncomingAction


@dataclass(slots=True, frozen=True)
class A2UIIncomingErrorResult:
    """A successfully parsed client→server ``error`` envelope."""

    error: A2UIIncomingError


@dataclass(slots=True, frozen=True)
class A2UIIncomingFunctionResponseResult:
    """A successfully parsed client→server ``functionResponse`` envelope (v1.0)."""

    function_response: A2UIIncomingFunctionResponse


@dataclass(slots=True, frozen=True)
class A2UIIncomingUnknownResult:
    """An envelope that could not be classified; ``parse_error`` says why."""

    parse_error: str


# Result of parsing one client→server envelope. A discriminated union over the
# four mutually exclusive outcomes — each arm carries only its own payload, so
# the old "kind says action but action is None" states are unrepresentable.
# Branch with ``isinstance`` (mirrors the ``A2UIIncomingError`` union above).
A2UIIncomingParseResult = (
    A2UIIncomingActionResult | A2UIIncomingErrorResult | A2UIIncomingFunctionResponseResult | A2UIIncomingUnknownResult
)


def parse_incoming_message(data: JsonValue) -> A2UIIncomingParseResult:
    """Classify a single client→server A2UI envelope.

    Accepts a dict shaped like ``{"version": ..., "action": {...}}``,
    ``{"version": ..., "functionResponse": {...}}``, or
    ``{"version": ..., "error": {...}}`` and returns a typed result. Parsing is
    structural and version-neutral — v1.0-only arms (``functionResponse``,
    ``action.wantResponse``/``actionId``, ``error.functionCallId``) decode when
    present and are simply absent for v0.9 clients. Does NOT validate the
    envelope against the JSON schema — use
    ``A2UISchemaManager.client_to_server_schema`` for strict validation.
    """
    if not isinstance(data, dict):
        return A2UIIncomingUnknownResult(parse_error="envelope is not a JSON object")

    action_obj = data.get("action")
    if isinstance(action_obj, dict):
        name = str(action_obj.get("name", ""))
        raw_action_id = action_obj.get("actionId")
        action_id = raw_action_id if isinstance(raw_action_id, str) else None
        response_request: ActionResponseRequest | None = None
        if action_obj.get("wantResponse", False):
            if action_id:
                response_request = ActionResponseRequest(action_id=action_id)
            else:
                # Malformed v1.0: wantResponse with no actionId. Without an id we
                # cannot correlate an actionResponse, so the request is dropped
                # (and the client's pending response can never be satisfied) —
                # surface it rather than fabricate a correlation handle.
                logger.warning(
                    "A2UI action %r set wantResponse but provided no actionId; cannot correlate an actionResponse.",
                    name,
                )
        raw_context = action_obj.get("context")
        return A2UIIncomingActionResult(
            action=A2UIIncomingAction(
                name=name,
                surface_id=str(action_obj.get("surfaceId", "")),
                source_component_id=str(action_obj.get("sourceComponentId", "")),
                timestamp=str(action_obj.get("timestamp", "")),
                context=dict(raw_context) if isinstance(raw_context, dict) else {},
                response_request=response_request,
            ),
        )

    function_response_obj = data.get("functionResponse")
    if isinstance(function_response_obj, dict):
        return A2UIIncomingFunctionResponseResult(
            function_response=A2UIIncomingFunctionResponse(
                function_call_id=str(function_response_obj.get("functionCallId", "")),
                call=str(function_response_obj.get("call", "")),
                value=function_response_obj.get("value"),
            ),
        )

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
        parse_error="envelope has none of 'action', 'functionResponse', or 'error' keys",
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

    # v1.0: when the client expects an actionResponse, tell the LLM the
    # correlation id so it can emit one inside its <a2ui-json> output. The
    # malformed "wantResponse without actionId" case is already dropped (and
    # warned) at parse time, so here a present ``response_request`` always
    # carries a usable id.
    response_hint = ""
    if action.response_request is not None:
        action_id = sanitize_for_prompt(action.response_request.action_id)
        response_hint = (
            f" The UI is awaiting a response: emit an A2UI 'actionResponse' message with "
            f"actionId '{action_id}' carrying the result (a 'value' on success, or an 'error')."
        )

    return f"The user clicked the '{name}' button{origin}. Context: {ctx_json}{response_hint}"


def error_to_prompt(err: A2UIIncomingError) -> str:
    """Rewrite a client-reported A2UI ``error`` as a corrective LLM instruction.

    Handles both error arms: a surface-scoped validation/render error, and the
    v1.0 generic error answering a server-initiated ``callFunction`` (carrying
    a ``functionCallId`` instead of a surface).
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
    """Rewrite a client ``functionResponse`` (v1.0) as a continuation instruction.

    The client has executed a server-initiated ``callFunction`` and returned its
    result. All client-supplied values pass through :func:`sanitize_for_prompt`.
    """
    call = sanitize_for_prompt(fr.call) if fr.call else "(unknown)"
    fc_id = sanitize_for_prompt(fr.function_call_id) if fr.function_call_id else "(unknown)"
    try:
        value_json = sanitize_for_prompt(json.dumps(fr.value))
    except (TypeError, ValueError):
        logger.warning("functionResponse value for call '%s' is not JSON-serializable; using a placeholder.", fr.call)
        value_json = "<non-serializable>"
    return (
        f"The client executed the '{call}' function call (id={fc_id}) and returned: {value_json}. "
        "Use this result to continue."
    )


def parse_incoming_interactions(envelopes: Iterable[JsonValue]) -> list[A2UIIncomingParseResult]:
    """Parse client→server envelopes into typed results, dropping unclassifiable ones.

    Used by the transports to surface each incoming interaction as an
    :class:`~ag2.a2ui.A2UIClientEvent`. Unlike :func:`iter_incoming_prompts`
    (which rewrites envelopes into prompt strings), this keeps the structured
    result — ``action`` / ``functionResponse`` / ``error`` — for observers.
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
) -> Iterator[str]:
    """Rewrite client→server A2UI envelopes into corrective prompt strings.

    Shared by the A2A and REST transports so the (security-sensitive) envelope
    classification + ``sanitize_for_prompt`` handling lives in one place; each
    transport only wraps the yielded strings in its own input type
    (``TextInput`` / ``Part(text=...)``). Clicks on a **registered** action
    (``resolve_action`` returns its declaration) are skipped here — they run on
    the server via ``run_server_action`` instead. Envelopes that cannot be safely
    mapped are dropped with a warning: a nameless ``action``, a
    ``functionResponse`` missing its ``functionCallId``, or an unrecognized kind.
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
        elif isinstance(result, A2UIIncomingFunctionResponseResult):
            if not result.function_response.function_call_id:
                logger.warning(
                    "Dropping A2UI functionResponse — missing functionCallId, cannot correlate to a callFunction.",
                )
                continue
            yield function_response_to_prompt(result.function_response)
        elif isinstance(result, A2UIIncomingErrorResult):
            yield error_to_prompt(result.error)
        elif isinstance(result, A2UIIncomingUnknownResult):
            logger.warning("Skipping A2UI envelope of unrecognized kind: %s", result.parse_error)
        else:
            assert_never(result)


__all__ = (
    "A2UIIncomingAction",
    "A2UIIncomingActionResult",
    "A2UIIncomingError",
    "A2UIIncomingErrorResult",
    "A2UIIncomingFunctionError",
    "A2UIIncomingFunctionResponse",
    "A2UIIncomingFunctionResponseResult",
    "A2UIIncomingParseResult",
    "A2UIIncomingSurfaceError",
    "A2UIIncomingUnknownResult",
    "ActionResponseRequest",
    "action_to_prompt",
    "error_to_prompt",
    "function_response_to_prompt",
    "iter_incoming_prompts",
    "parse_incoming_interactions",
    "parse_incoming_message",
    "sanitize_for_prompt",
)
