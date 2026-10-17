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

    Mirrors the ``action`` message of ``a2ui-core`` (every version): an action is
    fire-and-forget, so there is no response handle to correlate.
    """

    name: str
    surface_id: str
    source_component_id: str
    timestamp: str
    context: dict[str, JsonValue] = field(default_factory=dict)


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
    """A client→server ``error`` answering a server-initiated ``callRendererFunction`` (v1.0).

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
    """A client→server ``rendererFunctionResponse`` envelope content (v1.0, experimental).

    The renderer's reply to a server-initiated ``callRendererFunction``;
    ``function_call_id`` is copied verbatim from the originating call. The body
    carries a ``value`` or an ``error`` (``error_code`` / ``error_message`` set).
    """

    function_call_id: str
    value: JsonValue = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(slots=True, frozen=True)
class A2UIIncomingAgentFunctionCall:
    """A client→server ``callAgentFunction`` envelope content (v1.0, experimental).

    The renderer asks the agent to run a function and expects an
    ``agentFunctionResponse`` carrying the same ``function_call_id``.
    """

    surface_id: str
    function_call_id: str
    call: str
    args: dict[str, JsonValue] = field(default_factory=dict)


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
    """A successfully parsed client→server ``rendererFunctionResponse`` envelope (v1.0)."""

    function_response: A2UIIncomingFunctionResponse


@dataclass(slots=True, frozen=True)
class A2UIIncomingAgentFunctionCallResult:
    """A successfully parsed client→server ``callAgentFunction`` envelope (v1.0)."""

    agent_function_call: A2UIIncomingAgentFunctionCall


@dataclass(slots=True, frozen=True)
class A2UIIncomingUnknownResult:
    """An envelope that could not be classified; ``parse_error`` says why."""

    parse_error: str


# Result of parsing one client→server envelope. A discriminated union over the
# four mutually exclusive outcomes — each arm carries only its own payload, so
# the old "kind says action but action is None" states are unrepresentable.
# Branch with ``isinstance`` (mirrors the ``A2UIIncomingError`` union above).
A2UIIncomingParseResult = (
    A2UIIncomingActionResult
    | A2UIIncomingErrorResult
    | A2UIIncomingFunctionResponseResult
    | A2UIIncomingAgentFunctionCallResult
    | A2UIIncomingUnknownResult
)


def parse_incoming_message(data: JsonValue) -> A2UIIncomingParseResult:
    """Classify a single client→server A2UI envelope.

    Accepts a dict shaped like ``{"version": ..., "action": {...}}``,
    ``{"version": ..., "rendererFunctionResponse": {...}}``,
    ``{"version": ..., "callAgentFunction": {...}}``, or
    ``{"version": ..., "error": {...}}`` and returns a typed result. Parsing is
    structural and version-neutral — v1.0-only arms (``rendererFunctionResponse``,
    ``callAgentFunction``, ``error.functionCallId``) decode when present and are
    simply absent for v0.9 clients. It is lenient on purpose and does NOT
    validate the envelope against the JSON schema.
    """
    if not isinstance(data, dict):
        return A2UIIncomingUnknownResult(parse_error="envelope is not a JSON object")

    action_obj = data.get("action")
    if isinstance(action_obj, dict):
        name = str(action_obj.get("name", ""))
        raw_context = action_obj.get("context")
        return A2UIIncomingActionResult(
            action=A2UIIncomingAction(
                name=name,
                surface_id=str(action_obj.get("surfaceId", "")),
                source_component_id=str(action_obj.get("sourceComponentId", "")),
                timestamp=str(action_obj.get("timestamp", "")),
                context=dict(raw_context) if isinstance(raw_context, dict) else {},
            ),
        )

    function_response_obj = data.get("rendererFunctionResponse")
    if isinstance(function_response_obj, dict):
        response_error = function_response_obj.get("error")
        error_body = response_error if isinstance(response_error, dict) else {}
        return A2UIIncomingFunctionResponseResult(
            function_response=A2UIIncomingFunctionResponse(
                function_call_id=str(function_response_obj.get("functionCallId", "")),
                value=function_response_obj.get("value"),
                error_code=str(error_body.get("code", "")) if error_body else None,
                error_message=str(error_body.get("message", "")) if error_body else None,
            ),
        )

    agent_call_obj = data.get("callAgentFunction")
    if isinstance(agent_call_obj, dict):
        call_obj = agent_call_obj.get("callFunction")
        call_body = call_obj if isinstance(call_obj, dict) else {}
        raw_args = call_body.get("args")
        return A2UIIncomingAgentFunctionCallResult(
            agent_function_call=A2UIIncomingAgentFunctionCall(
                surface_id=str(agent_call_obj.get("surfaceId", "")),
                function_call_id=str(agent_call_obj.get("functionCallId", "")),
                call=str(call_body.get("@call", call_body.get("call", ""))),
                args=dict(raw_args) if isinstance(raw_args, dict) else {},
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
        parse_error="envelope has none of 'action', 'rendererFunctionResponse', 'callAgentFunction', or 'error' keys",
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

    return f"The user clicked the '{name}' button{origin}. Context: {ctx_json}"


def error_to_prompt(err: A2UIIncomingError) -> str:
    """Rewrite a client-reported A2UI ``error`` as a corrective LLM instruction.

    Handles both error arms: a surface-scoped validation/render error, and the
    v1.0 generic error answering a server-initiated ``callRendererFunction`` (carrying
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
    """Rewrite a client ``rendererFunctionResponse`` (v1.0) as a continuation instruction.

    The renderer has executed a server-initiated ``callRendererFunction`` and returned its
    result or an error. All client-supplied values pass through :func:`sanitize_for_prompt`.
    """
    fc_id = sanitize_for_prompt(fr.function_call_id) if fr.function_call_id else "(unknown)"
    if fr.error_code is not None:
        code = sanitize_for_prompt(fr.error_code) or "(none)"
        message = sanitize_for_prompt(fr.error_message or "")
        return (
            f"The client failed to execute function call (id={fc_id}). Code: {code}. Message: {message}. "
            "Continue without relying on that function's result."
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


def agent_function_call_to_prompt(call: A2UIIncomingAgentFunctionCall) -> str:
    """Rewrite a renderer ``callAgentFunction`` (v1.0) as an instruction to answer it.

    The LLM is told the correlation id so it can emit an ``agentFunctionResponse``
    inside its ``<a2ui-json>`` output. All client-supplied values pass through
    :func:`sanitize_for_prompt`.
    """
    name = sanitize_for_prompt(call.call)
    fc_id = sanitize_for_prompt(call.function_call_id)
    surface = f" on surface '{sanitize_for_prompt(call.surface_id)}'" if call.surface_id else ""
    args_json = sanitize_for_prompt(json.dumps(call.args))
    return (
        f"The UI called the agent function '{name}'{surface} with arguments: {args_json}. "
        f"Emit an A2UI 'agentFunctionResponse' message with functionCallId '{fc_id}' carrying the result "
        "(a 'value' on success, or an 'error')."
    )


def parse_incoming_interactions(envelopes: Iterable[JsonValue]) -> list[A2UIIncomingParseResult]:
    """Parse client→server envelopes into typed results, dropping unclassifiable ones.

    Used by the transports to surface each incoming interaction as an
    :class:`~ag2.a2ui.beta.A2UIClientEvent`. Unlike :func:`iter_incoming_prompts`
    (which rewrites envelopes into prompt strings), this keeps the structured
    result — ``action`` / ``rendererFunctionResponse`` / ``callAgentFunction`` / ``error`` — for observers.
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
    ``rendererFunctionResponse`` or ``callAgentFunction`` missing its ``functionCallId``, or an
    unrecognized kind.
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
                    "Dropping A2UI rendererFunctionResponse — missing functionCallId, "
                    "cannot correlate to a callRendererFunction.",
                )
                continue
            yield function_response_to_prompt(result.function_response)
        elif isinstance(result, A2UIIncomingAgentFunctionCallResult):
            if not result.agent_function_call.function_call_id or not result.agent_function_call.call:
                logger.warning("Dropping A2UI callAgentFunction — missing functionCallId or function name.")
                continue
            yield agent_function_call_to_prompt(result.agent_function_call)
        elif isinstance(result, A2UIIncomingErrorResult):
            yield error_to_prompt(result.error)
        elif isinstance(result, A2UIIncomingUnknownResult):
            logger.warning("Skipping A2UI envelope of unrecognized kind: %s", result.parse_error)
        else:
            assert_never(result)


__all__ = (
    "A2UIIncomingAction",
    "A2UIIncomingActionResult",
    "A2UIIncomingAgentFunctionCall",
    "A2UIIncomingAgentFunctionCallResult",
    "A2UIIncomingError",
    "A2UIIncomingErrorResult",
    "A2UIIncomingFunctionError",
    "A2UIIncomingFunctionResponse",
    "A2UIIncomingFunctionResponseResult",
    "A2UIIncomingParseResult",
    "A2UIIncomingSurfaceError",
    "A2UIIncomingUnknownResult",
    "action_to_prompt",
    "agent_function_call_to_prompt",
    "error_to_prompt",
    "function_response_to_prompt",
    "iter_incoming_prompts",
    "parse_incoming_interactions",
    "parse_incoming_message",
    "sanitize_for_prompt",
)
