# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Decode renderer→agent envelopes into typed results, dropping those the protocol does not allow."""

import logging
from collections.abc import Iterable

from .._messages import CLIENT_TO_SERVER_KEYS
from .._types import A2UIVersion, JsonValue
from .._versions import A2UIProtocol, get_protocol
from ._results import (
    A2UIFunctionError,
    A2UIIncomingAction,
    A2UIIncomingActionResult,
    A2UIIncomingError,
    A2UIIncomingErrorResult,
    A2UIIncomingFunctionCall,
    A2UIIncomingFunctionCallResult,
    A2UIIncomingFunctionError,
    A2UIIncomingFunctionResponse,
    A2UIIncomingFunctionResponseResult,
    A2UIIncomingInvalidFunctionCallResult,
    A2UIIncomingParseResult,
    A2UIIncomingSurfaceError,
    A2UIIncomingUnknownResult,
)
from ._rules import envelope_problem

logger = logging.getLogger(__name__)


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
