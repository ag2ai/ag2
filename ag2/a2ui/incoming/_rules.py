# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Whether a renderer→agent envelope conforms to the served protocol version."""

from collections.abc import Callable

from .._formats import is_date_time
from .._messages import CLIENT_TO_SERVER_KEYS
from .._types import JsonObject, JsonValue
from .._versions import A2UIProtocol

_FUNCTION_RPC_ENVELOPES = frozenset({"callAgentFunction", "rendererFunctionResponse"})
"""The envelope kinds that exist only with function RPC (v1.0)."""

_SURFACE_ERROR_CODES_V1 = frozenset({"VALIDATION_FAILED", "UNALLOWED_PARENT", "UNALLOWED_CHILD"})
"""v1.0 error codes that point at a component on a surface, so they carry its ``path`` (``renderer_to_agent.json``)."""


def _missing_strings(obj: JsonObject, *keys: str) -> list[str]:
    return [key for key in keys if not isinstance(obj.get(key), str)]


def _unknown_properties(obj: JsonObject, where: str, allowed: frozenset[str]) -> str | None:
    """The problem with ``obj`` (named ``where``) having properties besides ``allowed``, if it has any."""
    extra = sorted(obj.keys() - allowed)
    return f"{where} has unknown properties {', '.join(map(repr, extra))}" if extra else None


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
    if problem := _unknown_properties(metadata, "'action.metadata'", frozenset({"extensions"})):
        return problem
    extensions = metadata.get("extensions", {})
    if not isinstance(extensions, dict):
        return "'action.metadata.extensions' must be an object"
    # ``str.isidentifier`` is defined by XID_Start (plus ``_``) and XID_Continue, the sets UAX #31 names.
    invalid = sorted(key for key in extensions if not key.isidentifier())
    if invalid:
        return f"'action.metadata.extensions' keys must be UAX #31 identifiers: {', '.join(map(repr, invalid))}"
    return None


_CALL_AGENT_FUNCTION_KEYS = frozenset({"surfaceId", "functionCallId", "callFunction"})
_FUNCTION_CALL_KEYS = frozenset({"@call", "catalogId", "args"})
"""``FunctionCall`` is ``unevaluatedProperties: false`` over ``FunctionCommon`` and the function's own definition
(``common_types.json``); an agent function's definition is its handler, whose arguments sit under ``args``."""
_FUNCTION_RESPONSE_KEYS = frozenset({"functionCallId", "value", "error"})
_FUNCTION_RESPONSE_ERROR_KEYS = frozenset({"code", "message"})
_SURFACE_ERROR_KEYS = frozenset({"code", "path", "message", "surfaceId"})


def _function_call_problem(call: JsonObject) -> str | None:
    if problem := _unknown_properties(call, "'callAgentFunction'", _CALL_AGENT_FUNCTION_KEYS):
        return problem
    if not isinstance(call.get("functionCallId"), str):
        return "'callAgentFunction' needs a string 'functionCallId'"
    # The official SDK answers an empty surfaceId or function name with INVALID_FUNCTION_CALL, as here.
    if not isinstance(call.get("surfaceId"), str) or not call["surfaceId"]:
        return "'callAgentFunction' needs a non-empty string 'surfaceId'"
    function = call.get("callFunction")
    if not isinstance(function, dict):
        return "'callAgentFunction' needs a 'callFunction' object"
    if problem := _unknown_properties(function, "'callAgentFunction.callFunction'", _FUNCTION_CALL_KEYS):
        return problem
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
    if problem := _unknown_properties(response, "'rendererFunctionResponse'", _FUNCTION_RESPONSE_KEYS):
        return problem
    if ("value" in response) == ("error" in response):
        return "'rendererFunctionResponse' must carry exactly one of 'value' or 'error'"
    error = response.get("error")
    if "error" in response and (
        not isinstance(error, dict)
        or _missing_strings(error, *_FUNCTION_RESPONSE_ERROR_KEYS)
        or error.keys() - _FUNCTION_RESPONSE_ERROR_KEYS
    ):
        return "'rendererFunctionResponse.error' must be an object with exactly string 'code' and 'message'"
    return None


def _error_problem(error: JsonObject, protocol: A2UIProtocol) -> str | None:
    code = error.get("code")
    surface_codes = _SURFACE_ERROR_CODES_V1 if protocol.function_rpc else frozenset({"VALIDATION_FAILED"})
    if code in surface_codes:
        if _missing_strings(error, *_SURFACE_ERROR_KEYS) or error.keys() - _SURFACE_ERROR_KEYS:
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


_PAYLOAD_PROBLEMS: dict[str, Callable[[JsonObject, A2UIProtocol], str | None]] = {
    "action": _action_problem,
    # Neither depends on the version: both payloads exist only in v1.0.
    "callAgentFunction": lambda payload, _protocol: _function_call_problem(payload),
    "rendererFunctionResponse": lambda payload, _protocol: _function_response_problem(payload),
    "error": _error_problem,
}
"""What checks each renderer→agent payload kind (``CLIENT_TO_SERVER_KEYS``)."""


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
    if problem := _unknown_properties(envelope, "envelope", frozenset({"version", kind})):
        return problem
    version = envelope.get("version")
    if version not in protocol.accepted_wire_versions:
        accepted = " or ".join(map(repr, protocol.accepted_wire_versions))
        return f"envelope 'version' is {version!r}; this server speaks {accepted}"
    payload = envelope[kind]
    if not isinstance(payload, dict):
        return f"'{kind}' is not an object"
    return _PAYLOAD_PROBLEMS[kind](payload, protocol)
