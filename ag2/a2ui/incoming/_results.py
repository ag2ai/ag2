# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The typed results a renderer→agent envelope decodes into."""

from dataclasses import dataclass, field

from .._types import JsonValue


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
