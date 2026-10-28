# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from functools import cache
from typing import Literal, TypeAlias, TypedDict, get_args, get_type_hints, is_typeddict

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = "JsonScalar | list[JsonValue] | dict[str, JsonValue]"
JsonObject: TypeAlias = dict[str, "JsonValue"]

JsonSchema: TypeAlias = dict[str, "JsonValue"]
"""A JSON Schema document; named distinctly from ``JsonObject`` to express intent."""

A2UIVersion: TypeAlias = Literal["v0.9", "v0.9.1", "v1.0"]
"""Supported A2UI protocol versions. ``callRendererFunction`` / ``agentFunctionResponse`` are ``v1.0``-only."""


class CreateSurfaceContent(TypedDict, total=False):
    """Payload of a ``createSurface`` message.

    Mirrors ``server_to_client.json#/$defs/CreateSurfaceMessage/properties/createSurface``
    (``agent_to_renderer.json`` in v1.0). ``theme`` is v0.9.x-only; ``components``,
    ``dataModel`` and ``metadata`` are v1.0-only (a whole UI in one message).
    """

    surfaceId: str
    catalogId: str
    theme: JsonObject
    sendDataModel: bool
    components: list[JsonObject]
    dataModel: JsonObject
    metadata: JsonObject


class CreateSurfaceMessage(TypedDict, total=False):
    """``createSurface`` envelope."""

    version: A2UIVersion
    createSurface: CreateSurfaceContent


class UpdateComponentsContent(TypedDict, total=False):
    """Payload of an ``updateComponents`` message."""

    surfaceId: str
    components: list[JsonObject]


class UpdateComponentsMessage(TypedDict, total=False):
    """``updateComponents`` envelope."""

    version: A2UIVersion
    updateComponents: UpdateComponentsContent


class UpdateDataModelContent(TypedDict, total=False):
    """Payload of an ``updateDataModel`` message.

    ``value`` is optional in v0.9.x and required in v1.0, where ``None`` deletes ``path``.
    """

    surfaceId: str
    path: str
    value: JsonValue


class UpdateDataModelMessage(TypedDict, total=False):
    """``updateDataModel`` envelope."""

    version: A2UIVersion
    updateDataModel: UpdateDataModelContent


class DeleteSurfaceContent(TypedDict, total=False):
    """Payload of a ``deleteSurface`` message."""

    surfaceId: str


class DeleteSurfaceMessage(TypedDict, total=False):
    """``deleteSurface`` envelope."""

    version: A2UIVersion
    deleteSurface: DeleteSurfaceContent


# The reserved ``@call`` key cannot be a keyword argument, so the functional form is needed.
CallRendererFunctionCall = TypedDict(
    "CallRendererFunctionCall",
    {"@call": str, "catalogId": str, "args": JsonObject},
    total=False,
)
"""A ``common_types.json#/$defs/FunctionCall``: the named function, its catalog, and its arguments.

Per spec, ``args`` values may be any ``DynamicValue`` (including ``@path`` bindings and nested
function calls), not just plain JSON scalars/objects.
"""


class CallRendererFunctionContent(TypedDict, total=False):
    """Payload of a ``callRendererFunction`` message (v1.0).

    Mirrors ``agent_to_renderer.json#/$defs/CallRendererFunctionMessage/properties/callRendererFunction``.
    ``surfaceId`` selects the data model the call's bindings resolve against; without it
    the function runs against an empty one.
    """

    surfaceId: str
    functionCallId: str
    callFunction: CallRendererFunctionCall


class CallRendererFunctionMessage(TypedDict, total=False):
    """``callRendererFunction`` envelope (v1.0): agent-initiated renderer function call.

    The renderer always answers with a ``rendererFunctionResponse`` that copies
    ``functionCallId`` verbatim. ``version`` is ``v1.0``-only — this message type
    does not exist in v0.9.
    """

    version: Literal["v1.0"]
    callRendererFunction: CallRendererFunctionContent


class FunctionResponseError(TypedDict, total=False):
    """The ``error`` arm of a ``FunctionResponse`` (v1.0).

    Both ``code`` and ``message`` are required by spec; kept ``total=False`` to
    match the structural (duck-typed) convention used throughout this module.
    """

    code: str
    message: str


class FunctionResponseContent(TypedDict, total=False):
    """Payload of an ``agentFunctionResponse`` message (v1.0).

    Mirrors ``common_types.json#/$defs/FunctionResponse``: the body carries
    ``functionCallId`` and exactly one of ``value`` or ``error``.
    """

    functionCallId: str
    value: JsonValue
    error: FunctionResponseError


class AgentFunctionResponseMessage(TypedDict, total=False):
    """``agentFunctionResponse`` envelope (v1.0): the agent's reply to a ``callAgentFunction``.

    ``version`` is ``v1.0``-only — this message type does not exist in v0.9.
    """

    version: Literal["v1.0"]
    agentFunctionResponse: FunctionResponseContent


ServerToClientMessage: TypeAlias = (
    CreateSurfaceMessage
    | UpdateComponentsMessage
    | UpdateDataModelMessage
    | DeleteSurfaceMessage
    | CallRendererFunctionMessage
    | AgentFunctionResponseMessage
)


@cache
def server_to_client_payload_keys() -> frozenset[str]:
    """The key carrying the payload of each ``ServerToClientMessage``, read off its ``TypedDict``."""
    return frozenset(
        key
        for message in get_args(ServerToClientMessage)
        for key, annotation in get_type_hints(message).items()
        if is_typeddict(annotation)
    )


# NOTE: client→server (action / callAgentFunction / rendererFunctionResponse /
# error) and client-capability wire shapes are intentionally NOT modeled as
# TypedDicts here. The code that parses them
# decodes into dedicated dataclasses instead — see ``incoming.py``
# (``A2UIIncomingAction`` / ``A2UIIncomingError``) and ``capabilities.py``
# (``A2UIClientCapabilities``). If a future phase needs typed inbound wire
# structures, add them here AND wire them into those parsers.


__all__ = (
    "A2UIVersion",
    "AgentFunctionResponseMessage",
    "CallRendererFunctionCall",
    "CallRendererFunctionContent",
    "CallRendererFunctionMessage",
    "CreateSurfaceContent",
    "CreateSurfaceMessage",
    "DeleteSurfaceContent",
    "DeleteSurfaceMessage",
    "FunctionResponseContent",
    "FunctionResponseError",
    "JsonObject",
    "JsonScalar",
    "JsonSchema",
    "JsonValue",
    "ServerToClientMessage",
    "UpdateComponentsContent",
    "UpdateComponentsMessage",
    "UpdateDataModelContent",
    "UpdateDataModelMessage",
)
