# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, TypeAlias, TypedDict

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = "JsonScalar | list[JsonValue] | dict[str, JsonValue]"
JsonObject: TypeAlias = dict[str, "JsonValue"]

JsonSchema: TypeAlias = dict[str, "JsonValue"]
"""A JSON Schema document; named distinctly from ``JsonObject`` to express intent."""

A2UIVersion: TypeAlias = Literal["v0.9", "v0.9.1", "v1.0"]
"""Supported A2UI protocol versions. ``callFunction`` / ``actionResponse`` are ``v1.0``-only."""


class CreateSurfaceContent(TypedDict, total=False):
    """Payload of a ``createSurface`` message.

    Mirrors ``server_to_client.json#/$defs/CreateSurfaceMessage/properties/createSurface``.
    """

    surfaceId: str
    catalogId: str
    theme: JsonObject
    sendDataModel: bool


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
    """Payload of an ``updateDataModel`` message."""

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


class CallFunctionContent(TypedDict, total=False):
    """Payload of a ``callFunction`` message (v1.0).

    Mirrors ``server_to_client.json#/$defs/CallFunctionMessage/properties/callFunction``
    (a ``common_types.json#/$defs/FunctionCall`` with ``call`` required). Per spec,
    ``args`` values may be any ``DynamicValue`` (including ``DataBinding`` refs and
    nested function calls), not just plain JSON scalars/objects.
    """

    call: str
    args: JsonObject


class CallFunctionMessage(TypedDict, total=False):
    """``callFunction`` envelope (v1.0): server-initiated client function call.

    ``functionCallId`` is a unique id the client copies verbatim into its
    function response or error. ``version`` is ``v1.0``-only — this message type
    does not exist in v0.9.
    """

    version: Literal["v1.0"]
    functionCallId: str
    wantResponse: bool
    callFunction: CallFunctionContent


class ActionResponseError(TypedDict, total=False):
    """The ``error`` arm of an ``actionResponse`` body (v1.0).

    Both ``code`` and ``message`` are required by spec; kept ``total=False`` to
    match the structural (duck-typed) convention used throughout this module.
    """

    code: str
    message: str


class ActionResponseContent(TypedDict, total=False):
    """Payload of an ``actionResponse`` message (v1.0).

    Per spec the body carries exactly one of ``value`` or ``error``.
    """

    value: JsonValue
    error: ActionResponseError


class ActionResponseMessage(TypedDict, total=False):
    """``actionResponse`` envelope (v1.0): server response to a client action.

    ``version`` is ``v1.0``-only — this message type does not exist in v0.9.
    """

    version: Literal["v1.0"]
    actionId: str
    actionResponse: ActionResponseContent


ServerToClientMessage: TypeAlias = (
    CreateSurfaceMessage
    | UpdateComponentsMessage
    | UpdateDataModelMessage
    | DeleteSurfaceMessage
    | CallFunctionMessage
    | ActionResponseMessage
)


# NOTE: client→server (action/error) and client-capability wire shapes are
# intentionally NOT modeled as TypedDicts here. The code that parses them
# decodes into dedicated dataclasses instead — see ``incoming.py``
# (``A2UIIncomingAction`` / ``A2UIIncomingError``) and ``capabilities.py``
# (``A2UIClientCapabilities``). If a future phase needs typed inbound wire
# structures, add them here AND wire them into those parsers.


__all__ = (
    "A2UIVersion",
    "ActionResponseContent",
    "ActionResponseError",
    "ActionResponseMessage",
    "CallFunctionContent",
    "CallFunctionMessage",
    "CreateSurfaceContent",
    "CreateSurfaceMessage",
    "DeleteSurfaceContent",
    "DeleteSurfaceMessage",
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
