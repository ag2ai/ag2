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
"""Supported A2UI protocol versions. ``callRendererFunction`` / ``agentFunctionResponse`` are ``v1.0``-only (experimental)."""


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
    """Body of a ``callRendererFunction`` message (v1.0): the function to run on the renderer.

    Mirrors a ``FunctionCall`` of a2ui-core's v1.0 schema. ``args`` values may be any
    ``DynamicValue`` (including ``DataBinding`` refs), not just plain JSON, and
    ``catalogId`` is required by the schema.
    """

    call: str
    args: JsonObject
    catalogId: str


class CallRendererFunctionContent(TypedDict, total=False):
    """Payload of a ``callRendererFunction`` message (v1.0)."""

    functionCallId: str
    callFunction: CallFunctionContent


class CallRendererFunctionMessage(TypedDict, total=False):
    """``callRendererFunction`` envelope (v1.0, experimental): server-initiated renderer function call.

    ``functionCallId`` is a unique id the renderer copies verbatim into its
    ``rendererFunctionResponse`` or ``error``. The message does not exist before v1.0.
    """

    version: Literal["v1.0"]
    callRendererFunction: CallRendererFunctionContent


class FunctionResponseError(TypedDict, total=False):
    """The ``error`` arm of a function response body (v1.0)."""

    code: str
    message: str


class AgentFunctionResponseContent(TypedDict, total=False):
    """Payload of an ``agentFunctionResponse`` message (v1.0): exactly one of ``value`` or ``error``."""

    functionCallId: str
    value: JsonValue
    error: FunctionResponseError


class AgentFunctionResponseMessage(TypedDict, total=False):
    """``agentFunctionResponse`` envelope (v1.0, experimental): the reply to a renderer's ``callAgentFunction``."""

    version: Literal["v1.0"]
    agentFunctionResponse: AgentFunctionResponseContent


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


# NOTE: client→server (action/error) and client-capability wire shapes are
# intentionally NOT modeled as TypedDicts here. The code that parses them
# decodes into dedicated dataclasses instead — see ``incoming.py``
# (``A2UIIncomingAction`` / ``A2UIIncomingError``) and ``capabilities.py``
# (``A2UIClientCapabilities``). If a future phase needs typed inbound wire
# structures, add them here AND wire them into those parsers.


__all__ = (
    "A2UIVersion",
    "AgentFunctionResponseContent",
    "AgentFunctionResponseMessage",
    "CallFunctionContent",
    "CallRendererFunctionContent",
    "CallRendererFunctionMessage",
    "CreateSurfaceContent",
    "CreateSurfaceMessage",
    "DeleteSurfaceContent",
    "DeleteSurfaceMessage",
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
