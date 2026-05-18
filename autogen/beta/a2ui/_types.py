# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, TypeAlias, TypedDict

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = "JsonScalar | list[JsonValue] | dict[str, JsonValue]"
JsonObject: TypeAlias = dict[str, "JsonValue"]
"""A JSON-shaped dict: keys are strings, values are any JSON value."""

JsonSchema: TypeAlias = dict[str, "JsonValue"]
"""A JSON Schema document. Same shape as ``JsonObject``; named distinctly
so call sites express intent (schema vs. arbitrary data)."""


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

    version: Literal["v0.9"]
    createSurface: CreateSurfaceContent


class UpdateComponentsContent(TypedDict, total=False):
    """Payload of an ``updateComponents`` message."""

    surfaceId: str
    components: list[JsonObject]


class UpdateComponentsMessage(TypedDict, total=False):
    """``updateComponents`` envelope."""

    version: Literal["v0.9"]
    updateComponents: UpdateComponentsContent


class UpdateDataModelContent(TypedDict, total=False):
    """Payload of an ``updateDataModel`` message."""

    surfaceId: str
    path: str
    value: JsonValue


class UpdateDataModelMessage(TypedDict, total=False):
    """``updateDataModel`` envelope."""

    version: Literal["v0.9"]
    updateDataModel: UpdateDataModelContent


class DeleteSurfaceContent(TypedDict, total=False):
    """Payload of a ``deleteSurface`` message."""

    surfaceId: str


class DeleteSurfaceMessage(TypedDict, total=False):
    """``deleteSurface`` envelope."""

    version: Literal["v0.9"]
    deleteSurface: DeleteSurfaceContent


ServerToClientMessage: TypeAlias = (
    CreateSurfaceMessage | UpdateComponentsMessage | UpdateDataModelMessage | DeleteSurfaceMessage
)


class ActionPayload(TypedDict, total=False):
    """Payload of an ``action`` client→server message."""

    name: str
    surfaceId: str
    sourceComponentId: str
    timestamp: str
    context: JsonObject


class ErrorPayload(TypedDict, total=False):
    """Payload of an ``error`` client→server message.

    ``path`` is only set for ``code == "VALIDATION_FAILED"`` per spec.
    """

    code: str
    surfaceId: str
    message: str
    path: str


class ActionEnvelope(TypedDict, total=False):
    """``action`` envelope."""

    version: Literal["v0.9"]
    action: ActionPayload


class ErrorEnvelope(TypedDict, total=False):
    """``error`` envelope."""

    version: Literal["v0.9"]
    error: ErrorPayload


ClientToServerMessage: TypeAlias = ActionEnvelope | ErrorEnvelope


class ClientCapabilitiesV09Payload(TypedDict, total=False):
    """``a2uiClientCapabilities.v0.9`` body."""

    supportedCatalogIds: list[str]
    inlineCatalogs: list[JsonObject]


class ClientCapabilitiesEnvelope(TypedDict, total=False):
    """Outer wrapper keyed by version, e.g. ``{"v0.9": {...}}``."""

    v09: ClientCapabilitiesV09Payload


class ClientDataModelPayload(TypedDict, total=False):
    """``a2uiClientDataModel`` body."""

    version: Literal["v0.9"]
    surfaces: dict[str, JsonObject]


__all__ = (
    "ActionEnvelope",
    "ActionPayload",
    "ClientCapabilitiesEnvelope",
    "ClientCapabilitiesV09Payload",
    "ClientDataModelPayload",
    "ClientToServerMessage",
    "CreateSurfaceContent",
    "CreateSurfaceMessage",
    "DeleteSurfaceContent",
    "DeleteSurfaceMessage",
    "ErrorEnvelope",
    "ErrorPayload",
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
