# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Parsers for A2UI client metadata carried in A2A message ``metadata``.

A2UI v0.9 places ``a2uiClientCapabilities`` (per
``client_capabilities.json``) and ``a2uiClientDataModel`` (per
``client_data_model.json``) inside the A2A message metadata. These
helpers decode those nested objects into typed dataclasses.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .._types import JsonObject

A2UI_CLIENT_CAPABILITIES_METADATA_KEY = "a2uiClientCapabilities"
A2UI_CLIENT_DATA_MODEL_METADATA_KEY = "a2uiClientDataModel"

# TODO(a2ui-transports): parametrize by negotiated protocol version. This is
# A2A transport-level version negotiation; left hardcoded to v0.9 until the
# transport phase wires the negotiated version through to these parsers.
_VERSION_KEY = "v0.9"


@dataclass(slots=True)
class A2UIClientCapabilities:
    """Decoded ``a2uiClientCapabilities.v0.9`` payload."""

    supported_catalog_ids: list[str] = field(default_factory=list)
    inline_catalogs: list[JsonObject] = field(default_factory=list)


@dataclass(slots=True)
class A2UIClientDataModel:
    """Decoded ``a2uiClientDataModel`` payload (v0.9)."""

    surfaces: dict[str, JsonObject] = field(default_factory=dict)


def parse_client_capabilities(metadata: Mapping[str, Any] | None) -> A2UIClientCapabilities | None:
    """Decode ``metadata.a2uiClientCapabilities.v0.9``.

    Returns ``None`` if the metadata is missing, malformed, or does not
    declare v0.9.
    """
    if not metadata:
        return None
    caps = metadata.get(A2UI_CLIENT_CAPABILITIES_METADATA_KEY)
    if not isinstance(caps, dict):
        return None
    v = caps.get(_VERSION_KEY)
    if not isinstance(v, dict):
        return None
    raw_ids = v.get("supportedCatalogIds")
    raw_inline = v.get("inlineCatalogs")
    return A2UIClientCapabilities(
        supported_catalog_ids=[str(x) for x in raw_ids] if isinstance(raw_ids, list) else [],
        inline_catalogs=[c for c in raw_inline if isinstance(c, dict)] if isinstance(raw_inline, list) else [],
    )


def parse_client_data_model(metadata: Mapping[str, Any] | None) -> A2UIClientDataModel | None:
    """Decode ``metadata.a2uiClientDataModel``.

    Returns ``None`` if the metadata is missing, malformed, or its
    declared ``version`` is not ``v0.9``.
    """
    if not metadata:
        return None
    dm = metadata.get(A2UI_CLIENT_DATA_MODEL_METADATA_KEY)
    if not isinstance(dm, dict):
        return None
    if dm.get("version") != _VERSION_KEY:
        return None
    raw_surfaces = dm.get("surfaces")
    if not isinstance(raw_surfaces, dict):
        return None
    surfaces: dict[str, JsonObject] = {str(k): v for k, v in raw_surfaces.items() if isinstance(v, dict)}
    return A2UIClientDataModel(surfaces=surfaces)
