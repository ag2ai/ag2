# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ._types import A2UIVersion

A2UI_MIME_TYPE = "application/a2ui+json"

# Official A2UI "Standard Prompt Tags" (a2ui-project/a2ui, agent_sdk_guide.md):
# the LLM wraps its A2UI output between these tags for deterministic parsing,
# e.g. ``CONVERSATIONAL TEXT\n<a2ui-json>[ {…} ]</a2ui-json>``. This replaces
# AG2's earlier homegrown ``---a2ui_JSON---`` delimiter.
A2UI_JSON_OPEN_TAG = "<a2ui-json>"
A2UI_JSON_CLOSE_TAG = "</a2ui-json>"

A2UI_DEFAULT_VERSION: A2UIVersion = "v0.9"

# Canonical default ("basic") catalog id per protocol version. This value is
# stamped into ``createSurface`` messages and advertised in
# ``supportedCatalogIds``; it MUST equal the ``$id`` of the catalog schema we
# vendored (the parser builds component ``$ref``s as ``{catalog_id}#/...`` and
# resolves them against the schema registry keyed by that ``$id``).
#
# v0.9.1 reuses v0.9's catalog id ON PURPOSE: the upstream v0.9.1 catalog file
# (``specification/v0_9_1/catalogs/basic/catalog.json``) itself declares
# ``"$id": ".../v0_9/catalogs/basic/catalog.json"``. The upstream v0.9.1
# *extension-spec doc* example advertises a ``v0_9_1`` URL, but that conflicts
# with the catalog file's own ``$id`` — following the doc would break ``$ref``
# resolution here, so we mirror the authoritative ``$id`` (v0_9) instead.
A2UI_DEFAULT_CATALOG_ID_BY_VERSION: dict[A2UIVersion, str] = {
    "v0.9": "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json",
    "v0.9.1": "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json",
    "v1.0": "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json",
}

# Canonical A2A extension URI per protocol version (a2ui-project/a2ui,
# ``a2ui_extension_specification.md``; the v1.0 evolution guide bumps the
# namespace from v0.9/v0.9.1 to v1.0). A client and server agree on A2UI by
# matching this exact URI in the AgentCard ``capabilities.extensions``.
A2UI_EXTENSION_URI_BY_VERSION: dict[A2UIVersion, str] = {
    "v0.9": "https://a2ui.org/a2a-extension/a2ui/v0.9",
    "v0.9.1": "https://a2ui.org/a2a-extension/a2ui/v0.9.1",
    "v1.0": "https://a2ui.org/a2a-extension/a2ui/v1.0",
}

__all__ = (
    "A2UI_DEFAULT_CATALOG_ID_BY_VERSION",
    "A2UI_DEFAULT_VERSION",
    "A2UI_EXTENSION_URI_BY_VERSION",
    "A2UI_JSON_CLOSE_TAG",
    "A2UI_JSON_OPEN_TAG",
    "A2UI_MIME_TYPE",
)
