# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Every per-version fact about the supported A2UI protocol versions, in one place.

Modules ask the :class:`A2UIProtocol` descriptor instead of comparing version strings.
"""

from dataclasses import dataclass

from ._types import A2UIVersion, A2UIWireVersion, server_to_client_payload_keys

A2UI_CLIENT_CAPABILITIES_METADATA_KEY = "a2uiClientCapabilities"
# v1.0 renamed the client "renderer" (A2A extension spec).
A2UI_RENDERER_CAPABILITIES_METADATA_KEY = "a2uiRendererCapabilities"
A2UI_CLIENT_DATA_MODEL_METADATA_KEY = "a2uiClientDataModel"
A2UI_RENDERER_DATA_MODEL_METADATA_KEY = "a2uiRendererDataModel"


@dataclass(frozen=True, slots=True)
class A2UIProtocol:
    """One supported A2UI protocol version."""

    version: A2UIVersion
    """The name users pick (``A2UIServer(protocol_version=...)``)."""

    wire_version: A2UIWireVersion
    """The ``version`` value stamped on emitted messages; v1.0 drops the ``v`` (a2ui-project/a2ui#3726)."""

    accepted_wire_versions: tuple[str, ...]
    """The ``version`` values a received message may carry."""

    package_dir: str
    """Directory under ``ag2/a2ui`` holding the vendored ``spec/`` and the prompt templates."""

    server_to_client_file: str
    """Vendored agent→renderer message schema. Upstream renamed server/client to agent/renderer in v1.0."""

    schema_base_uri: str
    """The base the vendored schemas' ``$id``s and relative ``$ref``s resolve against."""

    catalog_id: str
    """The basic catalog's ``$id``: stamped as ``catalogId`` and advertised in ``supportedCatalogIds``."""

    extension_uri: str
    """The A2A extension URI a client requests to agree on A2UI."""

    capabilities_key: str
    """The request metadata key carrying the client's capabilities."""

    capabilities_version_key: str
    """The key nested under :attr:`capabilities_key` that holds this version's capabilities."""

    data_model_key: str
    """The request metadata key carrying the client's per-surface data models."""

    catalog_protocol_version: str
    """The major.minor a custom catalog's ``protocolVersion`` must name; a catalog without one is ``"0.9"``
    (v1_0 a2ui_protocol.md L635: "defaults to `"0.9"` if omitted, required for catalogs targeting `1.0` and beyond")."""

    catalog_definition_file: str | None
    """Vendored schema every custom catalog must satisfy; v0.9 publishes none."""

    rules_file: str | None
    """Vendored basic-catalog rules file, or ``None`` when the catalog embeds them as ``instructions``."""

    per_surface_catalogs: bool
    """Components and functions resolve against the catalog their surface was created with."""

    restricted_catalog_refs: bool
    """A catalog may ``$ref`` only what it declares and the listed standard types."""

    function_rpc: bool
    """``callRendererFunction`` / ``callAgentFunction`` and their responses exist."""

    entity_naming_rules: bool
    """Component ids and a catalog's component, function, property and argument names are UAX #31 identifiers,
    and ``Surface`` is a reserved component name (v1_0 a2ui_protocol.md, "Catalog Entity Naming Rules")."""

    turn_scoped_capabilities: bool
    """A request without this version's renderer capabilities offers no A2UI, so the agent sends none in reply:
    "Capabilities carried per message are scoped to that message's turn, and omitting them withdraws A2UI for that
    turn" (v1_0 a2ui_protocol.md, "Renderer capabilities"; the A2A extension's "Capability scoping" states it for
    A2A). Every ag2 transport carries them per request. Before v1.0 the capabilities are optional."""

    @property
    def server_to_client_keys(self) -> frozenset[str]:
        """The payload key of each message kind an agent may stream in this version."""
        keys = server_to_client_payload_keys()
        return keys if self.function_rpc else keys - _FUNCTION_RPC_MESSAGE_KEYS


_FUNCTION_RPC_MESSAGE_KEYS = frozenset({"callRendererFunction", "agentFunctionResponse"})
"""The agent→renderer message kinds that exist only with function RPC (v1.0)."""


_V0_9_BASE = "https://a2ui.org/specification/v0_9/"
_V1_0_BASE = "https://a2ui.org/specification/v1_0/"

# v0.9.1 is a backward-compatible patch over v0.9: same spec files, schema ``$id``s, catalog
# (the upstream v0.9.1 catalog file declares the v0_9 ``$id``) and capabilities key. Upstream's
# v0.9.1 files differ only in accepting ``"v0.9.1"`` as the ``version`` and in lacking the newer
# ``List.listStyle`` (see ``v0_9/spec/README.md``).
PROTOCOLS: dict[A2UIVersion, A2UIProtocol] = {
    "v0.9": A2UIProtocol(
        version="v0.9",
        wire_version="v0.9",
        accepted_wire_versions=("v0.9",),
        package_dir="v0_9",
        server_to_client_file="server_to_client.json",
        schema_base_uri=_V0_9_BASE,
        catalog_id=f"{_V0_9_BASE}catalogs/basic/catalog.json",
        extension_uri="https://a2ui.org/a2a-extension/a2ui/v0.9",
        capabilities_key=A2UI_CLIENT_CAPABILITIES_METADATA_KEY,
        capabilities_version_key="v0.9",
        data_model_key=A2UI_CLIENT_DATA_MODEL_METADATA_KEY,
        catalog_protocol_version="0.9",
        catalog_definition_file=None,
        rules_file="basic_catalog_rules.txt",
        per_surface_catalogs=False,
        restricted_catalog_refs=False,
        function_rpc=False,
        entity_naming_rules=False,
        turn_scoped_capabilities=False,
    ),
    "v0.9.1": A2UIProtocol(
        version="v0.9.1",
        wire_version="v0.9.1",
        accepted_wire_versions=("v0.9", "v0.9.1"),
        package_dir="v0_9",
        server_to_client_file="server_to_client.json",
        schema_base_uri=_V0_9_BASE,
        catalog_id=f"{_V0_9_BASE}catalogs/basic/catalog.json",
        extension_uri="https://a2ui.org/a2a-extension/a2ui/v0.9.1",
        capabilities_key=A2UI_CLIENT_CAPABILITIES_METADATA_KEY,
        capabilities_version_key="v0.9",
        data_model_key=A2UI_CLIENT_DATA_MODEL_METADATA_KEY,
        catalog_protocol_version="0.9",
        catalog_definition_file=None,
        rules_file="basic_catalog_rules.txt",
        per_surface_catalogs=False,
        restricted_catalog_refs=False,
        function_rpc=False,
        entity_naming_rules=False,
        turn_scoped_capabilities=False,
    ),
    "v1.0": A2UIProtocol(
        version="v1.0",
        wire_version="1.0",
        accepted_wire_versions=("1.0",),
        package_dir="v1_0",
        server_to_client_file="agent_to_renderer.json",
        schema_base_uri=_V1_0_BASE,
        catalog_id=f"{_V1_0_BASE}catalogs/basic/catalog.json",
        extension_uri="https://a2ui.org/a2a-extension/a2ui/v1.0",
        capabilities_key=A2UI_RENDERER_CAPABILITIES_METADATA_KEY,
        capabilities_version_key="1.0",
        data_model_key=A2UI_RENDERER_DATA_MODEL_METADATA_KEY,
        catalog_protocol_version="1.0",
        catalog_definition_file="catalog_definition.json",
        rules_file=None,
        per_surface_catalogs=True,
        restricted_catalog_refs=True,
        function_rpc=True,
        entity_naming_rules=True,
        turn_scoped_capabilities=True,
    ),
}

DEFAULT_PROTOCOL = PROTOCOLS["v0.9"]


def get_protocol(version: str) -> A2UIProtocol:
    """The descriptor for ``version``; raises ``ValueError`` for an unsupported one."""
    if version not in PROTOCOLS:
        raise ValueError(f"Unsupported A2UI protocol version: {version!r}. Supported versions: {', '.join(PROTOCOLS)}")
    return PROTOCOLS[version]


__all__ = (
    "A2UI_CLIENT_CAPABILITIES_METADATA_KEY",
    "A2UI_CLIENT_DATA_MODEL_METADATA_KEY",
    "A2UI_RENDERER_CAPABILITIES_METADATA_KEY",
    "A2UI_RENDERER_DATA_MODEL_METADATA_KEY",
    "DEFAULT_PROTOCOL",
    "PROTOCOLS",
    "A2UIProtocol",
    "get_protocol",
)
