# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Load the A2UI schemas an agent serves and derive from them the validation schema, registry and prompt section."""

import copy
import os
from collections.abc import Collection
from dataclasses import dataclass
from urllib.parse import urljoin

from referencing import Registry, Resource

from .._graph import ComponentTypes, component_types
from .._types import A2UIVersion, JsonSchema
from .._versions import A2UIProtocol, get_protocol
from ..actions import A2UIEventAction
from ._assembly import (
    _IMPLICIT_COMPONENT_SCHEMA_URI,
    _IMPLICIT_FUNCTION_SCHEMA_URI,
    _AgentCatalogs,
    _catalog_component_variant,
    _catalog_function_variant,
    _closed_objects,
    _owned_functions,
    _with_any_definitions,
    _with_catalog_component_selection,
    _with_catalog_function_dispatch,
    _with_identifier_component_ids,
)

# ``parser.py`` narrows the registry per surface with these.
from ._assembly import _registry_for_surface_catalog as _registry_for_surface_catalog
from ._assembly import _registry_for_unknown_catalog as _registry_for_unknown_catalog
from ._custom_catalog import _load_custom_catalog, _xid_property_names
from ._json import _as_object, _resource
from ._prompt import _prompt_section
from ._spec import _vendored_spec


@dataclass(frozen=True, slots=True)
class A2UIValidationSchemas:
    """Everything a validating :class:`~ag2.a2ui.parser.A2UIResponseParser` checks responses against."""

    server_to_client_schema: JsonSchema
    """The message schema responses are validated against."""
    schema_registry: Registry
    """The registry ``$ref``s in ``server_to_client_schema`` resolve against."""
    component_types: ComponentTypes
    """How each component type of the served catalogs references and nests other components."""
    catalog_ids: tuple[str, ...]
    """Every catalog a surface may name."""


class A2UISchemaManager:
    """Loads vendored A2UI schemas for a protocol version and builds the
    system-prompt section that instructs an LLM to emit valid A2UI output.

    No catalog is merged into another. Before v1.0 a custom catalog (file path
    or dict) is served in place of the basic catalog; in v1.0 both are served
    and a custom component or function names its catalog (see ADR 0040). In
    v1.0 the custom catalog's own ``$ref``s may target only what it declares
    and the standard types in ``common_types.json`` (a2ui_protocol.md,
    "Restricted $ref Targets"), not the basic catalog's components.
    """

    def __init__(
        self,
        protocol_version: A2UIVersion = "v0.9",
        custom_catalog: "str | os.PathLike[str] | JsonSchema | None" = None,
        custom_catalog_rules: str | None = None,
    ) -> None:
        """Initialize the schema manager.

        Args:
            protocol_version: The A2UI protocol version: "v0.9" (default), "v0.9.1", or "v1.0".
            custom_catalog: A custom catalog, served in place of the basic one before
                v1.0 and beside it in v1.0. Can be:
                - A file path (str or PathLike) to a JSON catalog file
                - A dict with the catalog schema directly
                It must declare a ``catalogId`` (its id in A2UI messages; a ``$id``, if any, must equal it) and,
                for v1.0, ``"protocolVersion": "1.0"``; a v1.0 catalog must satisfy ``catalog_definition.json``.
            custom_catalog_rules: Plain-text rules for the custom catalog components. In
                v1.0 they follow the basic catalog's rules in the system prompt; before
                v1.0 they replace them, as the custom catalog replaces the basic one.
        """
        self._protocol = get_protocol(protocol_version)
        self._custom_catalog_input = custom_catalog
        self._custom_catalog_rules = custom_catalog_rules or ""
        self._served: frozenset[str] | None = None
        self._loaded = False

    def restricted_to(self, catalog_ids: Collection[str]) -> "A2UISchemaManager | None":
        """This agent's catalogs that ``catalog_ids`` names, as a manager serving only those; ``None`` if none.

        A renderer advertises the catalogs it supports, and only those become active for its turn, as the
        official SDK's ``resolve_catalogs`` does. Returns ``self`` when every catalog is named.
        """
        served = frozenset(catalog_ids).intersection(self.catalog_ids)
        if not served:
            return None
        if len(served) == len(self.catalog_ids):
            return self
        restricted = A2UISchemaManager(
            protocol_version=self._protocol.version,
            custom_catalog=self._custom_catalog_input,
            custom_catalog_rules=self._custom_catalog_rules,
        )
        restricted._served = served
        return restricted

    def _ensure_loaded(self) -> None:
        """Read the vendored specs and the custom catalog on first use, and derive the validation schemas."""
        if self._loaded:
            return
        protocol = self._protocol
        # A copy, so no manager (or a caller holding one of its schemas) can change another's.
        spec = copy.deepcopy(_vendored_spec(protocol))
        self._spec = spec
        custom = (
            _load_custom_catalog(self._custom_catalog_input, protocol)
            if self._custom_catalog_input is not None
            else None
        )

        basic = spec.basic_catalog
        common_types = _as_object(_xid_property_names(spec.common_types))
        if protocol.entity_naming_rules:
            common_types = _with_identifier_component_ids(common_types)
        server_to_client = spec.server_to_client
        if protocol.per_surface_catalogs:
            # A closed function lets each call be checked against one definition (see ``_closed_objects``).
            basic = {**basic, "functions": _closed_objects(basic.get("functions"))}
        served_basic: JsonSchema | None = basic
        served_custom = _with_any_definitions(custom) if custom is not None else None
        if self._served is not None:
            # Only a v1.0 agent serves more than one catalog, so only there can a renderer leave one out.
            if str(basic["$id"]) not in self._served:
                served_basic = None
            if served_custom is not None and str(served_custom["$id"]) not in self._served:
                served_custom = None
        catalog = _AgentCatalogs(basic=served_basic, custom=served_custom, mixable=protocol.per_surface_catalogs)
        if protocol.per_surface_catalogs:
            common_types = _with_catalog_function_dispatch(common_types, catalog.by_id, catalog.surface_catalog)
            server_to_client = _with_catalog_component_selection(server_to_client, catalog.by_id)
        self._catalog = catalog
        self._common_types = common_types
        self._validation_schema = server_to_client
        self._loaded = True

    @property
    def protocol(self) -> A2UIProtocol:
        return self._protocol

    @property
    def catalog_id(self) -> str:
        self._ensure_loaded()
        return self._catalog.id

    @property
    def catalog_ids(self) -> tuple[str, ...]:
        """Every catalog a surface may name: in v1.0 the basic one and any custom one; before, the one in use."""
        self._ensure_loaded()
        return tuple(self._catalog.by_id)

    def validation_schemas(self) -> A2UIValidationSchemas:
        """The schemas, registry and component types responses are validated against."""
        self._ensure_loaded()
        return A2UIValidationSchemas(
            server_to_client_schema=self._validation_schema,
            schema_registry=self._schema_registry(),
            component_types=self._component_types(),
            catalog_ids=self.catalog_ids,
        )

    def _component_types(self) -> ComponentTypes:
        return component_types(
            *self._catalog.by_id.values(), documents={"basic_catalog.json": self._spec.basic_catalog}
        )

    def _schema_registry(self) -> Registry:
        """Each resource sits under its own ``$id``; the alias table below adds the other names the specs use."""
        base = self._protocol.schema_base_uri
        common_types = _resource(self._common_types)
        catalogs = self._catalog.by_id
        resources: dict[str, Resource] = {
            str(self._validation_schema["$id"]): _resource(self._validation_schema),
            str(self._common_types["$id"]): common_types,
            **{catalog_id: _resource(catalog) for catalog_id, catalog in catalogs.items()},
            # Aliases. ``catalog.json`` next to the message schema names the agent's catalog.
            f"{base}catalog.json": _resource(self._catalog.surface_catalog),
            # Before v1.0 a custom catalog may reach the basic one as ``basic_catalog.json``, resolved
            # against ``catalog.json`` above.
            f"{base}basic_catalog.json": _resource(self._spec.basic_catalog),
        }
        # Catalogs reference the shared types by a bare ``common_types.json``, resolved next to their ``$id``.
        for catalog_id in catalogs:
            resources.setdefault(urljoin(catalog_id, "common_types.json"), common_types)
        if self._protocol.per_surface_catalogs:
            # With no surface context a stateless validator can only establish that an implicit call
            # or component is valid in at least one configured catalog. The parser narrows these
            # resources when ``createSurface`` names the catalog.
            resources[_IMPLICIT_FUNCTION_SCHEMA_URI] = _resource({
                "anyOf": [
                    _catalog_function_variant(catalog, agent_initiated=False, explicit=False, owned=owned)
                    for catalog, owned in _owned_functions(list(catalogs.values()))
                ]
            })
            # Prefer the agent's catalog's diagnostics when both branches fail.
            resources[_IMPLICIT_COMPONENT_SCHEMA_URI] = _resource({
                "anyOf": [
                    _catalog_component_variant(catalog, explicit=False) for catalog in reversed(catalogs.values())
                ],
            })
        return Registry().with_resources(resources.items())

    def generate_prompt_section(
        self,
        include_schema: bool = True,
        include_rules: bool = True,
        actions: list[A2UIEventAction] | None = None,
    ) -> str:
        """Generate the A2UI portion of the system prompt."""
        self._ensure_loaded()
        return _prompt_section(
            self._protocol,
            self._spec,
            self._catalog,
            self._common_types,
            self._custom_catalog_rules,
            include_schema=include_schema,
            include_rules=include_rules,
            actions=actions,
        )


__all__ = ("A2UISchemaManager", "A2UIValidationSchemas")
