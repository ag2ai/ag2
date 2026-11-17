# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
from collections.abc import Collection
from dataclasses import dataclass
from urllib.parse import urljoin

from referencing import Registry, Resource

from .._graph import ComponentTypes, component_types
from .._types import A2UIVersion, JsonObject, JsonSchema, JsonValue
from .._versions import A2UIProtocol, get_protocol
from ..actions import A2UIEventAction
from ..constants import A2UI_JSON_CLOSE_TAG, A2UI_JSON_OPEN_TAG
from ..diagnostics import (
    AGENT_CANNOT_CALL_FORMAT,
    COMPONENT_ID_FORMAT,
    NO_DEFAULT_CATALOG_FORMAT,
    RENDERER_CANNOT_CALL_FORMAT,
    RESERVED_SURFACE_FORMAT,
)
from ._custom_catalog import _RESERVED_COMPONENT_NAME, _load_custom_catalog, _xid_property_names
from ._json import _as_object, _pointer_key, _pointer_token, _resource
from ._spec import _instructions_of, _vendored_spec

_IMPLICIT_FUNCTION_SCHEMA_URI = "urn:ag2:a2ui:implicit-renderer-function"
_IMPLICIT_COMPONENT_SCHEMA_URI = "urn:ag2:a2ui:implicit-component"


# A schema with any of these can take properties from outside its own ``properties``.
_COMPOSING_KEYWORDS = ("$ref", "allOf", "oneOf", "if", "dependentSchemas", "patternProperties", "additionalProperties")


def _closed_objects(schema: JsonValue) -> JsonValue:
    """Rewrite ``unevaluatedProperties: false`` as ``additionalProperties: false`` on self-contained objects.

    ``unevaluatedProperties`` makes ``jsonschema`` re-walk whatever matched, which compounds with every nested call.
    """
    if isinstance(schema, list):
        return [_closed_objects(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    closed = {key: _closed_objects(value) for key, value in schema.items()}
    branches = closed.get("anyOf")
    adds_nothing = (
        all(isinstance(b, dict) and set(b) <= {"required"} for b in branches) if isinstance(branches, list) else True
    )
    if (
        closed.get("unevaluatedProperties") is False
        and isinstance(closed.get("properties"), dict)
        and adds_nothing
        and not any(key in closed for key in _COMPOSING_KEYWORDS)
    ):
        del closed["unevaluatedProperties"]
        closed["additionalProperties"] = False
    return closed


def _component_description(comp_def: JsonSchema, envelope: frozenset[str]) -> str:
    """A custom component's prompt line: its ``description``, else the properties it adds to ``envelope``."""
    raw_desc = comp_def.get("description")
    if isinstance(raw_desc, str) and raw_desc:
        return raw_desc

    props: set[str] = set()
    all_of = comp_def.get("allOf")
    if isinstance(all_of, list):
        for item in all_of:
            item_props = _as_object(item.get("properties")) if isinstance(item, dict) else {}
            props.update(k for k in item_props if k not in envelope)
    props.update(k for k in _as_object(comp_def.get("properties")) if k not in envelope)
    return f"Properties: {', '.join(sorted(props))}" if props else ""


_BASIC_COMPONENT_REF_PREFIX = "basic_catalog.json#/components/"
"""How a catalog used in place of the basic one (before v1.0) reaches a basic component."""


def _example_component_types(messages: JsonValue) -> set[str]:
    """The ``component`` types the prompt example's messages use."""
    types: set[str] = set()
    for message in messages if isinstance(messages, list) else ():
        for payload in _as_object(message).values():
            components = _as_object(payload).get("components")
            for component in components if isinstance(components, list) else ():
                kind = _as_object(component).get("component")
                if isinstance(kind, str):
                    types.add(kind)
    return types


def _catalog_definition_ref(catalog_id: str, collection: str, key: str) -> str:
    return f"{catalog_id}#/{collection}/{_pointer_token(key)}"


_COMPONENT_CATALOG_KEY = "@catalogId"
"""Where a v1.0 component names its catalog (an ``@`` envelope key, a2ui-project/a2ui#4044)."""
_FUNCTION_CATALOG_KEY = "catalogId"
"""Where a function call names its catalog (``FunctionCommon.catalogId``)."""


def _catalog_selection(catalog_id: str, *, explicit: bool, key: str) -> JsonSchema:
    if explicit:
        return {"properties": {key: {"const": catalog_id}}, "required": [key]}
    return {"not": {"required": [key]}}


def _catalog_component_variant(catalog: JsonSchema, *, explicit: bool) -> JsonSchema:
    """Check a component against the catalog's ``anyComponent`` when it selects this catalog.

    Like a function variant, a component that does not select the catalog fails only through ``else``, so
    diagnostics skip the variant instead of reporting its missing ``@catalogId``.
    """
    catalog_id = str(catalog["$id"])
    selection = _catalog_selection(catalog_id, explicit=explicit, key=_COMPONENT_CATALOG_KEY)
    return {"if": selection, "then": {"$ref": f"{catalog_id}#/$defs/anyComponent"}, "else": selection}


_NO_DEFAULT_CATALOG: JsonSchema = {"format": NO_DEFAULT_CATALOG_FORMAT}
"""What an implicit component or function call meets on a v1.0 surface created without a default ``catalogId``.

Upstream resolution has no fallback: such a component or call does not resolve (v1_0 a2ui_protocol.md).
"""


_CALL_ENVELOPE = ("@call", _FUNCTION_CATALOG_KEY)

# Keywords a function definition may carry and still be closed by listing its ``properties``; anything else
# (``anyOf``, ``not``, ``then``, ...) may admit properties it does not list.
_PLAIN_DEFINITION_KEYWORDS = frozenset((
    "type",
    "title",
    "description",
    "returnType",
    "properties",
    "required",
    "allowedCallers",
    "requiresUserActivation",
))


def _call_tag_is(tag: str) -> JsonSchema:
    """Match a call whose ``@call`` is ``tag``."""
    return {"properties": {"@call": {"const": tag}}, "required": ["@call"]}


def _call_tag(key: str, definition: JsonValue) -> str:
    """The ``@call`` value a function answers to: the ``const`` the spec requires, else its catalog key."""
    properties = _as_object(_as_object(definition).get("properties"))
    tag = _as_object(properties.get("@call")).get("const")
    return tag if isinstance(tag, str) else key


def _callable_by(definition: JsonValue, *, agent_initiated: bool) -> bool:
    """Whether the function's ``allowedCallers`` admits a caller in this position."""
    callers = _as_object(definition).get("allowedCallers", "rendererOnly")
    if agent_initiated:
        return callers in ("agentOnly", "rendererOrAgent")
    return callers != "agentOnly"


def _function_body(ref: str, definition: JsonValue) -> JsonSchema:
    """The function's schema, closed over the properties a call may carry: the envelope and its own."""
    body = _as_object(definition)
    properties = _as_object(body.get("properties"))
    if properties and set(body) <= _PLAIN_DEFINITION_KEYWORDS:
        declared: JsonObject = dict.fromkeys((*_CALL_ENVELOPE, *properties), True)
        return {"allOf": [{"$ref": ref}], "properties": declared, "additionalProperties": False}
    envelope: JsonObject = dict.fromkeys(_CALL_ENVELOPE, True)
    return {"allOf": [{"properties": envelope}, {"$ref": ref}], "unevaluatedProperties": False}


def _catalog_function_variant(
    catalog: JsonSchema, *, agent_initiated: bool, explicit: bool, owned: Collection[str] | None = None
) -> JsonSchema:
    """Validate a call against only the function its ``@call`` names, among the ``owned`` ones (default: all).

    Trying every function's ``args`` compounds with each level of nested calls.
    """
    catalog_id = str(catalog["$id"])
    functions = _as_object(catalog.get("functions"))
    tags = {key: _call_tag(key, definition) for key, definition in functions.items()}
    callable_here = {
        key: _callable_by(definition, agent_initiated=agent_initiated) for key, definition in functions.items()
    }
    answered = [key for key in functions if owned is None or key in owned]
    dispatch: list[JsonValue] = []
    for key in answered:
        ref = _catalog_definition_ref(catalog_id, "functions", key)
        body = _function_body(ref, functions[key])
        if not callable_here[key]:
            # Report malformed arguments before the caller restriction.
            refusal = AGENT_CANNOT_CALL_FORMAT if agent_initiated else RENDERER_CANNOT_CALL_FORMAT
            body = {"allOf": [body, {"if": {"$ref": ref}, "then": {"format": refusal}}]}
        dispatch.append({"if": _call_tag_is(tags[key]), "then": body})
    selection = _catalog_selection(catalog_id, explicit=explicit, key=_FUNCTION_CATALOG_KEY)
    return {
        "if": selection,
        "then": {
            "allOf": [{"oneOf": [_call_tag_is(tag) for tag in dict.fromkeys(tags[key] for key in answered)]}, *dispatch]
        },
        "else": selection,
    }


def _owned_functions(catalogs: list[JsonSchema]) -> list[tuple[JsonSchema, set[str] | None]]:
    """Pair each catalog with the functions no earlier catalog defines identically (``None``: all of them).

    A custom catalog contains the basic one, so checking both would repeat the shared functions at every level.
    """
    seen: dict[str, JsonValue] = {}
    pairs: list[tuple[JsonSchema, set[str] | None]] = []
    for index, catalog in enumerate(catalogs):
        functions = _as_object(catalog.get("functions"))
        owned = {key for key, definition in functions.items() if seen.get(key) != definition}
        seen.update(functions)
        if index == 0 or owned:
            pairs.append((catalog, None if index == 0 else owned))
    return pairs


def _registry_for_surface_catalog(registry: Registry, catalog_id: str | None) -> Registry:
    """Select the surface's default catalog for components and function calls in a request-local registry.

    ``None`` is a surface without a default: only components and calls that name their catalog resolve.
    """
    catalog = registry.get(catalog_id) if catalog_id is not None else None
    function_schema: JsonSchema
    component_schema: JsonSchema
    if catalog is None:
        # An unknown catalog is reported on its ``createSurface``; what it holds cannot be checked. Evaluating every
        # property keeps the closed ``Component`` envelope from reporting them as unexpected.
        function_schema = component_schema = (
            _NO_DEFAULT_CATALOG if catalog_id is None else {"additionalProperties": True}
        )
    else:
        function_schema = _catalog_function_variant(catalog.contents, agent_initiated=False, explicit=False)
        component_schema = _catalog_component_variant(catalog.contents, explicit=False)
    return registry.with_resources([
        (_IMPLICIT_FUNCTION_SCHEMA_URI, _resource(function_schema)),
        (_IMPLICIT_COMPONENT_SCHEMA_URI, _resource(component_schema)),
    ])


_OPEN_CATALOG: JsonSchema = {"$defs": {"anyComponent": {}, "anyFunction": {}, "theme": {}}}
"""A catalog that accepts any component, function and theme: what a surface on an unknown catalog is checked by."""


def _registry_for_unknown_catalog(registry: Registry, schema_base_uri: str) -> Registry:
    """Before v1.0, the message schema reaches the agent's catalog as ``catalog.json``; open it for a surface whose
    catalog is unknown, since what that catalog holds cannot be checked."""
    return registry.with_resources([(f"{schema_base_uri}catalog.json", _resource(_OPEN_CATALOG))])


def _with_identifier_component_ids(common_types: JsonSchema) -> JsonSchema:
    """``ComponentId`` (so every id and every reference to one) checked as a UAX #31 identifier."""
    definitions = dict(_as_object(common_types.get("$defs")))
    definitions["ComponentId"] = {**_as_object(definitions.get("ComponentId")), "format": COMPONENT_ID_FORMAT}
    return {**common_types, "$defs": definitions}


@dataclass(frozen=True, slots=True)
class _AgentCatalogs:
    """The catalogs the agent serves, each as written; none is merged into another.

    v1.0 catalogs are mixable (a2ui_protocol.md, "Mixable catalogs and component resolution logic"): the basic and
    the custom catalog are both served, the prompt's ``createSurface`` names the basic one as the surface default,
    and a custom component or function names its catalog. Before v1.0 a surface has one catalog and a custom
    catalog is used "in place of the basic catalog" (v0_9 a2ui_protocol.md, "Swappable Catalogs & Validation"); it
    may still ``$ref`` basic definitions. ``basic`` is ``None`` only for a v1.0 turn whose renderer does not support
    the basic catalog (see :meth:`A2UISchemaManager.restricted_to`).
    """

    basic: JsonSchema | None
    custom: JsonSchema | None
    mixable: bool

    @property
    def surface_catalog(self) -> JsonSchema:
        """The catalog the prompt's ``createSurface`` names as the surface default."""
        if self.custom is not None and (not self.mixable or self.basic is None):
            return self.custom
        if self.basic is None:
            raise ValueError("an agent serves at least one catalog")
        return self.basic

    @property
    def id(self) -> str:
        return str(self.surface_catalog["$id"])

    @property
    def custom_components(self) -> JsonObject:
        return _as_object(self.custom.get("components")) if self.custom is not None else {}

    @property
    def by_id(self) -> dict[str, JsonSchema]:
        """Every catalog a surface may name, by ``$id``."""
        if not self.mixable:
            return {str(self.surface_catalog["$id"]): self.surface_catalog}
        return {str(catalog["$id"]): catalog for catalog in (self.basic, self.custom) if catalog is not None}


def _with_any_definitions(catalog: JsonSchema) -> JsonSchema:
    """``catalog`` with the ``anyComponent`` / ``anyFunction`` the message schemas reference, built from its own
    components and functions where it does not define them; a catalog without any accepts none (``false``)."""
    defs = dict(_as_object(catalog.get("$defs")))
    components = _as_object(catalog.get("components"))
    functions = _as_object(catalog.get("functions"))
    if "anyComponent" not in defs:
        defs["anyComponent"] = (
            {
                "oneOf": [{"$ref": _catalog_definition_ref("", "components", name)} for name in components],
                "discriminator": {"propertyName": "component"},
            }
            if components
            else False
        )
    if "anyFunction" not in defs:
        defs["anyFunction"] = (
            {"oneOf": [{"$ref": _catalog_definition_ref("", "functions", name)} for name in functions]}
            if functions
            else False
        )
    return {**catalog, "$defs": defs}


def _with_catalog_function_dispatch(
    common_types: JsonSchema, catalogs: dict[str, JsonSchema], active: JsonSchema
) -> JsonSchema:
    """``FunctionCall`` / ``AgentFunctionCall`` that check a call against the function its catalog defines.

    Restricting references rather than walking component JSON keeps data models, extension metadata
    and custom literals that happen to hold an ``@call`` key out of it. Renderer-evaluated bindings
    exclude agent-only functions; the top-level renderer RPC uses the agent-callable definition.
    """
    definitions = dict(_as_object(common_types.get("$defs")))
    definitions["IndexSystemFunction"] = _closed_objects(definitions.get("IndexSystemFunction"))
    function_call = _as_object(definitions.get("FunctionCall"))
    for name, agent_initiated in (("FunctionCall", False), ("AgentFunctionCall", True)):
        variants: list[JsonValue] = [
            _catalog_function_variant(catalog, agent_initiated=agent_initiated, explicit=True)
            for catalog in catalogs.values()
        ]
        if agent_initiated:
            variants.append(_catalog_function_variant(active, agent_initiated=True, explicit=False))
        else:
            variants.append({"$ref": _IMPLICIT_FUNCTION_SCHEMA_URI})
            variants.append({"$ref": "#/$defs/IndexSystemFunction"})
        # Each variant closes its own property set, so the call-wide ``unevaluatedProperties`` goes.
        definitions[name] = {
            **{key: value for key, value in function_call.items() if key != "unevaluatedProperties"},
            "allOf": [{"$ref": "#/$defs/FunctionCommon"}, {"oneOf": variants}],
        }
    return {**common_types, "$defs": definitions}


def _with_catalog_component_selection(server_to_client: JsonSchema, catalogs: dict[str, JsonSchema]) -> JsonSchema:
    """A ``Component`` checked against its surface's catalog, and a renderer call that names its catalog."""
    definitions = dict(_as_object(server_to_client.get("$defs")))
    component = {key: value for key, value in _as_object(definitions.get("Component")).items() if key != "allOf"}
    definitions["Component"] = {
        **component,
        # The reserved ``Surface`` gets one error saying so, not one per catalog that lacks it.
        "if": {"properties": {"component": {"const": _RESERVED_COMPONENT_NAME}}, "required": ["component"]},
        "then": {"format": RESERVED_SURFACE_FORMAT},
        "else": {
            "allOf": [
                {"$ref": "common_types.json#/$defs/ComponentCommon"},
                {
                    "oneOf": [
                        *[_catalog_component_variant(catalog, explicit=True) for catalog in catalogs.values()],
                        {"$ref": _IMPLICIT_COMPONENT_SCHEMA_URI},
                    ],
                },
            ],
        },
    }
    message = _as_object(definitions.get("CallRendererFunctionMessage"))
    message_properties = _as_object(message.get("properties"))
    payload = _as_object(message_properties.get("callRendererFunction"))
    payload_properties = _as_object(payload.get("properties"))
    call_constraints: list[JsonValue] = [
        {"$ref": "common_types.json#/$defs/AgentFunctionCall"},
        {"required": [_FUNCTION_CATALOG_KEY]},
    ]
    call: JsonObject = {**_as_object(payload_properties.get("callFunction")), "allOf": call_constraints}
    definitions["CallRendererFunctionMessage"] = {
        **message,
        "properties": {
            **message_properties,
            "callRendererFunction": {**payload, "properties": {**payload_properties, "callFunction": call}},
        },
    }
    return {**server_to_client, "$defs": definitions}


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

    @property
    def server_to_client_schema(self) -> JsonSchema:
        """The message schema responses are validated against."""
        self._ensure_loaded()
        return self._validation_schema

    def component_types(self) -> ComponentTypes:
        """How each component type of the served catalogs references and nests other components."""
        self._ensure_loaded()
        return component_types(
            *self._catalog.by_id.values(), documents={"basic_catalog.json": self._spec.basic_catalog}
        )

    def build_schema_registry(self) -> Registry:
        """The registry ``$ref``s in :attr:`server_to_client_schema` resolve against.

        Each resource sits under its own ``$id``; the alias table below adds the other names the specs use.
        """
        self._ensure_loaded()
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

    def _build_prompt_example(self) -> str:
        """Build the prompt example JSON string from the version-specific template.

        The example's components come from the basic catalog; a catalog used in place of it may lack them, and then
        the example shows only the messages that carry no components.
        """
        example = self._spec.prompt_example
        available = _as_object(self._catalog.surface_catalog.get("components"))
        if isinstance(example, list) and not _example_component_types(example) <= available.keys():
            example = [message for message in example if not _example_component_types([message])]
        raw = json.dumps(example, separators=(",", ": "))
        return self._fill_template(raw)

    def _component_lines(self, components: JsonObject, envelope: frozenset[str]) -> str:
        """One described line per component; a bare ``$ref`` into the basic catalog is described by its target."""
        basic = _as_object(self._spec.basic_catalog.get("components"))
        lines = ""
        for name in sorted(components):
            definition = _as_object(components.get(name))
            ref = definition.get("$ref")
            if isinstance(ref, str) and ref.startswith(_BASIC_COMPONENT_REF_PREFIX):
                definition = _as_object(basic.get(_pointer_key(ref.removeprefix(_BASIC_COMPONENT_REF_PREFIX))))
            if desc := _component_description(definition, envelope):
                lines += f"- **{name}**: {desc}\n"
        return lines + "\n" if lines else ""

    def _fill_template(self, text: str) -> str:
        return text.replace("{catalog_id}", self.catalog_id).replace("{version_string}", self._protocol.wire_version)

    def generate_prompt_section(
        self,
        include_schema: bool = True,
        include_rules: bool = True,
        actions: list[A2UIEventAction] | None = None,
    ) -> str:
        """Generate the A2UI portion of the system prompt."""
        self._ensure_loaded()
        protocol_version = self._protocol.version
        sections: list[str] = []

        example_json = self._build_prompt_example()

        sections.append(
            f"## A2UI Response Format ({protocol_version})\n\n"
            f"You can generate rich UI responses using the A2UI {protocol_version} protocol. "
            "When you want to display a UI, write your conversational text first, "
            f"then wrap a JSON array of A2UI message objects between the "
            f"`{A2UI_JSON_OPEN_TAG}` and `{A2UI_JSON_CLOSE_TAG}` tags.\n\n"
            "Example response format:\n"
            "```\n"
            "Here is the UI you requested.\n"
            f"{A2UI_JSON_OPEN_TAG}\n"
            f"{example_json}\n"
            f"{A2UI_JSON_CLOSE_TAG}\n"
            "```"
        )

        message_types = self._fill_template(self._spec.prompt_message_types)
        sections.append(f"\n\n## A2UI Message Types\n\n{message_types}")

        catalog = self._catalog
        # The properties every component shares: the ``component`` discriminator and ``ComponentCommon``
        # (``id`` in every version, then v0.9 ``accessibility`` or v1.0 ``@catalogId``, ...).
        common = _as_object(_as_object(self._common_types.get("$defs")).get("ComponentCommon"))
        envelope = frozenset({"component", *_as_object(common.get("properties"))})
        default_components = _as_object(catalog.surface_catalog.get("components"))
        mixes_catalogs = catalog.mixable and catalog.basic is not None and catalog.custom is not None
        mixed_in = catalog.custom_components if mixes_catalogs else {}
        if default_components or mixed_in:
            comp_section = "\n\n## Available Components\n\n"
            if catalog.custom is None:
                comp_section += f"**Basic catalog:** {', '.join(sorted(default_components))}\n\n"
            elif mixes_catalogs and catalog.custom is not None:
                comp_section += f"**Basic catalog** (the surface default): {', '.join(sorted(default_components))}\n\n"
                if mixed_in:
                    custom_id = str(catalog.custom["$id"])
                    comp_section += (
                        f"**Custom catalog** (`{custom_id}`): {', '.join(sorted(mixed_in))}\n\n"
                        f'These are not in the surface\'s default catalog: give each of them `"@catalogId": "{custom_id}"`, '
                        f'and each call to one of this catalog\'s functions `"catalogId": "{custom_id}"`.\n\n'
                    )
                    comp_section += self._component_lines(mixed_in, envelope)
            else:
                comp_section += f"**Components:** {', '.join(sorted(default_components))}\n\n"
                comp_section += self._component_lines(default_components, envelope)

            comp_section += (
                "Each component uses a flat discriminator format:\n"
                '```json\n{"id": "myId", "component": "Text", "text": "Hello"}\n```\n\n'
                "Components are referenced by `id`. Use `children` (array of ids) for layout components (Column, Row). "
                "Use `child` (single id) for wrapper components (Card, Button)."
            )
            sections.append(comp_section)

        if include_rules:
            rules_parts: list[str] = []
            # A catalog used in place of the basic one replaces its rules too.
            if self._spec.catalog_rules and catalog.surface_catalog is catalog.basic:
                rules_parts.append(self._spec.catalog_rules)
            if self._custom_catalog_rules:
                rules_parts.append(self._custom_catalog_rules)
            elif catalog.custom is not None and (custom_instructions := _instructions_of(catalog.custom)):
                rules_parts.append(custom_instructions)
            if rules_parts:
                sections.append("\n\n## Component Rules\n\n" + "\n\n".join(rules_parts))

        if actions:
            action_lines = ["\n\n## Available Actions\n"]
            action_lines.append("The following actions can be triggered by buttons in the UI:\n")

            action_lines.append("### Server Events\n")
            for a in actions:
                desc = f": {a.description}" if a.description else ""
                action_lines.append(f"- `{a.name}`{desc}")
                if a.example_context:
                    action_lines.append(f"  Context: {json.dumps(a.example_context)}")
            action_lines.append(
                "\nTo create a button that triggers a server event:\n"
                "```json\n"
                '{"id": "action_btn", "component": "Button", "child": "action_btn_text", '
                '"action": {"event": {"name": "<action_name>", "context": {...}}}},\n'
                '{"id": "action_btn_text", "component": "Text", "text": "Button Label"}\n'
                "```"
            )

            sections.append("\n".join(action_lines))

        if include_schema:
            schema_str = json.dumps(self._spec.server_to_client, indent=2)
            sections.append(f"\n\n## A2UI Message Schema ({protocol_version})\n\n```json\n{schema_str}\n```")

        return "\n".join(sections)
