# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Assemble the validation schema and registry from the served catalogs: catalog selection, function dispatch."""

from collections.abc import Collection
from dataclasses import dataclass

from referencing import Registry

from .._types import JsonObject, JsonSchema, JsonValue
from ..diagnostics import (
    AGENT_CANNOT_CALL_FORMAT,
    COMPONENT_ID_FORMAT,
    NO_DEFAULT_CATALOG_FORMAT,
    RENDERER_CANNOT_CALL_FORMAT,
    RESERVED_SURFACE_FORMAT,
)
from ._custom_catalog import _RESERVED_COMPONENT_NAME
from ._json import _as_object, _pointer_token, _resource

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
