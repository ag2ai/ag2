# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import os
from collections.abc import Collection
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from urllib.parse import urljoin

from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from ._graph import ComponentReferences, component_references
from ._types import A2UIVersion, JsonObject, JsonSchema, JsonValue
from ._versions import A2UIProtocol, get_protocol
from .actions import A2UIEventAction
from .constants import A2UI_JSON_CLOSE_TAG, A2UI_JSON_OPEN_TAG
from .diagnostics import (
    AGENT_CANNOT_CALL_FORMAT,
    COMPONENT_ID_FORMAT,
    NO_DEFAULT_CATALOG_FORMAT,
    RENDERER_CANNOT_CALL_FORMAT,
    RESERVED_SURFACE_FORMAT,
    UAX31_IDENTIFIER,
    XID_FORMAT,
)

_PACKAGE_DIR = Path(__file__).parent


def _as_object(value: "JsonValue | None") -> JsonObject:
    """Narrow a ``JsonValue`` to a JSON object, returning ``{}`` when it is not a dict.

    JSON Schema documents are dynamic ``dict[str, JsonValue]`` trees, so reading a
    nested member (``components`` / ``functions`` / ``$defs`` / ``properties``)
    yields a broad ``JsonValue``. Funnelling those reads through this helper keeps
    the schema-munging code statically typed without scattering ``isinstance``
    checks or ``cast``s at every access.
    """
    return value if isinstance(value, dict) else {}


# v1.0 constrains extension keys with Unicode property escapes, which Python's ``re``
# (what ``jsonschema`` matches patterns with) rejects. Check those keys with a format
# instead, at load time, leaving the vendored files byte-faithful.
_XID_KEYS = r"^[\p{XID_Start}_][\p{XID_Continue}]*$"
_IMPLICIT_FUNCTION_SCHEMA_URI = "urn:ag2:a2ui:implicit-renderer-function"
_IMPLICIT_COMPONENT_SCHEMA_URI = "urn:ag2:a2ui:implicit-component"


def _xid_property_names(schema: JsonValue) -> JsonValue:
    """Rewrite objects whose keys must be UAX #31 identifiers to a ``propertyNames`` format check."""
    if isinstance(schema, list):
        return [_xid_property_names(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    if schema.get("patternProperties") == {_XID_KEYS: {}} and schema.get("additionalProperties") is False:
        rest = {key: value for key, value in schema.items() if key not in ("patternProperties", "additionalProperties")}
        return {**rest, "propertyNames": {"format": XID_FORMAT}}
    return {key: _xid_property_names(value) for key, value in schema.items()}


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


def _instructions_of(catalog: JsonSchema) -> str:
    """Return a catalog's embedded ``instructions`` text (v1.0), or ``""``."""
    instructions = catalog.get("instructions")
    return instructions.strip() if isinstance(instructions, str) else ""


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


def _pointer_token(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def _pointer_key(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _catalog_definition_ref(catalog_id: str, collection: str, key: str) -> str:
    return f"{catalog_id}#/{collection}/{_pointer_token(key)}"


# The ``common_types.json`` definitions a v1.0 catalog may ``$ref`` (a2ui_protocol.md, "Restricted $ref Targets").
_CATALOG_COMMON_TYPES = frozenset({
    "ComponentId",
    "Child",
    "ChildList",
    "DynamicString",
    "DynamicNumber",
    "DynamicBoolean",
    "DynamicStringList",
    "DynamicValue",
    "AccessibilityAttributes",
    "CheckRule",
    "Checkable",
    "Action",
    "DataBinding",
    "FunctionCall",
})


_COMMON_TYPES_REF_PREFIX = "common_types.json#/$defs/"

# The catalog sections whose definitions are checked and may be ``$ref``-ed as ``#/<section>/<name>``.
_CATALOG_SECTIONS = ("components", "functions", "$defs")

# Keys whose values are annotations or literal data rather than subschemas, so a ``$ref`` inside them
# is not a reference. Mirrors upstream ``specification/v1_0/test/run_tests.py``.
_NON_SCHEMA_KEYS = frozenset({"metadata", "examples", "const", "default", "enum", "description"})


def _is_allowed_catalog_reference(catalog: JsonSchema, ref: str) -> bool:
    """Whether a v1.0 catalog may ``$ref`` ``ref`` (a2ui_protocol.md, "Restricted $ref Targets").

    A local target names a component, function or leaf ``$defs`` entry the catalog declares
    (``anyComponent`` / ``anyFunction`` excluded); an external one names a listed standard type
    as ``common_types.json#/$defs/<TypeName>``.
    """
    if ref.startswith(_COMMON_TYPES_REF_PREFIX):
        return ref.removeprefix(_COMMON_TYPES_REF_PREFIX) in _CATALOG_COMMON_TYPES
    for section in _CATALOG_SECTIONS:
        prefix = f"#/{section}/"
        if ref.startswith(prefix):
            name = _pointer_key(ref.removeprefix(prefix))
            if section == "$defs" and name in ("anyComponent", "anyFunction"):
                return False
            return name in _as_object(catalog.get(section))
    return False


def _check_catalog_references(catalog: JsonSchema) -> None:
    """Reject a v1.0 catalog with a ``$ref`` the spec does not allow.

    The spec makes such a catalog invalid and requires reporting it at load time, not fetching the target.
    """
    for section in _CATALOG_SECTIONS:
        for name, definition in _as_object(catalog.get(section)).items():
            _check_definition_references(catalog, definition, f"/{section}/{_pointer_token(name)}")


def _check_definition_references(catalog: JsonSchema, schema: JsonValue, pointer: str) -> None:
    if isinstance(schema, list):
        for index, item in enumerate(schema):
            _check_definition_references(catalog, item, f"{pointer}/{index}")
        return
    if not isinstance(schema, dict):
        return
    ref = schema.get("$ref")
    if isinstance(ref, str) and not _is_allowed_catalog_reference(catalog, ref):
        raise ValueError(
            f"Unsupported $ref '{ref}' at {pointer}/$ref: a v1.0 catalog may only reference the components, "
            "functions and leaf $defs it declares (#/components/..., #/functions/..., #/$defs/...) "
            f"and the standard types in {_COMMON_TYPES_REF_PREFIX}..."
        )
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            # Declared properties are subschemas whatever they are named.
            for name, subschema in value.items():
                _check_definition_references(catalog, subschema, f"{pointer}/properties/{_pointer_token(name)}")
        elif key not in _NON_SCHEMA_KEYS:
            _check_definition_references(catalog, value, f"{pointer}/{_pointer_token(key)}")


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


_RESERVED_COMPONENT_NAME = "Surface"
"""The canonical container ``createSurface`` instantiates; no catalog may define a component by that name."""


def _check_entity_names(catalog: JsonSchema) -> None:
    """Reject a v1.0 catalog whose component, function, property or argument names are not UAX #31 identifiers.

    Like the official SDK, a component's names are its key and its own ``properties``; a function's are its key
    and the properties of its ``args``.
    """
    for name, definition in _as_object(catalog.get("components")).items():
        if name == _RESERVED_COMPONENT_NAME:
            raise ValueError(
                f"Invalid custom catalog: component name {name!r} is reserved for the surface container "
                "that createSurface creates"
            )
        _check_entity_name("component name", name)
        for prop in _as_object(_as_object(definition).get("properties")):
            _check_entity_name("property name", prop, owner=f"component {name!r}")
    for name, definition in _as_object(catalog.get("functions")).items():
        _check_entity_name("function name", name)
        args = _as_object(_as_object(_as_object(definition).get("properties")).get("args"))
        for argument in _as_object(args.get("properties")):
            _check_entity_name("argument name", argument, owner=f"function {name!r}")


def _check_entity_name(kind: str, name: str, *, owner: str | None = None) -> None:
    # ``str.isidentifier`` is defined by XID_Start (plus ``_``) and XID_Continue, the sets UAX #31 names.
    if not name.isidentifier():
        of_owner = f" of {owner}" if owner else ""
        raise ValueError(f"Invalid custom catalog: {kind} {name!r}{of_owner} is not a {UAX31_IDENTIFIER}")


def _with_identifier_component_ids(common_types: JsonSchema) -> JsonSchema:
    """``ComponentId`` (so every id and every reference to one) checked as a UAX #31 identifier."""
    definitions = dict(_as_object(common_types.get("$defs")))
    definitions["ComponentId"] = {**_as_object(definitions.get("ComponentId")), "format": COMPONENT_ID_FORMAT}
    return {**common_types, "$defs": definitions}


def _load_json(path: Path) -> JsonValue:
    """Load a JSON file, raising a clear error on a missing or malformed one."""
    try:
        with open(path, encoding="utf-8") as f:
            result: JsonValue = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"A2UI file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"A2UI file {path} is not valid JSON: {e}") from e
    return result


def _load_object(path: Path) -> JsonSchema:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"A2UI file {path} must be a JSON object")
    return data


def _accept_wire_versions(schema: JsonValue, versions: tuple[str, ...]) -> JsonValue:
    """Widen every message's ``version: {const}`` to ``{enum: versions}``.

    v0.9.1 reuses the v0.9 spec files, whose messages pin ``"v0.9"``; v0.9.1 also accepts ``"v0.9.1"``.
    """
    if isinstance(schema, list):
        return [_accept_wire_versions(item, versions) for item in schema]
    if not isinstance(schema, dict):
        return schema
    result = {key: _accept_wire_versions(value, versions) for key, value in schema.items()}
    version = _as_object(_as_object(result.get("properties")).get("version"))
    if "const" in version:
        widened = {key: value for key, value in version.items() if key != "const"}
        widened["enum"] = list(versions)
        result["properties"] = {**_as_object(result["properties"]), "version": widened}
    return result


@dataclass(frozen=True, slots=True)
class _VendoredSpec:
    """One protocol version's vendored files, read once per process; each schema manager works on a deep copy."""

    server_to_client: JsonSchema
    common_types: JsonSchema
    basic_catalog: JsonSchema
    catalog_rules: str
    prompt_example: JsonValue
    prompt_message_types: str


@cache
def _vendored_spec(protocol: A2UIProtocol) -> _VendoredSpec:
    version_dir = _PACKAGE_DIR / protocol.package_dir
    spec_dir = version_dir / "spec"
    server_to_client = _load_object(spec_dir / protocol.server_to_client_file)
    if len(protocol.accepted_wire_versions) > 1:
        server_to_client = _as_object(_accept_wire_versions(server_to_client, protocol.accepted_wire_versions))
    basic_catalog = _load_object(spec_dir / "basic_catalog.json")
    rules = (
        (spec_dir / protocol.rules_file).read_text(encoding="utf-8").strip()
        if protocol.rules_file
        else _instructions_of(basic_catalog)
    )
    return _VendoredSpec(
        server_to_client=server_to_client,
        common_types=_load_object(spec_dir / "common_types.json"),
        basic_catalog=basic_catalog,
        catalog_rules=rules,
        prompt_example=_load_json(version_dir / "prompt_example.json"),
        prompt_message_types=(version_dir / "prompt_message_types.md").read_text(encoding="utf-8").strip(),
    )


def _load_custom_catalog(custom_catalog: "str | os.PathLike[str] | JsonSchema", protocol: A2UIProtocol) -> JsonSchema:
    if isinstance(custom_catalog, dict):
        catalog = custom_catalog
    else:
        path = Path(custom_catalog)
        if not path.exists():
            raise FileNotFoundError(f"Custom catalog file not found: {path}")
        catalog = _load_object(path)
    if protocol.restricted_catalog_refs:
        _check_catalog_references(catalog)
    if protocol.entity_naming_rules:
        _check_entity_names(catalog)
    if "$id" not in catalog:
        raise ValueError(
            "Custom catalog must include a '$id' field. "
            "This is used as the catalogId in A2UI createSurface messages. "
            'Example: {"$id": "https://mycompany.com/my_catalog.json", ...}'
        )
    if not isinstance(catalog["$id"], str):
        raise ValueError(
            f"Custom catalog '$id' must be a string, got {type(catalog['$id']).__name__!r}. "
            "It is used verbatim as the catalogId and as a schema-registry key."
        )
    return catalog


@dataclass(frozen=True, slots=True)
class _MergedCatalog:
    """The agent's catalog: basic plus custom, custom winning on a name collision.

    The prompt, the component listing and validation all read this one view.
    """

    basic: JsonSchema
    custom: JsonSchema | None
    schema: JsonSchema
    """The merged catalog document, under the custom catalog's ``$id`` (the basic one when there is none)."""

    @property
    def id(self) -> str:
        return str(self.schema["$id"])

    @property
    def custom_components(self) -> JsonObject:
        return _as_object(self.custom.get("components")) if self.custom is not None else {}

    @property
    def by_id(self) -> dict[str, JsonSchema]:
        """Every catalog a surface may name, by ``$id``: the basic one, and the merged one when it differs."""
        return {str(self.basic["$id"]): self.basic, self.id: self.schema}


def _merge_catalogs(basic: JsonSchema, custom: JsonSchema | None) -> _MergedCatalog:
    """Merge ``custom`` over ``basic``, rebuilding ``anyComponent`` / ``anyFunction`` to cover the union."""
    if custom is None:
        return _MergedCatalog(basic=basic, custom=None, schema=basic)
    merged: JsonObject = {**basic, "$id": custom["$id"]}
    for section in _CATALOG_SECTIONS:
        merged[section] = {**_as_object(basic.get(section)), **_as_object(custom.get(section))}
    components = _as_object(merged["components"])
    functions = _as_object(merged["functions"])
    defs = _as_object(merged["$defs"])
    if components:
        defs["anyComponent"] = {
            "oneOf": [{"$ref": _catalog_definition_ref("", "components", name)} for name in components],
            "discriminator": {"propertyName": "component"},
        }
    if functions:
        defs["anyFunction"] = {
            "oneOf": [{"$ref": _catalog_definition_ref("", "functions", name)} for name in functions]
        }
    return _MergedCatalog(basic=basic, custom=custom, schema=merged)


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


def _resource(contents: JsonSchema) -> Resource:
    return Resource.from_contents(contents, default_specification=DRAFT202012)


class A2UISchemaManager:
    """Loads vendored A2UI schemas for a protocol version and builds the
    system-prompt section that instructs an LLM to emit valid A2UI output.

    A custom catalog (file path or dict) extends the basic catalog; its
    components are merged into the schema registry and listed in the prompt.
    In v1.0 the custom catalog's own ``$ref``s may target only what it declares
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
            custom_catalog: A custom catalog that extends the basic catalog. Can be:
                - A file path (str or PathLike) to a JSON catalog file
                - A dict with the catalog schema directly
                Must include a ``$id`` field (used as the catalogId in A2UI messages).
            custom_catalog_rules: Plain-text rules for the custom catalog components,
                appended to the basic catalog rules in the system prompt.
        """
        self._protocol = get_protocol(protocol_version)
        self._custom_catalog_input = custom_catalog
        self._custom_catalog_rules = custom_catalog_rules or ""
        self._loaded = False

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
        catalog = _merge_catalogs(basic, custom)
        if protocol.per_surface_catalogs:
            common_types = _with_catalog_function_dispatch(common_types, catalog.by_id, catalog.schema)
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
        """Every catalog a surface may name: the basic one, and the agent's merged one when it differs."""
        self._ensure_loaded()
        return tuple(self._catalog.by_id)

    @property
    def server_to_client_schema(self) -> JsonSchema:
        """The message schema responses are validated against."""
        self._ensure_loaded()
        return self._validation_schema

    def component_references(self) -> ComponentReferences:
        """Where each component type of the merged catalog references other components."""
        self._ensure_loaded()
        return component_references(self._catalog.schema)

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
            f"{base}catalog.json": _resource(self._catalog.schema),
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
        """Build the prompt example JSON string from the version-specific template."""
        raw = json.dumps(self._spec.prompt_example, separators=(",", ": "))
        return self._fill_template(raw)

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
        if _as_object(catalog.schema.get("components")):
            basic_components = sorted(_as_object(catalog.basic.get("components")))
            custom_comp_defs = catalog.custom_components
            custom_components = sorted(custom_comp_defs)

            comp_section = "\n\n## Available Components\n\n"
            if basic_components:
                comp_section += f"**Basic catalog:** {', '.join(basic_components)}\n\n"
            if custom_components:
                comp_section += f"**Custom components:** {', '.join(custom_components)}\n\n"
                # The properties every component shares: the ``component`` discriminator and ``ComponentCommon``
                # (``id`` in every version, then v0.9 ``accessibility`` or v1.0 ``@catalogId``, ...).
                common = _as_object(_as_object(self._common_types.get("$defs")).get("ComponentCommon"))
                envelope = frozenset({"component", *_as_object(common.get("properties"))})
                for comp_name in custom_components:
                    comp_def = _as_object(custom_comp_defs.get(comp_name))
                    desc = _component_description(comp_def, envelope)
                    if desc:
                        comp_section += f"- **{comp_name}**: {desc}\n"
                comp_section += "\n"

            comp_section += (
                "Each component uses a flat discriminator format:\n"
                '```json\n{"id": "myId", "component": "Text", "text": "Hello"}\n```\n\n'
                "Components are referenced by `id`. Use `children` (array of ids) for layout components (Column, Row). "
                "Use `child` (single id) for wrapper components (Card, Button)."
            )
            sections.append(comp_section)

        if include_rules:
            rules_parts: list[str] = []
            if self._spec.catalog_rules:
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
