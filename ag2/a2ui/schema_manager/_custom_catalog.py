# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Load a custom catalog and reject one the served protocol version does not allow."""

import os
import re
from collections.abc import Iterator
from functools import cache
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import best_match
from referencing import Registry

from .._types import JsonSchema, JsonValue
from .._versions import A2UIProtocol
from ..diagnostics import UAX31_IDENTIFIER, XID_FORMAT
from ._json import _PACKAGE_DIR, _as_list, _as_object, _load_object, _pointer_key, _pointer_token, _resource

# v1.0 constrains extension keys with Unicode property escapes, which Python's ``re``
# (what ``jsonschema`` matches patterns with) rejects. Check those keys with a format
# instead, at load time, leaving the vendored files byte-faithful.
_XID_KEYS = r"^[\p{XID_Start}_][\p{XID_Continue}]*$"


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


def _check_component_discriminators(catalog: JsonSchema) -> None:
    """Reject a v1.0 catalog component without a required ``component`` const equal to its key.

    v1_0 a2ui_protocol.md, "Catalog Schema Rules", 4: every component "must have a required property named
    `component` whose value is a constant (`const`) matching the key under which it is defined". Rule 5 lets the
    discriminator sit beside imported traits in an ``allOf`` (the basic catalog's ``CheckBox`` does), so the
    definition and its ``allOf`` branches are searched. The prompt names components by key and the validator by
    const, so a component that breaks the rule could be named but never sent.
    """
    for name, definition in _as_object(catalog.get("components")).items():
        schemas = list(_definition_schemas(_as_object(definition)))
        tags = [
            tag for schema in schemas if isinstance(tag := _as_object(schema.get("properties")).get("component"), dict)
        ]
        if not tags or any("const" not in tag for tag in tags):
            raise ValueError(f'Invalid custom catalog: component {name!r} must declare "component" as a const')
        for tag in tags:
            if tag["const"] != name:
                raise ValueError(
                    f'Invalid custom catalog: component {name!r} declares "component": {tag["const"]!r}; '
                    f"it must be {name!r}, the key it is defined under"
                )


def _with_required_discriminators(catalog: JsonSchema) -> JsonSchema:
    """``catalog`` with ``component`` in every component's ``required``, as the official SDK serves it.

    Rule 4 makes the discriminator "a required property", yet upstream's own catalog cases
    (v1_0/test/cases/composition_constraints.json) leave it out of ``required`` and are valid; the SDK adds it
    (``message_processor.py``). Without it a catalog of one component would accept a component the renderer
    cannot route. The caller's catalog is not changed.
    """
    components = _as_object(catalog.get("components"))
    if not components:
        return catalog
    return {
        **catalog,
        "components": {
            name: (
                {**definition, "required": [*required, "component"]}
                if isinstance(definition, dict)
                and "component" not in (required := _as_list(definition.get("required")))
                else definition
            )
            for name, definition in components.items()
        },
    }


def _definition_schemas(definition: JsonSchema) -> Iterator[JsonSchema]:
    """``definition`` and the object schemas its ``allOf`` composes, depth first."""
    yield definition
    for branch in _as_list(definition.get("allOf")):
        if isinstance(branch, dict):
            yield from _definition_schemas(branch)


def _check_entity_name(kind: str, name: str, *, owner: str | None = None) -> None:
    if not _is_identifier_key(name):
        of_owner = f" of {owner}" if owner else ""
        raise ValueError(f"Invalid custom catalog: {kind} {name!r}{of_owner} is not a {UAX31_IDENTIFIER}")


def _load_custom_catalog(custom_catalog: "str | os.PathLike[str] | JsonSchema", protocol: A2UIProtocol) -> JsonSchema:
    """The custom catalog, checked against the served version's catalog rules, with ``$id`` set to its ``catalogId``.

    Internally a catalog is keyed by ``$id``, the name its ``$ref``s resolve against; the spec asks for both fields
    "set to the same URI", so a catalog that omits ``$id`` is given its ``catalogId``.
    """
    if isinstance(custom_catalog, dict):
        catalog = custom_catalog
    else:
        path = Path(custom_catalog)
        if not path.exists():
            raise FileNotFoundError(f"Custom catalog file not found: {path}")
        catalog = _load_object(path)
    catalog_id = _checked_catalog_id(catalog)
    if catalog_id == protocol.catalog_id:
        # Two catalogs under one id cannot be told apart in createSurface or negotiation; a2ui_agent rejects
        # duplicate ids (conformance/agent/multi_catalog_formats.yaml, test_direct_json_create_format_rejects_
        # duplicate_catalog_ids), and a renderer would draw this id with its own basic catalog.
        raise ValueError(
            f"Custom catalog 'catalogId' {catalog_id!r} is the basic catalog's id; give the custom catalog its own URI."
        )
    _check_catalog_protocol_version(catalog, protocol)
    if protocol.entity_naming_rules:
        # Before the full schema, for clearer messages on the rules a catalog author most often breaks.
        _check_entity_names(catalog)
    if protocol.catalog_definition_file is not None:
        _check_catalog_definition(catalog, protocol)
        _check_component_discriminators(catalog)
        catalog = _with_required_discriminators(catalog)
    if protocol.restricted_catalog_refs:
        _check_catalog_references(catalog)
    return {**catalog, "$id": catalog_id}


def _checked_catalog_id(catalog: JsonSchema) -> str:
    """The catalog's ``catalogId``, which identifies it; ``$id`` must match.

    See v0_9 a2ui_protocol.md, "Catalog Identification & Compatibility", and v1_0, "Catalog structure".

    The spec says both "should" be set "to the same URI"; ag2 requires it, since ``$ref``s resolve against ``$id``
    and the catalog is served under its ``catalogId`` (ADR 0040). The official SDK also takes the id from
    ``catalogId`` only (its ``_settle_catalog_id``).
    """
    catalog_id = catalog.get("catalogId")
    if not isinstance(catalog_id, str) or not catalog_id:
        raise ValueError(
            "Custom catalog must include a non-empty string 'catalogId': it identifies the catalog in "
            "createSurface, supportedCatalogIds and catalog negotiation. "
            'Example: {"catalogId": "https://mycompany.com/my_catalog.json", ...}'
        )
    if "$id" in catalog and catalog["$id"] != catalog_id:
        raise ValueError(
            f"Custom catalog '$id' {catalog['$id']!r} differs from its 'catalogId' {catalog_id!r}; set both to the "
            "same URI (or leave '$id' out)."
        )
    return catalog_id


_SEMVER_IDENTIFIER = r"(?:0|[1-9][0-9]*|[0-9]*[a-zA-Z-][0-9a-zA-Z-]*)"
_CATALOG_PROTOCOL_VERSION = re.compile(
    r"v?(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*))?"
    rf"(?:-{_SEMVER_IDENTIFIER}(?:\.{_SEMVER_IDENTIFIER})*)?"
    r"(?:\+[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*)?",
    re.ASCII,
)
"""``protocolVersion``'s ``pattern`` in v1.0 ``catalog_definition.json``, plus the ``v`` prefix the official SDK
accepts (``'v0.9'``); v0.9 has no catalog definition schema, so this is the only check there."""


def _check_catalog_protocol_version(catalog: JsonSchema, protocol: A2UIProtocol) -> None:
    """Reject a catalog written against another protocol version than the one served.

    ``protocolVersion`` "defaults to `"0.9"` if omitted, required for catalogs targeting `1.0` and beyond" (v1_0
    a2ui_protocol.md, "Catalog Schema Rules and Conventions"). Its major.minor must be the served version's. The
    official SDK rejects a mismatch too and is stricter: it matches the patch number against its known versions as
    well, so its v0.9.1 provider rejects a ``"0.9"`` catalog. ag2 serves the same ``"0.9"`` catalogs on v0.9 and
    v0.9.1.
    """
    declared = catalog.get("protocolVersion", "0.9")
    match = _CATALOG_PROTOCOL_VERSION.fullmatch(declared) if isinstance(declared, str) else None
    if match is None:
        raise ValueError(f"Custom catalog 'protocolVersion' {declared!r} is not a version such as '1.0'")
    major_minor = f"{match[1]}.{match[2]}"
    if major_minor != protocol.catalog_protocol_version:
        stated = f"declares protocolVersion {declared!r}" if "protocolVersion" in catalog else "has no protocolVersion"
        raise ValueError(
            f"Custom catalog {stated}, so it is written for A2UI {major_minor}, but this agent serves "
            f'{protocol.version}: set "protocolVersion": "{protocol.catalog_protocol_version}"'
        )


_MAX_SCHEMA_MESSAGE = 300
"""Longer messages are cut: a failed ``oneOf`` prints the whole instance."""


def _check_catalog_definition(catalog: JsonSchema, protocol: A2UIProtocol) -> None:
    """Reject a catalog that does not satisfy the version's vendored ``catalog_definition.json``."""
    errors = list(_catalog_definition_validator(protocol).iter_errors(catalog))
    # A failing subschema yields no annotations, so alongside another error "unevaluated properties" lists
    # properties that are in fact allowed (``allowedCallers`` beside a bad ``allowedCallers`` value).
    error = best_match([e for e in errors if e.validator != "unevaluatedProperties"] or errors)
    if error is None:
        return
    pointer = "".join(f"/{_pointer_token(str(part))}" for part in error.absolute_path) or "/"
    message = error.message
    if len(message) > _MAX_SCHEMA_MESSAGE:
        message = f"{message[:_MAX_SCHEMA_MESSAGE]}..."
    raise ValueError(f"Invalid custom catalog at {pointer} ({protocol.catalog_definition_file}): {message}")


def _is_identifier_key(instance: object) -> bool:
    """Whether ``instance`` is a UAX #31 identifier; a non-string passes, as a JSON Schema ``format`` does.

    ``str.isidentifier`` is defined by XID_Start (plus ``_``) and XID_Continue, the sets UAX #31 names.
    """
    return not isinstance(instance, str) or instance.isidentifier()


@cache
def _catalog_definition_validator(protocol: A2UIProtocol) -> Draft202012Validator:
    """A validator for the version's ``catalog_definition.json``, with ``common_types.json`` in its registry.

    Both files key extensions by a ``\\p{XID_Start}`` pattern Python's ``re`` cannot compile, so those objects are
    checked with the :data:`XID_FORMAT` format instead (see :func:`_xid_property_names`).
    """
    spec_dir = _PACKAGE_DIR / protocol.package_dir / "spec"
    definition = _as_object(_xid_property_names(_load_object(spec_dir / str(protocol.catalog_definition_file))))
    common_types = _as_object(_xid_property_names(_load_object(spec_dir / "common_types.json")))
    registry: Registry = Registry().with_resources(
        (str(schema["$id"]), _resource(schema)) for schema in (definition, common_types)
    )
    formats = FormatChecker(formats=())
    formats.checks(XID_FORMAT)(_is_identifier_key)
    return Draft202012Validator(definition, registry=registry, format_checker=formats)
