# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Turn the JSON-schema failures of an A2UI message into issues an LLM can act on.

The message schema is a ``oneOf`` over one definition per message, and a catalog's
component and function lists are ``oneOf``\\s over one definition per variant. Left alone,
``jsonschema`` reports a failed ``oneOf`` as "not valid under any of the given schemas"
and echoes the whole operation, which tells the model nothing. This module narrows each
``oneOf`` to the variant the instance was aiming for, so the issues point at the
offending property.
"""

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import jsonschema
from jsonschema import FormatChecker
from referencing import Registry

from ._types import JsonObject, JsonSchema, JsonValue, server_to_client_payload_keys

# ``jsonschema`` ships no type information, so its error objects are opaque to mypy.
_SchemaError: TypeAlias = Any

_Path: TypeAlias = tuple[str | int, ...]


@dataclass(slots=True, frozen=True)
class A2UIIssue:
    """One reason an A2UI message does not match the schema."""

    path: _Path
    """Where in the message the problem is, as object keys and list indices."""
    message: str

    def render(self, message: JsonValue) -> str:
        """The issue as a sentence for the LLM, naming the component it sits in when there is one."""
        node = message
        component: JsonObject | None = None
        for part in self.path:
            if isinstance(node, dict) and isinstance(part, str):
                node = node.get(part)
            elif isinstance(node, list) and isinstance(part, int) and 0 <= part < len(node):
                node = node[part]
            else:
                break
            if isinstance(node, dict) and isinstance(node.get("component"), str):
                component = node
        if component is not None:
            return f"Component '{component.get('id', '?')}' ({component['component']}): {self.message}"
        if not self.path:
            return self.message
        return f"{self.message} (at /{'/'.join(map(str, self.path))})"


@dataclass(slots=True, frozen=True)
class _Leaf:
    path: _Path
    message: str
    keyword: str


XID_FORMAT = "xid-identifier"
COMPONENT_ID_FORMAT = "a2ui-component-id"
"""A component id, or a reference to one, that must be a UAX #31 identifier (v1.0)."""
UAX31_IDENTIFIER = "UAX #31 identifier (a letter or '_', then letters, digits or '_')"

# Formats no value has. A derived schema puts one where the instance is known to be wrong for a reason the JSON
# schema cannot word, and the diagnostics replace jsonschema's text with that reason.
NO_DEFAULT_CATALOG_FORMAT = "a2ui-no-default-catalog"
"""An implicit component or call on a surface created without a default catalog (v1.0)."""
AGENT_CANNOT_CALL_FORMAT = "a2ui-agent-cannot-call"
"""A ``callRendererFunction`` naming a ``rendererOnly`` function."""
RENDERER_CANNOT_CALL_FORMAT = "a2ui-renderer-cannot-call"
"""A binding or action naming an ``agentOnly`` function."""
RESERVED_SURFACE_FORMAT = "a2ui-reserved-surface"
"""A component of type ``Surface``, the container ``createSurface`` creates (v1.0)."""
_NO_DEFAULT_CATALOG_HINT = "and the surface has no default 'catalogId' (set it in createSurface)"


def _no_default_catalog(call: JsonValue) -> str:
    if isinstance(call, str):
        return f"function {call!r} names no 'catalogId' {_NO_DEFAULT_CATALOG_HINT}"
    return f"names no '@catalogId' {_NO_DEFAULT_CATALOG_HINT}"


def _agent_cannot_call(call: JsonValue) -> str:
    return f"function {call!r} cannot be called by agent (allowedCallers is rendererOnly)"


def _renderer_cannot_call(call: JsonValue) -> str:
    return f"function {call!r} cannot be called by renderer (allowedCallers is agentOnly)"


def _reserved_surface(_call: JsonValue) -> str:
    return (
        "component type 'Surface' is reserved for the container createSurface creates and cannot be sent as a component"
    )


_REJECTING_FORMATS: Mapping[str, Callable[[JsonValue], str]] = {
    NO_DEFAULT_CATALOG_FORMAT: _no_default_catalog,
    AGENT_CANNOT_CALL_FORMAT: _agent_cannot_call,
    RENDERER_CANNOT_CALL_FORMAT: _renderer_cannot_call,
    RESERVED_SURFACE_FORMAT: _reserved_surface,
}
"""Each rejecting format, with the reason it reports given the failing instance's ``@call`` (if any)."""

_UNKNOWN_TAG = "unknown-tag"
"""The keyword of a leaf reporting a tag value no variant accepts."""
_AIMED_KEYWORDS = frozenset((_UNKNOWN_TAG, *_REJECTING_FORMATS))
"""Leaf keywords that show the instance meant the variant reporting them (it names a function, say)."""
_TAG_NOUNS = {"call": "function", "@call": "function", "catalogId": "catalog", "@catalogId": "catalog"}


def _is_identifier(value: JsonValue) -> bool:
    # ``str.isidentifier`` is defined by XID_Start (plus ``_``) and XID_Continue, the sets UAX #31 names.
    return not isinstance(value, str) or value.isidentifier()


def _check_identifier(instance: object) -> bool:
    return not isinstance(instance, str) or _is_identifier(instance)


class MessageChecker:
    """Check A2UI messages against a message schema and report where they are wrong.

    Each message carries exactly one payload key (``createSurface``, ``updateComponents``,
    ...) that decides which definition it must satisfy, so a message is checked against
    that definition alone instead of the schema's top-level ``oneOf``.
    """

    def __init__(self, schema: JsonSchema, registry: Registry | None = None) -> None:
        validator_cls = jsonschema.validators.validator_for(schema)
        formats = FormatChecker(formats=())
        formats.checks(XID_FORMAT)(_check_identifier)
        formats.checks(COMPONENT_ID_FORMAT)(_check_identifier)
        for rejecting in _REJECTING_FORMATS:
            formats.checks(rejecting)(_never)
        self._validators: dict[str, Any] = {}
        defs = schema.get("$defs")
        branches = schema.get("oneOf")
        base = {key: value for key, value in schema.items() if key != "oneOf"}
        payload_keys = server_to_client_payload_keys()
        for branch in branches if isinstance(branches, list) else ():
            ref = branch.get("$ref") if isinstance(branch, dict) else None
            definition = defs.get(ref.rsplit("/", 1)[-1]) if isinstance(ref, str) and isinstance(defs, dict) else None
            required = definition.get("required") if isinstance(definition, dict) else None
            for key in payload_keys.intersection(required if isinstance(required, list) else ()):
                branch_schema = {**base, "$ref": ref}
                self._validators[key] = (
                    validator_cls(branch_schema, registry=registry, format_checker=formats)
                    if registry
                    else validator_cls(branch_schema, format_checker=formats)
                )

    def issues(self, message: JsonObject) -> list[A2UIIssue]:
        """Every way ``message`` violates the schema; empty when it is valid.

        Raises:
            referencing.exceptions.Unresolvable: A ``$ref`` in the schema cannot be resolved.
        """
        validator = next((v for key, v in self._validators.items() if key in message), None)
        if validator is None:
            keys = sorted(self._validators)
            return [A2UIIssue((), f"Not an A2UI message: expected an object with exactly one of {keys}")]
        found = [leaf for error in validator.iter_errors(message) for leaf in _narrow(error)]
        # A subschema that fails contributes no annotations, so alongside any other failure at or below
        # an object, "unevaluated property" lists properties that are in fact fine (e.g. a Button's ``child``).
        reliable = [leaf for leaf in found if leaf.keyword != "unevaluatedProperties"]
        issues = [
            leaf
            for leaf in found
            if leaf.keyword != "unevaluatedProperties"
            or not any(other.path[: len(leaf.path)] == leaf.path for other in reliable)
        ]
        return [A2UIIssue(leaf.path, leaf.message) for leaf in issues]


def _never(instance: object) -> bool:
    return False


def _rejecting_format(error: _SchemaError) -> str | None:
    """The rejecting format that failed in ``error``, if that is what it reports."""
    if error.validator == "format" and error.validator_value in _REJECTING_FORMATS:
        return str(error.validator_value)
    return None


def _leaf_message(error: _SchemaError, rejecting: str | None) -> str:
    instance = error.instance
    if rejecting is not None:
        return _REJECTING_FORMATS[rejecting](instance.get("@call") if isinstance(instance, dict) else None)
    if error.validator is None and isinstance(instance, dict):
        # A ``false`` schema: a catalog's ``anyComponent`` / ``anyFunction`` when it defines none.
        for tag, noun in (("component", "component"), ("@call", "function")):
            if isinstance(instance.get(tag), str):
                return f"unknown {noun} {instance[tag]!r}"
    if error.validator == "format" and error.validator_value == COMPONENT_ID_FORMAT:
        return f"{instance!r} is not a valid component id: it must be a {UAX31_IDENTIFIER}"
    if error.validator == "format" and error.validator_value == XID_FORMAT:
        return f"{instance!r} is not a {UAX31_IDENTIFIER}"
    return str(error.message)


def _narrow(error: _SchemaError) -> list[_Leaf]:
    """The leaf failures of ``error``, descending a failed ``oneOf`` / ``anyOf`` into its closest variant."""
    if not error.context:
        rejecting = _rejecting_format(error)
        return [_Leaf(tuple(error.absolute_path), _leaf_message(error, rejecting), rejecting or error.validator)]

    variants: Mapping[int, list[_SchemaError]] = _group_by_variant(error.context)
    # A variant that fails on a ``const`` tag (``component: "Slider"``, ``@call: "openUrl"``) is
    # simply not the one the instance meant; what is left are the variants worth reporting on.
    tags = {index: tag for index, errors in variants.items() if (tag := _tag_mismatch(errors)) is not None}
    unselected = {index for index, errors in variants.items() if all(_unselected(e) for e in errors)}
    candidates = {index: errors for index, errors in variants.items() if index not in tags.keys() | unselected}
    if not candidates and not tags:
        candidates = dict(variants)
    if not candidates:
        tag = next(iter(tags.values()))
        value = error.instance.get(tag) if isinstance(error.instance, dict) else None
        return [_Leaf(tuple(error.absolute_path), f"unknown {_TAG_NOUNS.get(tag, tag)} {value!r}", _UNKNOWN_TAG)]
    narrowed = [[issue for e in errors for issue in _narrow(e)] for errors in candidates.values()]
    # An object carrying a tag the catalog does not know (``{"call": "nosuch"}``), or a call with no catalog to
    # resolve against, meant that variant, not a plain ``string`` one beside it that fails only on the JSON type.
    here = tuple(error.absolute_path)
    if any(leaves and all(leaf.keyword in _AIMED_KEYWORDS for leaf in leaves) for leaves in narrowed):
        narrowed = [
            leaves
            for leaves in narrowed
            if not (len(leaves) == 1 and leaves[0].keyword == "type" and leaves[0].path == here)
        ]
    return min(narrowed, key=len)


def _group_by_variant(context: list[_SchemaError]) -> dict[int, list[_SchemaError]]:
    variants: dict[int, list[_SchemaError]] = defaultdict(list)
    for error in context:
        variants[error.schema_path[0]].append(error)
    return variants


def _unselected(error: _SchemaError) -> bool:
    """Whether the error only says the variant does not apply, so the variant was never tried.

    That is a selector (``if``) that did not match, failing through its ``else``, or a ``not: {required: ...}``
    guard such as "no ``@catalogId``" on the implicit-catalog component.
    """
    if list(error.schema_path)[1:2] == ["else"]:
        return True
    if error.validator == "not":
        return isinstance(error.validator_value, dict) and set(error.validator_value) == {"required"}
    return error.validator in ("anyOf", "oneOf") and bool(error.context) and all(_unselected(e) for e in error.context)


def _tag_mismatch(errors: list[_SchemaError]) -> str | None:
    """The property whose ``const`` value rules this variant out, if that is why it failed."""
    for error in errors:
        if error.validator == "const" and len(error.relative_path) == 1:
            return str(error.relative_path[0])
    return None


__all__ = (
    "AGENT_CANNOT_CALL_FORMAT",
    "COMPONENT_ID_FORMAT",
    "NO_DEFAULT_CATALOG_FORMAT",
    "RENDERER_CANNOT_CALL_FORMAT",
    "RESERVED_SURFACE_FORMAT",
    "UAX31_IDENTIFIER",
    "XID_FORMAT",
    "A2UIIssue",
    "MessageChecker",
)
