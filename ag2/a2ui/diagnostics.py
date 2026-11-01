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
from collections.abc import Mapping
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


def _is_identifier(value: JsonValue) -> bool:
    # ``str.isidentifier`` is defined by XID_Start (plus ``_``) and XID_Continue, the sets UAX #31 names.
    return not isinstance(value, str) or value.isidentifier()


class MessageChecker:
    """Check A2UI messages against a message schema and report where they are wrong.

    Each message carries exactly one payload key (``createSurface``, ``updateComponents``,
    ...) that decides which definition it must satisfy, so a message is checked against
    that definition alone instead of the schema's top-level ``oneOf``.
    """

    def __init__(self, schema: JsonSchema, registry: Registry | None = None) -> None:
        validator_cls = jsonschema.validators.validator_for(schema)
        formats = FormatChecker(formats=())
        formats.checks(XID_FORMAT)(_is_identifier)
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

    def issues(self, message: JsonValue) -> list[A2UIIssue]:
        """Every way ``message`` violates the schema; empty when it is valid.

        Raises:
            referencing.exceptions.Unresolvable: A ``$ref`` in the schema cannot be resolved.
        """
        if not isinstance(message, dict):
            return [A2UIIssue((), f"Expected operation to be an object, got {type(message).__name__}")]
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


def _narrow(error: _SchemaError) -> list[_Leaf]:
    """The leaf failures of ``error``, descending a failed ``oneOf`` / ``anyOf`` into its closest variant."""
    if not error.context:
        return [_Leaf(tuple(error.absolute_path), error.message, error.validator)]

    variants: Mapping[int, list[_SchemaError]] = _group_by_variant(error.context)
    # A variant that fails on a ``const`` tag (``component: "Slider"``, ``@call: "openUrl"``) is
    # simply not the one the instance meant; what is left are the variants worth reporting on.
    tags = {index: tag for index, errors in variants.items() if (tag := _tag_mismatch(errors)) is not None}
    # A variant whose selector (``if``) did not match fails only through its ``else``: it was never tried.
    unselected = {index for index, errors in variants.items() if all(_in_else(e) for e in errors)}
    candidates = {index: errors for index, errors in variants.items() if index not in tags.keys() | unselected}
    if not candidates and not tags:
        candidates = dict(variants)
    if not candidates:
        tag = next(iter(tags.values()))
        value = error.instance.get(tag) if isinstance(error.instance, dict) else None
        return [_Leaf(tuple(error.absolute_path), f"unknown {tag} {value!r}", "oneOf")]
    return min(([issue for e in errors for issue in _narrow(e)] for errors in candidates.values()), key=len)


def _group_by_variant(context: list[_SchemaError]) -> dict[int, list[_SchemaError]]:
    variants: dict[int, list[_SchemaError]] = defaultdict(list)
    for error in context:
        variants[error.schema_path[0]].append(error)
    return variants


def _in_else(error: _SchemaError) -> bool:
    """Whether the error comes from an ``if``/``then``/``else`` schema's ``else`` branch."""
    return list(error.schema_path)[1:2] == ["else"]


def _tag_mismatch(errors: list[_SchemaError]) -> str | None:
    """The property whose ``const`` value rules this variant out, if that is why it failed."""
    for error in errors:
        if error.validator == "const" and len(error.relative_path) == 1:
            return str(error.relative_path[0])
    return None


__all__ = ("A2UIIssue", "MessageChecker")
