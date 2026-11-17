# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Checks on a response's component tree that the JSON schema cannot express. Not exported."""

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from ._types import JsonObject, JsonValue

_JSON_POINTER = re.compile(r"(/([^/~]|~[01])*)*")
_ANY_ITEM = "*"
_Path: TypeAlias = tuple[str, ...]
_Kind: TypeAlias = Literal["id", "list"]

# Common-types definitions whose values name other components on the same surface.
_REFERENCE_DEFINITIONS: dict[str, _Kind] = {"ComponentId": "id", "Child": "id", "ChildList": "list"}


@dataclass(frozen=True, slots=True)
class _Reference:
    path: _Path
    """Property names from the component to the reference; ``*`` steps into every list item."""
    kind: _Kind


ComponentReferences: TypeAlias = Mapping[str, tuple[_Reference, ...]]
"""Where each component type holds references to other components, by component type."""


def component_references(
    *catalogs: JsonObject, documents: Mapping[str, JsonObject] | None = None
) -> ComponentReferences:
    """Read, from catalogs, where each of their component types references other components.

    As the spec says, only a property whose schema ``$ref``s ``ComponentId``, ``Child`` or ``ChildList`` is a
    reference; a plain string is static text. Like the official SDK, an inline object schema with ``componentId``
    and ``path`` (or ``dataBinding``) also counts, as a template child list. A later catalog's component wins over
    an earlier one's of the same type. ``documents`` resolves a ``$ref`` into another file (``basic_catalog.json``).
    """
    references: dict[str, tuple[_Reference, ...]] = {}
    for catalog in catalogs:
        components = catalog.get("components")
        for key, definition in components.items() if isinstance(components, dict) else ():
            found = dict.fromkeys(_references_in(definition, (), catalog, frozenset(), documents or {}))
            references[_component_type(definition) or key] = tuple(found)
    return references


def _component_type(definition: JsonValue) -> str | None:
    """The ``component`` tag a definition accepts (its ``const``), which may differ from its key."""
    if not isinstance(definition, dict):
        return None
    properties = definition.get("properties")
    tag = properties.get("component") if isinstance(properties, dict) else None
    if isinstance(tag, dict) and isinstance(tag.get("const"), str):
        return str(tag["const"])
    branches = definition.get("allOf")
    for branch in branches if isinstance(branches, list) else ():
        if (found := _component_type(branch)) is not None:
            return found
    return None


def _references_in(
    schema: JsonValue, path: _Path, catalog: JsonObject, seen: frozenset[str], documents: Mapping[str, JsonObject]
) -> Iterator[_Reference]:
    if not isinstance(schema, dict):
        return
    ref = schema.get("$ref")
    if isinstance(ref, str):
        document, _, pointer = ref.partition("#")
        if (kind := _REFERENCE_DEFINITIONS.get(ref.rsplit("/", 1)[-1])) is not None:
            yield _Reference(path, kind)
        elif ref.startswith("#/") and ref not in seen:
            # A catalog-local definition (``#/$defs/Kid``) may itself name ``ComponentId``.
            yield from _references_in(_resolve_local(catalog, ref), path, catalog, seen | {ref}, documents)
        elif document in documents and pointer.startswith("/") and ref not in seen:
            # A definition in another catalog file (before v1.0, ``basic_catalog.json#/components/Column``).
            target = documents[document]
            yield from _references_in(_resolve_local(target, f"#{pointer}"), path, target, seen | {ref}, documents)
    for keyword in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(keyword)
        for branch in branches if isinstance(branches, list) else ():
            yield from _references_in(branch, path, catalog, seen, documents)
    properties = schema.get("properties")
    if isinstance(properties, dict) and "componentId" in properties and properties.keys() & {"path", "dataBinding"}:
        yield _Reference(path, "list")
        return
    for name, subschema in properties.items() if isinstance(properties, dict) else ():
        yield from _references_in(subschema, (*path, name), catalog, seen, documents)
    yield from _references_in(schema.get("items"), (*path, _ANY_ITEM), catalog, seen, documents)


def _resolve_local(catalog: JsonObject, ref: str) -> JsonValue:
    node: JsonValue = catalog
    for token in ref[2:].split("/"):
        key = token.replace("~1", "/").replace("~0", "~")
        node = node.get(key) if isinstance(node, dict) else None
    return node


def component_graph_errors(messages: Sequence[JsonObject], references: ComponentReferences) -> list[str]:
    """Check each surface's components in ``messages`` as a tree.

    For a surface this response creates: every reference resolves, every component is reachable from
    ``root``, there are no cycles and no id appears twice in one message. A surface created earlier may be
    updated partially, so only duplicate ids and cycles among this response's own components are checked.
    """
    created: set[str] = set()
    surfaces: dict[str, list[list[JsonObject]]] = {}
    for message in messages:
        for key in ("createSurface", "updateComponents"):
            payload = message.get(key)
            if not isinstance(payload, dict) or not isinstance(surface_id := payload.get("surfaceId"), str):
                continue
            if key == "createSurface":
                created.add(surface_id)
            listed = payload.get("components")
            surfaces.setdefault(surface_id, []).append([
                c for c in (listed if isinstance(listed, list) else ()) if isinstance(c, dict)
            ])
    errors: list[str] = []
    for surface_id, components in surfaces.items():
        errors.extend(_surface_errors(surface_id, components, references, complete=surface_id in created))
    return errors


def _surface_errors(
    surface_id: str, batches: list[list[JsonObject]], references: ComponentReferences, *, complete: bool
) -> Iterator[str]:
    """``batches`` holds each message's components; a later message may update an earlier one's by id."""
    prefix = f"Surface '{surface_id}':"
    edges: dict[str, list[str]] = {}
    for components in batches:
        in_message: set[str] = set()
        for component in components:
            component_id = component.get("id")
            if not isinstance(component_id, str):
                continue
            if component_id in in_message:
                yield f"{prefix} duplicate component id '{component_id}'"
                continue
            in_message.add(component_id)
            component_type = component.get("component")
            refs = references.get(component_type, ()) if isinstance(component_type, str) else ()
            edges[component_id] = list(_referenced_ids(component, refs))

    if complete:
        for component_id, targets in edges.items():
            for target in targets:
                if target not in edges:
                    yield f"{prefix} component '{component_id}' references missing component '{target}'"

    for cycle in _cycles(edges):
        yield f"{prefix} components form a cycle: {' -> '.join(cycle)}"

    if complete and "root" in edges:
        reachable = _reachable(edges, "root")
        for component_id in edges:
            if component_id not in reachable:
                yield f"{prefix} component '{component_id}' is not reachable from 'root'"


def _referenced_ids(component: JsonObject, references: tuple[_Reference, ...]) -> Iterator[str]:
    for reference in references:
        for value in _values_at(component, reference.path):
            if reference.kind == "id":
                if isinstance(value, str):
                    yield value
            elif isinstance(value, list):
                yield from (item for item in value if isinstance(item, str))
            elif isinstance(value, dict) and isinstance(template := value.get("componentId"), str):
                yield template


def _values_at(node: JsonValue, path: _Path) -> Iterator[JsonValue]:
    if not path:
        yield node
        return
    step, rest = path[0], path[1:]
    if step == _ANY_ITEM:
        for item in node if isinstance(node, list) else ():
            yield from _values_at(item, rest)
    elif isinstance(node, dict) and step in node:
        yield from _values_at(node[step], rest)


def _reachable(edges: Mapping[str, list[str]], start: str) -> set[str]:
    seen = {start}
    pending = [start]
    while pending:
        for target in edges.get(pending.pop(), ()):
            if target in edges and target not in seen:
                seen.add(target)
                pending.append(target)
    return seen


def _cycles(edges: Mapping[str, list[str]]) -> Iterator[list[str]]:
    """The cycles a depth-first search over the references meets, each as ids from where it closes back to it."""
    done: set[str] = set()
    for start in edges:
        if start in done:
            continue
        # Iterative depth-first search; ``trail`` is the current path from ``start``.
        trail: list[str] = [start]
        on_trail = {start}
        iterators = [iter(edges[start])]
        while iterators:
            target = next(iterators[-1], None)
            if target is None:
                iterators.pop()
                finished = trail.pop()
                on_trail.discard(finished)
                done.add(finished)
                continue
            if target in on_trail:
                yield [*trail[trail.index(target) :], target]
            elif target in edges and target not in done:
                trail.append(target)
                on_trail.add(target)
                iterators.append(iter(edges[target]))


_SURFACE_UPDATES = ("updateComponents", "updateDataModel")


def surface_lifecycle_errors(messages: Sequence[JsonObject]) -> list[str]:
    """Check, in message order, that each surface is created before it is updated and not created twice.

    A surface must be created before ``updateComponents`` / ``updateDataModel`` reach it, and creating an existing
    surface without deleting it first is an integrity error (a2ui_protocol.md, ``createSurface``). A surface this
    response never creates may come from an earlier turn, so updates to it are accepted until it is deleted.
    """
    errors: list[str] = []
    # Per surface: the last ``createSurface`` / ``deleteSurface``, or the first update while neither has happened.
    last: dict[str, tuple[str, int]] = {}
    for index, message in enumerate(messages):
        for key, payload in message.items():
            if not isinstance(payload, dict) or not isinstance(surface_id := payload.get("surfaceId"), str):
                continue
            event, at = last.get(surface_id, ("", -1))
            if key == "createSurface":
                if event == "createSurface":
                    errors.append(
                        f"Operation {index}: surface '{surface_id}' already exists (operation {at} created it); "
                        "send deleteSurface before creating it again"
                    )
                elif event in _SURFACE_UPDATES:
                    errors.append(
                        f"Operation {at}: '{event}' targets surface '{surface_id}' before operation {index} "
                        "creates it; send createSurface first"
                    )
                last[surface_id] = (key, index)
            elif key == "deleteSurface":
                last[surface_id] = (key, index)
            elif key in _SURFACE_UPDATES:
                if event == "deleteSurface":
                    errors.append(
                        f"Operation {index}: '{key}' targets surface '{surface_id}' after operation {at} deleted it"
                    )
                elif not event:
                    last[surface_id] = (key, index)
    return errors


def is_json_pointer(path: str) -> bool:
    """Whether ``path`` is an RFC 6901 JSON Pointer: empty, or ``/``-prefixed tokens with ``~`` only as ``~0`` / ``~1``."""
    return _JSON_POINTER.fullmatch(path) is not None


__all__ = (
    "ComponentReferences",
    "component_graph_errors",
    "component_references",
    "is_json_pointer",
    "surface_lifecycle_errors",
)
