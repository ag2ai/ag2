# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Checks on a response's component tree that the JSON schema cannot express. Not exported."""

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
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


@dataclass(frozen=True, slots=True)
class ComponentType:
    """What the component-tree checks need to know about one component type of a catalog."""

    references: tuple[_Reference, ...]
    """Where the type holds references to other components."""
    allowed_parents: frozenset[str] | None = None
    """The component types that may contain it (``Surface`` for a surface's ``root``); ``None`` allows any."""
    allowed_children: frozenset[str] | None = None
    """The component types it may contain; ``None`` allows any."""


ComponentTypes: TypeAlias = Mapping[tuple[str, str], ComponentType]
"""Each component type the catalogs define, by catalog id and component type: two catalogs may define one type."""


@dataclass(frozen=True, slots=True)
class _Typed:
    """A component's ``component`` type name and, when its catalog resolves it, that type's definition."""

    type_name: str
    definition: ComponentType | None


_SURFACE = "Surface"
"""The reserved container ``createSurface`` creates; a surface's ``root`` is always its child (v1.0)."""


def component_types(*catalogs: JsonObject, documents: Mapping[str, JsonObject] | None = None) -> ComponentTypes:
    """Read, from catalogs, where each of their component types references other components, and how it nests.

    As the spec says, only a property whose schema ``$ref``s ``ComponentId``, ``Child`` or ``ChildList`` is a
    reference; a plain string is static text. Like the official SDK, an inline object schema with ``componentId``
    and ``path`` (or ``dataBinding``) also counts, as a template child list. Each catalog's types are keyed by its
    ``$id``. ``documents`` resolves a ``$ref`` into another file (``basic_catalog.json``).
    ``allowedParents`` / ``allowedChildren`` are the v1.0 composition constraints (a2ui_protocol.md, "Composition
    validation rules").
    """
    types: dict[tuple[str, str], ComponentType] = {}
    for catalog in catalogs:
        # A served catalog always has one: the schema manager sets a custom catalog's ``$id`` to its ``catalogId``.
        if not isinstance(catalog_id := catalog.get("$id"), str):
            raise ValueError("A catalog needs a string '$id' to key its component types by")
        components = catalog.get("components")
        for key, definition in components.items() if isinstance(components, dict) else ():
            found = dict.fromkeys(_references_in(definition, (), catalog, frozenset(), documents or {}))
            types[catalog_id, _component_type(definition) or key] = ComponentType(
                references=tuple(found),
                allowed_parents=_names(definition, "allowedParents"),
                allowed_children=_names(definition, "allowedChildren"),
            )
    return types


def _names(definition: JsonValue, key: str) -> frozenset[str] | None:
    names = definition.get(key) if isinstance(definition, dict) else None
    if not isinstance(names, list):
        return None
    return frozenset(name for name in names if isinstance(name, str))


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


@dataclass(slots=True)
class Surface:
    """One surface's components in a response, from its creation (or first update) until it is deleted."""

    surface_id: str
    created_at: int | None = None
    """The operation that creates it in this response; ``None`` for a surface from an earlier turn, which may be
    updated partially."""
    catalog_id: str | None = None
    """The ``catalogId`` its ``createSurface`` names, if it is created here and names one."""
    batches: list[list[JsonObject]] = field(default_factory=list)
    """Each message's components; a later message may update an earlier one's by id."""
    first_update: tuple[str, int] | None = None
    """For a surface from an earlier turn: its first update and that update's operation."""

    @property
    def created(self) -> bool:
        return self.created_at is not None


@dataclass(frozen=True, slots=True)
class SurfaceWalk:
    """The surfaces a response's messages build, in message order, and the lifecycle errors on the way."""

    surfaces: list[Surface]
    """Each surface the renderer would build; a message it rejects adds nothing to any of them."""
    errors: list[str]


_SURFACE_UPDATES = ("updateComponents", "updateDataModel")


def walk_surfaces(messages: Sequence[JsonObject], *, lifetime_unique_surface_ids: bool) -> SurfaceWalk:
    """Follow each surface through ``messages``: created, updated, deleted, created again.

    A surface must be created before ``updateComponents`` / ``updateDataModel`` reach it, and creating an existing
    surface without deleting it first is an integrity error (a2ui_protocol.md, ``createSurface``); a message
    breaking either rule is reported and joins no surface, since the renderer rejects it. A surface this response
    never creates may come from an earlier turn, so updates to it are accepted until it is deleted. A
    ``deleteSurface`` ends a surface: one created again under its id starts empty ("to reconfigure them, the
    surface must be deleted and recreated"). With ``lifetime_unique_surface_ids`` (v0.9, v1.0) the hint for a
    duplicate also offers a new ``surfaceId``.
    """
    remedy = "send deleteSurface before creating it again" + (
        ", or use a new surfaceId" if lifetime_unique_surface_ids else ""
    )
    surfaces: list[Surface] = []
    errors: list[str] = []
    current: dict[str, Surface] = {}
    deleted_at: dict[str, int] = {}
    for index, message in enumerate(messages):
        for key, payload in message.items():
            if not isinstance(payload, dict) or not isinstance(surface_id := payload.get("surfaceId"), str):
                continue
            surface = current.get(surface_id)
            if key == "createSurface":
                if surface is not None and surface.created_at is not None:
                    errors.append(
                        f"Operation {index}: surface '{surface_id}' already exists "
                        f"(operation {surface.created_at} created it); {remedy}"
                    )
                    continue
                if surface is not None and surface.first_update is not None:
                    event, at = surface.first_update
                    errors.append(
                        f"Operation {at}: '{event}' targets surface '{surface_id}' before operation {index} "
                        "creates it; send createSurface first"
                    )
                    surfaces.remove(surface)
                catalog_id = payload.get("catalogId")
                surface = current[surface_id] = Surface(
                    surface_id, created_at=index, catalog_id=catalog_id if isinstance(catalog_id, str) else None
                )
                surfaces.append(surface)
            elif key == "deleteSurface":
                current.pop(surface_id, None)
                deleted_at[surface_id] = index
                continue
            elif key in _SURFACE_UPDATES:
                if surface is None and surface_id in deleted_at:
                    errors.append(
                        f"Operation {index}: '{key}' targets surface '{surface_id}' "
                        f"after operation {deleted_at[surface_id]} deleted it"
                    )
                    continue
                if surface is None:
                    surface = current[surface_id] = Surface(surface_id, first_update=(key, index))
                    surfaces.append(surface)
            else:
                continue
            if key in ("createSurface", "updateComponents"):
                listed = payload.get("components")
                surface.batches.append([c for c in (listed if isinstance(listed, list) else ()) if isinstance(c, dict)])
    return SurfaceWalk(surfaces, errors)


def component_graph_errors(surfaces: Sequence[Surface], types: ComponentTypes) -> list[str]:
    """Check each surface's components as a tree.

    For a surface this response creates: every reference resolves, every component is reachable from
    ``root``, there are no cycles and no id appears twice in one message. A surface created earlier may be
    updated partially, so only duplicate ids and cycles among this response's own components are checked.
    Wherever both ends of a reference are in this response, the composition constraints apply, and a ``root`` sits
    under ``Surface``, on any surface: "the component with ``"id": "root"`` mounts as the child of the surface's
    canonical ``Surface`` container" (v1.0 a2ui_protocol.md, ``updateComponents``). Each component's type is read
    from its own ``@catalogId`` or its surface's ``catalogId`` (v1.0 "Mixable catalogs and component resolution
    logic").
    """
    errors: list[str] = []
    for surface in surfaces:
        errors.extend(_surface_errors(surface, types))
    return errors


def missing_root_errors(surfaces: Sequence[Surface]) -> list[str]:
    """A surface this response creates must have a ``root`` component among the components it is given.

    The spec asks for a ``root`` in one of a surface's component lists, so a partial update to a surface created
    earlier (by an earlier turn or a server-side handler) needs none. v1.0's ``createSurface`` may carry the
    initial components itself. A surface deleted and created again needs a ``root`` of its own.
    """
    return [
        f"No component with id 'root' found on surface '{surface.surface_id}' — "
        "the A2UI spec requires the component tree to have a 'root' node."
        for surface in surfaces
        if surface.created
        and any(surface.batches)
        and not any(component.get("id") == "root" for batch in surface.batches for component in batch)
    ]


def _type_of(component: JsonObject, surface: Surface, types: ComponentTypes) -> ComponentType | None:
    """The definition ``component`` resolves to: in its ``@catalogId``, else in its surface's catalog.

    ``None`` when that catalog does not define the type, when it is not one the agent serves, or when a surface
    created here names no default catalog; the schema check reports each of these. Only on a surface from an
    earlier turn, whose catalog this response does not show, is the type looked up by name in any served catalog.
    """
    name = component.get("component")
    if not isinstance(name, str):
        return None
    own = component.get("@catalogId")
    if isinstance(own, str):
        return types.get((own, name))
    if surface.created:
        return types.get((surface.catalog_id, name)) if surface.catalog_id is not None else None
    candidates = [definition for (_, type_name), definition in types.items() if type_name == name]
    return candidates[-1] if candidates else None


def _surface_errors(surface: Surface, types: ComponentTypes) -> Iterator[str]:
    prefix = f"Surface '{surface.surface_id}':"
    # A component with no resolvable type hides its references; the schema check reports it, and the tree is
    # judged as far as it is known.
    unresolved = False
    edges: dict[str, list[str]] = {}
    kinds: dict[str, _Typed] = {}
    for components in surface.batches:
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
            known = _type_of(component, surface, types)
            if isinstance(component_type, str):
                kinds[component_id] = _Typed(component_type, known)
                unresolved = unresolved or known is None
            edges[component_id] = list(_referenced_ids(component, known.references if known else ()))
    complete = surface.created and not unresolved

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

    yield from _composition_errors(prefix, edges, kinds)


def _composition_errors(
    prefix: str,
    edges: Mapping[str, list[str]],
    kinds: Mapping[str, _Typed],
) -> Iterator[str]:
    """Each parent and child whose types the catalog's ``allowedParents`` / ``allowedChildren`` do not let nest.

    The renderer rejects these with ``UNALLOWED_PARENT`` / ``UNALLOWED_CHILD``. ``root`` is the child of the
    surface's ``Surface`` container.
    """
    pairs: list[tuple[str, _Typed, str]] = []
    if "root" in kinds:
        pairs.append((_SURFACE, _Typed(_SURFACE, None), "root"))
    for parent_id, targets in edges.items():
        parent_kind = kinds.get(parent_id)
        if parent_kind is not None:
            pairs.extend((parent_id, parent_kind, target) for target in targets)
    for parent_id, parent_kind, child_id in pairs:
        child_kind = kinds.get(child_id)
        if child_kind is None:
            continue
        parent_type, parent = parent_kind.type_name, parent_kind.definition
        child_type, child = child_kind.type_name, child_kind.definition
        if child is not None and child.allowed_parents is not None and parent_type not in child.allowed_parents:
            allowed = ", ".join(map(repr, sorted(child.allowed_parents))) or "none"
            yield (
                f"{prefix} component '{child_id}' ({child_type}) cannot be placed in '{parent_id}' ({parent_type}): "
                f"its allowedParents are {allowed}"
            )
        if parent is not None and parent.allowed_children is not None and child_type not in parent.allowed_children:
            allowed = ", ".join(map(repr, sorted(parent.allowed_children))) or "none"
            yield (
                f"{prefix} component '{parent_id}' ({parent_type}) cannot contain '{child_id}' ({child_type}): "
                f"its allowedChildren are {allowed}"
            )


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


def is_json_pointer(path: str) -> bool:
    """Whether ``path`` is an RFC 6901 JSON Pointer: empty, or ``/``-prefixed tokens with ``~`` only as ``~0`` / ``~1``."""
    return _JSON_POINTER.fullmatch(path) is not None


__all__ = (
    "ComponentType",
    "ComponentTypes",
    "Surface",
    "SurfaceWalk",
    "component_graph_errors",
    "component_types",
    "is_json_pointer",
    "missing_root_errors",
    "walk_surfaces",
)
