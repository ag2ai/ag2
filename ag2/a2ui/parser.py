# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, cast

from referencing import Registry
from referencing.exceptions import Unresolvable

from ._graph import ComponentTypes, component_graph_errors, is_json_pointer, missing_root_errors, walk_surfaces
from ._messages import is_server_message
from ._types import (
    JsonObject,
    JsonSchema,
    JsonValue,
    ServerToClientMessage,
)
from ._versions import A2UIProtocol
from .constants import A2UI_JSON_CLOSE_TAG, A2UI_JSON_OPEN_TAG
from .diagnostics import MessageChecker
from .schema_manager import A2UIValidationSchemas, _registry_for_surface_catalog, _registry_for_unknown_catalog

logger = logging.getLogger(__name__)

# The info string runs to the end of the line or to the start of a JSON payload
# written on the fence's own line, whichever comes first.
_OPENING_FENCE_RE = re.compile(r"\A`{3,}[^\n{[]*\n?")
_CLOSING_FENCE_RE = re.compile(r"\s*`{3,}\Z")


@dataclass
class A2UIValidationResult:
    """Result of validating A2UI operations against the schema."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class A2UIParseResult:
    """Result of parsing an agent response for A2UI content."""

    text: str
    """The conversational text portion of the response."""

    operations: list[ServerToClientMessage]
    """The parsed A2UI operation objects."""

    has_a2ui: bool
    """Whether the response contained A2UI content."""

    raw_json: str | None = None
    """The raw JSON string extracted from the response, if any."""

    parse_error: str | None = None
    """Error message if JSON parsing failed."""


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (`````json ... `````) wrapping JSON content.

    The payload may start on the opening fence's line (e.g. ```json {"a": 1}```)
    or on the next one, and the closing fence may share its last line or stand
    on its own.
    """
    text = _OPENING_FENCE_RE.sub("", text.strip(), count=1)
    return _CLOSING_FENCE_RE.sub("", text, count=1).strip()


def _split_blocks(response: str) -> tuple[list[str], list[str]]:
    """The prose segments and the non-empty ``<a2ui-json>`` block bodies of ``response``, each in order.

    Upstream lets a response hold "one or more A2UI JSON blocks" (``a2ui_agent`` ``constants.py``), and a close tag
    written inside a JSON string does not end its block (``conformance/agent/direct_json/response_parser.yaml``).
    """
    texts: list[str] = []
    blocks: list[str] = []
    pos = 0
    while (open_idx := response.find(A2UI_JSON_OPEN_TAG, pos)) != -1:
        texts.append(response[pos:open_idx])
        body_start = open_idx + len(A2UI_JSON_OPEN_TAG)
        close_idx = _find_close_tag(response, body_start)
        body_end = len(response) if close_idx is None else close_idx
        if body := strip_markdown_fences(response[body_start:body_end]):
            blocks.append(body)
        pos = len(response) if close_idx is None else close_idx + len(A2UI_JSON_CLOSE_TAG)
    texts.append(response[pos:])
    return texts, blocks


def _find_close_tag(response: str, start: int) -> int | None:
    """Where the close tag of the block whose body starts at ``start`` is, skipping JSON strings; ``None`` if absent."""
    in_string = False
    i = start
    n = len(response)
    while i < n:
        c = response[i]
        if in_string:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif response.startswith(A2UI_JSON_CLOSE_TAG, i):
            return i
        i += 1
    if in_string:
        # An unterminated string (malformed block): fall back to the first close tag, so the prose after it survives.
        close_idx = response.find(A2UI_JSON_CLOSE_TAG, start)
        return None if close_idx == -1 else close_idx
    return None


def _parse_json_block(json_part: str, kinds: Collection[str]) -> "tuple[list[ServerToClientMessage], str | None]":
    """Parse an ``<a2ui-json>`` block (JSON array/object, or JSONL) into A2UI messages.

    ``kinds`` are the served version's message kinds, named when a value is not a message.
    """
    try:
        parsed: JsonValue = json.loads(json_part)
    except json.JSONDecodeError as whole_error:
        operations, jsonl_error = _parse_jsonl(json_part, kinds)
        if jsonl_error is None:
            return operations, None
        # Surface the whole-block error: it's the more informative one when
        # the block was meant to be a single array/object.
        return [], f"Invalid JSON: {whole_error}"

    if isinstance(parsed, dict):
        return _as_messages([parsed], kinds)
    if isinstance(parsed, list):
        return _as_messages(parsed, kinds)
    return [], f"Expected JSON array or object, got {type(parsed).__name__}"


def _parse_jsonl(json_part: str, kinds: Collection[str]) -> "tuple[list[ServerToClientMessage], str | None]":
    values: list[JsonValue] = []
    for line in json_part.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value: JsonValue = json.loads(line)
        except json.JSONDecodeError as e:
            return [], f"Invalid JSON: {e}"
        if not isinstance(value, dict):
            return [], f"Expected JSON object per line, got {type(value).__name__}"
        values.append(value)
    if not values:
        return [], "No JSON objects found"
    return _as_messages(values, kinds)


def _as_messages(values: list[JsonValue], kinds: Collection[str]) -> "tuple[list[ServerToClientMessage], str | None]":
    messages: list[ServerToClientMessage] = []
    for i, value in enumerate(values):
        if not is_server_message(value):
            return (
                [],
                f"Operation {i} is not an A2UI message: expected an object with exactly one of {sorted(kinds)}",
            )
        messages.append(value)
    return messages, None


def _data_path_errors(messages: Sequence[JsonObject]) -> list[str]:
    """A data-model update's ``path`` must be a JSON Pointer (RFC 6901)."""
    errors: list[str] = []
    for i, message in enumerate(messages):
        payload = message.get("updateDataModel")
        path = payload.get("path") if isinstance(payload, dict) else None
        if isinstance(path, str) and not is_json_pointer(path):
            errors.append(f"Operation {i}: {path!r} is not a JSON Pointer (at /updateDataModel/path)")
    return errors


# An absolute JSON Pointer, or a relative path read in a collection scope (v1_0 a2ui_protocol.md "Collection scopes
# (relative paths)"); either way ``~`` only as ``~0`` / ``~1``. The same pattern as a2ui-core's RELAXED_PATH_PATTERN.
_BINDING_PATH = re.compile(r"(?:/(?:[^~/]|~[01])*)*|(?:[^~/]|~[01])+(?:/(?:[^~/]|~[01])*)*")
# A data binding (``{"path"}`` in v0.9, ``{"@path"}`` in v1.0) and a template child list (``{componentId, path}``).
_BINDING_SHAPES = (frozenset({"path"}), frozenset({"@path"}), frozenset({"componentId", "path"}))


def _binding_path_errors(messages: Sequence[JsonObject]) -> list[str]:
    """A data binding in a component must name a JSON Pointer, absolute or relative (v1_0 a2ui_protocol.md, "Data
    binding": "Data bindings in A2UI are defined using JSON Pointers ([RFC 6901])")."""
    errors: list[str] = []
    for i, message in enumerate(messages):
        payload = message.get("updateComponents")
        components = payload.get("components") if isinstance(payload, dict) else None
        if isinstance(components, list):
            location = ("updateComponents", "components")
            errors.extend(f"Operation {i}: {error}" for error in _bindings_in(components, location))
    return errors


def _bindings_in(node: JsonValue, location: tuple[str, ...]) -> Iterator[str]:
    if isinstance(node, list):
        for n, item in enumerate(node):
            yield from _bindings_in(item, (*location, str(n)))
        return
    if not isinstance(node, dict):
        return
    if node.keys() in _BINDING_SHAPES:
        key = "@path" if "@path" in node else "path"
        path = node[key]
        if isinstance(path, str) and _BINDING_PATH.fullmatch(path) is None:
            pointer = "/".join(part.replace("~", "~0").replace("/", "~1") for part in (*location, key))
            yield f"{path!r} is not a valid data path: '~' must be written '~0' or '~1' (at /{pointer})"
    for key, value in node.items():
        yield from _bindings_in(value, (*location, key))


@dataclass(frozen=True, slots=True)
class _UnknownCatalog:
    operation: int
    surface_id: str | None
    catalog_id: str


def _unknown_catalogs(messages: Sequence[JsonObject], catalog_ids: Collection[str]) -> list[_UnknownCatalog]:
    """The ``createSurface``\\s that name a catalog the agent does not have (the SDK answers ``CATALOG_NOT_FOUND``)."""
    found: list[_UnknownCatalog] = []
    for i, message in enumerate(messages):
        payload = message.get("createSurface")
        if not isinstance(payload, dict):
            continue
        catalog_id = payload.get("catalogId")
        if isinstance(catalog_id, str) and catalog_id not in catalog_ids:
            surface_id = payload.get("surfaceId")
            found.append(_UnknownCatalog(i, surface_id if isinstance(surface_id, str) else None, catalog_id))
    return found


def _unknown_catalog_error(unknown: _UnknownCatalog, catalog_ids: Collection[str]) -> str:
    known = ", ".join(repr(known_id) for known_id in catalog_ids)
    return (
        f"Operation {unknown.operation}: unknown catalog {unknown.catalog_id!r} (at /createSurface/catalogId); "
        f"use one of: {known}"
    )


def _targets_surface(message: JsonValue, surfaces: Collection[str]) -> bool:
    """Whether ``message``'s payload names one of ``surfaces`` as its ``surfaceId``."""
    return isinstance(message, dict) and any(
        isinstance(payload, dict) and payload.get("surfaceId") in surfaces for payload in message.values()
    )


class _SurfaceCatalog(Enum):
    UNKNOWN = "unknown"
    """The catalog of a surface this response did not create: any configured catalog may apply."""


def _surface_catalog(
    operation: JsonValue, surfaces: dict[str, str | None]
) -> "str | None | Literal[_SurfaceCatalog.UNKNOWN]":
    """The default catalog of the surface ``operation`` targets: an id, ``None`` (none), or ``UNKNOWN``.

    A v1.0 ``createSurface`` without ``catalogId`` gives its surface no default; upstream has no fallback.
    """
    if not isinstance(operation, dict):
        return _SurfaceCatalog.UNKNOWN
    for key in ("createSurface", "updateComponents", "callRendererFunction", "deleteSurface"):
        payload = operation.get(key)
        if not isinstance(payload, dict) or not isinstance(surface_id := payload.get("surfaceId"), str):
            continue
        if key == "createSurface":
            catalog_id = payload.get("catalogId")
            surfaces[surface_id] = catalog_id if isinstance(catalog_id, str) else None
        elif key == "deleteSurface":
            surfaces.pop(surface_id, None)
        return surfaces.get(surface_id, _SurfaceCatalog.UNKNOWN)
    return _SurfaceCatalog.UNKNOWN


class A2UIResponseParser:
    """Parses and validates A2UI messages from agent responses.

    Extracts the blocks the LLM wraps between the official A2UI tags
    (``<a2ui-json>…</a2ui-json>``), treating everything outside them as
    conversational prose, and optionally validates the messages against the
    A2UI schema.
    """

    def __init__(
        self,
        protocol: A2UIProtocol,
        server_to_client_schema: JsonSchema | None = None,
        schema_registry: Registry | None = None,
        component_types: ComponentTypes | None = None,
        catalog_ids: Collection[str] | None = None,
    ) -> None:
        self._checker = (
            MessageChecker(server_to_client_schema, schema_registry) if server_to_client_schema is not None else None
        )
        self._protocol = protocol
        self._schema = server_to_client_schema
        self._registry = schema_registry
        self._component_types = component_types
        self._catalog_ids = catalog_ids

    @classmethod
    def validating(cls, protocol: A2UIProtocol, schemas: A2UIValidationSchemas) -> "A2UIResponseParser":
        """A parser that validates every message against ``schemas``."""
        return cls(
            protocol=protocol,
            server_to_client_schema=schemas.server_to_client_schema,
            schema_registry=schemas.schema_registry,
            component_types=schemas.component_types,
            catalog_ids=schemas.catalog_ids,
        )

    @property
    def protocol(self) -> A2UIProtocol:
        return self._protocol

    def parse(self, response: str) -> A2UIParseResult:
        """Extract conversational text and A2UI messages from a response.

        A response may hold several ``<a2ui-json>`` blocks; text outside them is
        the prose. Each block is parsed as a JSON array/object or JSONL, and the
        messages of every block are returned in order. A close tag inside a JSON
        string does not end a block, and a missing close tag is tolerated: the
        last block then runs to the end of the response.
        """
        texts, blocks = _split_blocks(response)
        text = " ".join(part.strip() for part in texts if part.strip())
        if not blocks:
            return A2UIParseResult(text=text, operations=[], has_a2ui=False)

        operations: list[ServerToClientMessage] = []
        for n, block in enumerate(blocks):
            block_operations, parse_error = _parse_json_block(block, self._protocol.server_to_client_keys)
            if parse_error is not None:
                where = f"Block {n + 1}: " if len(blocks) > 1 else ""
                return A2UIParseResult(
                    text=text,
                    operations=[],
                    has_a2ui=True,
                    raw_json="\n".join(blocks),
                    parse_error=where + parse_error,
                )
            operations.extend(block_operations)

        return A2UIParseResult(text=text, operations=operations, has_a2ui=True, raw_json="\n".join(blocks))

    def format_validation_error(self, errors: Sequence[str]) -> str:
        """Format validation errors as feedback for the LLM to self-correct."""
        lines = ["Your A2UI output had validation errors:"]
        for error in errors:
            lines.append(f"- {error}")
        lines.append("")
        lines.append(
            "Please fix these errors and regenerate the A2UI JSON. "
            f'Make sure each message includes "version": "{self._protocol.wire_version}" and all required properties. '
            "The user never sees this exchange: do not mention the errors or the correction in your reply."
        )
        return "\n".join(lines)

    def validate(self, operations: Sequence[ServerToClientMessage]) -> A2UIValidationResult:
        """Validate messages, inheriting catalogs from surfaces created in this response.

        Surface context belongs to this validation call, since a parser can be
        shared by clients. For a surface created in an earlier response, an
        implicit function is accepted if any configured catalog supports it;
        without its saved catalog binding we cannot verify that association.
        An explicit function catalog is always checked exactly.
        """
        if self._checker is None:
            return A2UIValidationResult(is_valid=True)

        # Message TypedDicts are JSON objects on the wire; the checker reads them as such.
        objects = cast(Sequence[JsonObject], operations)
        errors: list[str] = []
        unknown_catalogs = _unknown_catalogs(objects, self._catalog_ids) if self._catalog_ids is not None else []
        uncheckable = {unknown.surface_id for unknown in unknown_catalogs if unknown.surface_id is not None}
        surface_catalogs: dict[str, str | None] = {}
        catalog_checkers: dict[str | None, MessageChecker] = {}
        open_checker: MessageChecker | None = None
        for i, op in enumerate(objects):
            try:
                checker = self._checker
                if self._registry is not None and self._schema is not None:
                    if self._protocol.per_surface_catalogs:
                        catalog_id = _surface_catalog(op, surface_catalogs)
                        if catalog_id is not _SurfaceCatalog.UNKNOWN:
                            if catalog_id not in catalog_checkers:
                                catalog_checkers[catalog_id] = MessageChecker(
                                    self._schema, _registry_for_surface_catalog(self._registry, catalog_id)
                                )
                            checker = catalog_checkers[catalog_id]
                    elif _targets_surface(op, uncheckable):
                        # Before v1.0 the catalog is not per surface: open it for a surface on an unknown catalog.
                        if open_checker is None:
                            open_checker = MessageChecker(
                                self._schema,
                                _registry_for_unknown_catalog(self._registry, self._protocol.schema_base_uri),
                            )
                        checker = open_checker
                errors.extend(f"Operation {i}: {issue.render(op)}" for issue in checker.issues(op))
            except Unresolvable as exc:
                # A model-written ``$ref`` target the registry lacks: report it rather than crash the turn.
                logger.warning("Schema ref resolution failed for operation %d: %s", i, exc)
                errors.append(f"Operation {i}: could not resolve schema reference ({exc})")

        errors.extend(self._check_function_messages(operations))
        errors.extend(_data_path_errors(objects))
        errors.extend(_binding_path_errors(objects))
        walk = walk_surfaces(objects, lifetime_unique_surface_ids=self._protocol.lifetime_unique_surface_ids)
        errors.extend(walk.errors)
        # A surface on an unknown catalog gets that one error: its components have no catalog to be judged by.
        checkable = walk.surfaces
        if self._catalog_ids is not None:
            errors.extend(_unknown_catalog_error(unknown, self._catalog_ids) for unknown in unknown_catalogs)
            checkable = [surface for surface in walk.surfaces if surface.surface_id not in uncheckable]
        if self._component_types is not None:
            errors.extend(component_graph_errors(checkable, self._component_types))

        errors.extend(missing_root_errors(checkable))

        return A2UIValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
        )

    def _check_function_messages(self, operations: Sequence[ServerToClientMessage]) -> list[str]:
        """v1.0 rules the JSON schema cannot express: who may send or be called."""
        errors: list[str] = []
        for i, op in enumerate(operations):
            if not isinstance(op, dict):
                continue
            if "agentFunctionResponse" in op:
                errors.append(
                    f"Operation {i}: 'agentFunctionResponse' is sent by the server in reply to a "
                    "'callAgentFunction'; do not emit it."
                )
            call = op.get("callRendererFunction")
            if isinstance(call, dict):
                # The renderer correlates its response by id and runs the call on a surface: neither may be empty
                # (a2ui_agent refuses both; conformance/agent/renderer_function_calls.yaml).
                errors.extend(
                    f"Operation {i}: 'callRendererFunction' needs a non-empty {key!r} (at /callRendererFunction/{key})"
                    for key in ("functionCallId", "surfaceId")
                    if call.get(key) == ""
                )
        return errors
