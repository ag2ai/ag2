# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from referencing import Registry
from referencing.exceptions import Unresolvable

from ._graph import ComponentReferences, component_graph_errors, is_json_pointer
from ._messages import is_server_message
from ._types import (
    JsonObject,
    JsonSchema,
    JsonValue,
    ServerToClientMessage,
    server_to_client_payload_keys,
)
from ._versions import A2UIProtocol
from .constants import A2UI_JSON_CLOSE_TAG, A2UI_JSON_OPEN_TAG
from .diagnostics import MessageChecker
from .schema_manager import _registry_for_surface_catalog

logger = logging.getLogger(__name__)

_SINGLE_LINE_FENCE_INFO_STRING_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*(?=\s|[{[])")


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

    Handles multi-line fences (info string on the opening line) and single-line
    fences with an optional info string (e.g. ```json {"a": 1}```).
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else _SINGLE_LINE_FENCE_INFO_STRING_RE.sub("", text[3:], count=1)
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _parse_json_block(json_part: str) -> "tuple[list[ServerToClientMessage], str | None]":
    """Parse an ``<a2ui-json>`` block (JSON array/object, or JSONL) into A2UI messages."""
    try:
        parsed: JsonValue = json.loads(json_part)
    except json.JSONDecodeError as whole_error:
        operations, jsonl_error = _parse_jsonl(json_part)
        if jsonl_error is None:
            return operations, None
        # Surface the whole-block error: it's the more informative one when
        # the block was meant to be a single array/object.
        return [], f"Invalid JSON: {whole_error}"

    if isinstance(parsed, dict):
        return _as_messages([parsed])
    if isinstance(parsed, list):
        return _as_messages(parsed)
    return [], f"Expected JSON array or object, got {type(parsed).__name__}"


def _parse_jsonl(json_part: str) -> "tuple[list[ServerToClientMessage], str | None]":
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
    return _as_messages(values)


def _as_messages(values: list[JsonValue]) -> "tuple[list[ServerToClientMessage], str | None]":
    messages: list[ServerToClientMessage] = []
    for i, value in enumerate(values):
        if not is_server_message(value):
            return (
                [],
                f"Operation {i} is not an A2UI message: expected an object with exactly one of {sorted(server_to_client_payload_keys())}",
            )
        messages.append(value)
    return messages, None


def _missing_root_errors(objects: Sequence[JsonValue]) -> list[str]:
    """A surface these messages create must have a ``root`` component among the components they give it.

    The spec asks for a ``root`` in one of a surface's component lists, so a partial update to a surface created
    earlier (by an earlier turn or a server-side handler) needs none. v1.0's ``createSurface`` may carry the
    initial components itself.
    """
    created: dict[str, list[JsonValue]] = {}
    for op in objects:
        payload = op.get("createSurface") if isinstance(op, dict) else None
        surface_id = payload.get("surfaceId") if isinstance(payload, dict) else None
        if isinstance(surface_id, str):
            created.setdefault(surface_id, []).extend(_components_of(payload))
    for op in objects:
        payload = op.get("updateComponents") if isinstance(op, dict) else None
        surface_id = payload.get("surfaceId") if isinstance(payload, dict) else None
        if isinstance(surface_id, str) and surface_id in created:
            created[surface_id].extend(_components_of(payload))
    return [
        f"No component with id 'root' found on surface '{surface_id}' — "
        "the A2UI spec requires the component tree to have a 'root' node."
        for surface_id, components in created.items()
        if components and not any(isinstance(c, dict) and c.get("id") == "root" for c in components)
    ]


def _data_path_errors(messages: Sequence[JsonObject]) -> list[str]:
    """A data-model update's ``path`` must be a JSON Pointer (RFC 6901)."""
    errors: list[str] = []
    for i, message in enumerate(messages):
        payload = message.get("updateDataModel")
        path = payload.get("path") if isinstance(payload, dict) else None
        if isinstance(path, str) and not is_json_pointer(path):
            errors.append(f"Operation {i}: {path!r} is not a JSON Pointer (at /updateDataModel/path)")
    return errors


def _components_of(update_components: JsonValue) -> list[JsonValue]:
    """Return the ``components`` list of an ``updateComponents`` (or v1.0 ``createSurface``) value, or ``[]``."""
    if not isinstance(update_components, dict):
        return []
    components = update_components.get("components", [])
    return components if isinstance(components, list) else []


def _surface_catalog(operation: JsonValue, surfaces: dict[str, str], registry: Registry) -> str | None:
    if not isinstance(operation, dict):
        return None
    for key in ("createSurface", "updateComponents", "callRendererFunction", "deleteSurface"):
        payload = operation.get(key)
        if not isinstance(payload, dict) or not isinstance(surface_id := payload.get("surfaceId"), str):
            continue
        if key == "createSurface":
            catalog_id = payload.get("catalogId")
            default = registry.get("catalog.json")
            if not isinstance(catalog_id, str):
                catalog_id = default.id() if default is not None else None
            if catalog_id is not None:
                surfaces[surface_id] = catalog_id
        elif key == "deleteSurface":
            surfaces.pop(surface_id, None)
        return surfaces.get(surface_id)
    return None


class A2UIResponseParser:
    """Parses and validates A2UI messages from agent responses.

    Extracts the block the LLM wraps between the official A2UI tags
    (``<a2ui-json>…</a2ui-json>``), treating everything outside the tag as
    conversational prose, and optionally validates the messages against the
    A2UI schema.
    """

    def __init__(
        self,
        protocol: A2UIProtocol,
        server_to_client_schema: JsonSchema | None = None,
        schema_registry: Registry | None = None,
        component_references: ComponentReferences | None = None,
    ) -> None:
        self._checker = (
            MessageChecker(server_to_client_schema, schema_registry) if server_to_client_schema is not None else None
        )
        self._protocol = protocol
        self._schema = server_to_client_schema
        self._registry = schema_registry
        self._component_references = component_references

    @property
    def protocol(self) -> A2UIProtocol:
        return self._protocol

    def parse(self, response: str) -> A2UIParseResult:
        """Extract conversational text and A2UI messages from a response.

        Text before/after the ``<a2ui-json>`` block is the prose; the block
        content is parsed as a JSON array/object or JSONL. A missing closing
        tag is tolerated — the block runs to the end of the response.
        """
        open_idx = response.find(A2UI_JSON_OPEN_TAG)
        if open_idx == -1:
            return A2UIParseResult(
                text=response.strip(),
                operations=[],
                has_a2ui=False,
            )

        text_before = response[:open_idx]
        after_open = response[open_idx + len(A2UI_JSON_OPEN_TAG) :]

        close_idx = after_open.find(A2UI_JSON_CLOSE_TAG)
        if close_idx == -1:
            # Tolerate a missing closing tag: take everything to the end.
            json_part = after_open
            text_after = ""
        else:
            json_part = after_open[:close_idx]
            text_after = after_open[close_idx + len(A2UI_JSON_CLOSE_TAG) :]

        text = " ".join(part.strip() for part in (text_before, text_after) if part.strip())
        json_part = strip_markdown_fences(json_part)

        if not json_part:
            return A2UIParseResult(
                text=text,
                operations=[],
                has_a2ui=False,
            )

        operations, parse_error = _parse_json_block(json_part)
        if parse_error is not None:
            return A2UIParseResult(
                text=text,
                operations=[],
                has_a2ui=True,
                raw_json=json_part,
                parse_error=parse_error,
            )

        return A2UIParseResult(
            text=text,
            operations=operations,
            has_a2ui=True,
            raw_json=json_part,
        )

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
        surface_catalogs: dict[str, str] = {}
        catalog_checkers: dict[str, MessageChecker] = {}
        for i, op in enumerate(objects):
            try:
                checker = self._checker
                if self._protocol.per_surface_catalogs and self._registry is not None and self._schema is not None:
                    catalog_id = _surface_catalog(op, surface_catalogs, self._registry)
                    if catalog_id is not None:
                        if catalog_id not in catalog_checkers:
                            catalog_checkers[catalog_id] = MessageChecker(
                                self._schema, _registry_for_surface_catalog(self._registry, catalog_id)
                            )
                        checker = catalog_checkers[catalog_id]
                errors.extend(f"Operation {i}: {issue.render(op)}" for issue in checker.issues(op))
            except Unresolvable as exc:
                # A model-written ``$ref`` target the registry lacks: report it rather than crash the turn.
                logger.warning("Schema ref resolution failed for operation %d: %s", i, exc)
                errors.append(f"Operation {i}: could not resolve schema reference ({exc})")

        errors.extend(self._check_function_messages(operations))
        errors.extend(_data_path_errors(objects))
        # A tree is only worth checking once its messages are well-formed; otherwise each mistake echoes.
        if self._component_references is not None and not errors:
            errors.extend(component_graph_errors(objects, self._component_references))

        errors.extend(_missing_root_errors(objects))

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
        return errors
