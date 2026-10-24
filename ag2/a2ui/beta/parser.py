# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeGuard

from a2ui.core import STRICT_VALIDATION, A2uiError, MessageProcessor, MessageProcessorOptions
from a2ui.core.catalog import CatalogApi

from ._types import A2UIVersion, JsonValue, ServerToClientMessage, server_to_client_payload_keys
from .constants import A2UI_JSON_CLOSE_TAG, A2UI_JSON_OPEN_TAG

logger = logging.getLogger(__name__)

# Core echoes schema fragments into some errors (e.g. an unresolvable ``$ref``); bound what goes back to the LLM.
_MAX_ERROR_LEN = 2000

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


def _is_message(value: JsonValue) -> TypeGuard[ServerToClientMessage]:
    """Whether ``value`` has the envelope of a message: an object with exactly one payload key.

    The payload's contents are left to ``validate()``, which checks them against the schema.
    """
    return isinstance(value, dict) and len(server_to_client_payload_keys().intersection(value)) == 1


def _as_messages(values: list[JsonValue]) -> "tuple[list[ServerToClientMessage], str | None]":
    messages: list[ServerToClientMessage] = []
    for i, value in enumerate(values):
        if not _is_message(value):
            return (
                [],
                f"Operation {i} is not an A2UI message: expected an object with exactly one of {sorted(server_to_client_payload_keys())}",
            )
        messages.append(value)
    return messages, None


class A2UIResponseParser:
    """Parses and validates A2UI messages from agent responses.

    Extracts the block the LLM wraps between the official A2UI tags
    (``<a2ui-json>…</a2ui-json>``), treating everything outside the tag as
    conversational prose, and optionally validates the messages with ``a2ui-core``
    against the active catalog.
    """

    def __init__(
        self,
        version_string: A2UIVersion,
        catalog: CatalogApi | None = None,
    ) -> None:
        self._catalog = catalog
        self._version_string = version_string

    @property
    def version_string(self) -> A2UIVersion:
        return self._version_string

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

    def format_validation_error(
        self,
        parse_result: A2UIParseResult,
        validation_result: A2UIValidationResult,
    ) -> str:
        """Format validation errors as feedback for the LLM to self-correct."""
        lines = ["Your A2UI output had validation errors:"]
        for error in validation_result.errors:
            lines.append(f"- {error}")
        if parse_result.parse_error:
            lines.append(f"- JSON parse error: {parse_result.parse_error}")
        lines.append("")
        lines.append(
            "Please fix these errors and regenerate the A2UI JSON. "
            f'Make sure each message includes "version": "{self._version_string}" and all required properties.'
        )
        return "\n".join(lines)

    def validate(self, operations: Sequence[ServerToClientMessage]) -> A2UIValidationResult:
        """Validate A2UI operations with ``a2ui-core``.

        The operations are replayed through a core ``MessageProcessor`` in strict mode,
        so besides the schema it rejects orphan components, dangling child ids, cycles
        and a missing ``root``. A surface the batch does not create is validated against
        a synthetic ``createSurface``. ``updateComponents`` of one surface are validated
        as one tree, so a response may split a surface across messages.
        """
        if self._catalog is None:
            return A2UIValidationResult(is_valid=True)

        errors: list[str] = []
        processor = MessageProcessor(
            [self._catalog],
            options=MessageProcessorOptions(validation_config=STRICT_VALIDATION),
        )
        failed_surfaces: set[str] = set()
        known_surfaces = {_surface_id(op) for op in operations if isinstance(op, dict) and "createSurface" in op}

        for i, op in self._coalesce_components(operations):
            if not isinstance(op, dict):
                errors.append(f"Operation {i}: expected an object, got {type(op).__name__}")
                continue
            surface_id = _surface_id(op)
            if surface_id is not None and surface_id in failed_surfaces:
                continue  # its first error already explains the rest
            try:
                if surface_id is not None and surface_id not in known_surfaces:
                    known_surfaces.add(surface_id)
                    processor.process_messages([self._synthetic_create_surface(surface_id, op)])
                processor.process_messages([op])
            except A2uiError as e:
                errors.append(_bounded(f"Operation {i}: {e}"))
                if surface_id is not None:
                    failed_surfaces.add(surface_id)
            except Exception as exc:
                # Core validates payloads with pydantic, whose errors are not A2uiErrors.
                logger.warning("Unexpected error validating operation %d: %s", i, exc)
                errors.append(_bounded(f"Operation {i}: {exc}"))
                if surface_id is not None:
                    failed_surfaces.add(surface_id)

        return A2UIValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
        )

    def _synthetic_create_surface(self, surface_id: str, op: Mapping[str, Any]) -> dict[str, Any]:
        assert self._catalog is not None
        version = op.get("version", self._version_string)
        return {"version": version, "createSurface": {"surfaceId": surface_id, "catalogId": self._catalog.id}}

    @staticmethod
    def _coalesce_components(operations: Sequence[ServerToClientMessage]) -> list[tuple[int, Any]]:
        """Pair each operation with its index, folding a surface's ``updateComponents`` into its first one.

        A later component replaces an earlier one with the same id, which is what the
        renderer does. The input operations are not modified.
        """
        first_of: dict[str, int] = {}
        merged: dict[str, dict[str, Any]] = {}
        out: list[tuple[int, Any]] = []
        for i, op in enumerate(operations):
            surface_id = _surface_id(op) if isinstance(op, dict) and "updateComponents" in op else None
            payload = op["updateComponents"] if surface_id is not None else None  # type: ignore[typeddict-item]
            if surface_id is None or not isinstance(payload, dict) or not isinstance(payload.get("components"), list):
                out.append((i, op))
                continue
            by_id = merged.setdefault(surface_id, {})
            for position, component in enumerate(payload["components"]):
                key = (
                    component["id"]
                    if isinstance(component, dict) and isinstance(component.get("id"), str)
                    else f"#{i}.{position}"
                )
                by_id[key] = component
            if surface_id not in first_of:
                first_of[surface_id] = len(out)
                out.append((i, op))
        for surface_id, position in first_of.items():
            index, op = out[position]
            out[position] = (
                index,
                {**op, "updateComponents": {**op["updateComponents"], "components": list(merged[surface_id].values())}},
            )
        return out


def _bounded(error: str) -> str:
    return error if len(error) <= _MAX_ERROR_LEN else error[:_MAX_ERROR_LEN] + "…[truncated]"


def _surface_id(op: Mapping[str, Any]) -> str | None:
    """The ``surfaceId`` of a message payload, or ``None`` for messages without a surface."""
    for key in ("createSurface", "updateComponents", "updateDataModel", "deleteSurface"):
        payload = op.get(key)
        if isinstance(payload, dict):
            surface_id = payload.get("surfaceId")
            return surface_id if isinstance(surface_id, str) else None
    return None
