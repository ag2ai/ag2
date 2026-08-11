# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import jsonschema
from referencing.exceptions import Unresolvable

from ._types import A2UIVersion, JsonSchema, JsonValue, ServerToClientMessage
from .constants import A2UI_JSON_CLOSE_TAG, A2UI_JSON_OPEN_TAG

if TYPE_CHECKING:
    from referencing import Registry

logger = logging.getLogger(__name__)


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
    """Remove markdown code fences (`````json ... `````) wrapping JSON content."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
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

    # The parsed JSON is a candidate A2UI message (object) or list of them; its
    # conformance to ``ServerToClientMessage`` is enforced downstream by
    # ``validate()`` against the schema, so the cast is the typed boundary
    # between untyped wire JSON and the structured message type.
    if isinstance(parsed, dict):
        return [cast(ServerToClientMessage, parsed)], None
    if isinstance(parsed, list):
        return cast("list[ServerToClientMessage]", parsed), None
    return [], f"Expected JSON array or object, got {type(parsed).__name__}"


def _parse_jsonl(json_part: str) -> "tuple[list[ServerToClientMessage], str | None]":
    operations: list[ServerToClientMessage] = []
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
        operations.append(cast(ServerToClientMessage, value))
    if not operations:
        return [], "No JSON objects found"
    return operations, None


def _components_of(update_components: object) -> list[object]:
    """Return the ``components`` list of an ``updateComponents`` value, or ``[]`` if malformed."""
    if not isinstance(update_components, dict):
        return []
    components = update_components.get("components", [])
    return components if isinstance(components, list) else []


class A2UIResponseParser:
    """Parses and validates A2UI messages from agent responses.

    Extracts the block the LLM wraps between the official A2UI tags
    (``<a2ui-json>…</a2ui-json>``), treating everything outside the tag as
    conversational prose, and optionally validates the messages against the
    A2UI schema.
    """

    def __init__(
        self,
        version_string: A2UIVersion,
        server_to_client_schema: JsonSchema | None = None,
        schema_registry: "Registry | None" = None,
        component_schemas: dict[str, JsonSchema] | None = None,
        catalog_id: str | None = None,
    ) -> None:
        self._schema = server_to_client_schema
        self._registry = schema_registry
        self._version_string = version_string
        self._component_schemas = component_schemas or {}
        self._catalog_id = catalog_id

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
        """Validate A2UI operations against the server_to_client schema."""
        if self._schema is None:
            return A2UIValidationResult(is_valid=True)

        errors: list[str] = []

        if self._registry is not None:
            validator_cls = jsonschema.validators.validator_for(self._schema)
            validator = validator_cls(self._schema, registry=self._registry)
        else:
            validator = None

        for i, op in enumerate(operations):
            try:
                if validator is not None:
                    validator.validate(op)
                else:
                    jsonschema.validate(instance=op, schema=self._schema)
            except jsonschema.ValidationError as e:
                comp_errors = self._drill_into_components(op)
                if comp_errors:
                    errors.extend(f"Operation {i}: {ce}" for ce in comp_errors)
                else:
                    errors.append(f"Operation {i}: {e.message}")
            except Unresolvable as re:
                # A ref we cannot resolve means we could not validate this op —
                # surface it instead of letting it escape as an unhandled crash.
                # (``referencing.exceptions.Unresolvable`` replaces the deprecated
                # ``jsonschema.RefResolutionError`` as of jsonschema 4.18.)
                logger.warning("Schema ref resolution failed for operation %d: %s", i, re)
                errors.append(f"Operation {i}: could not resolve schema reference ({re})")
            except Exception as exc:
                logger.warning("Unexpected error validating operation %d: %s", i, exc)
                errors.append(f"Operation {i}: unexpected validation error ({exc})")

        # Spec rule (server_to_client.json + a2ui.org):
        # across all updateComponents ops, at least one component must have id 'root'.
        update_components_ops = [op for op in operations if isinstance(op, dict) and "updateComponents" in op]
        if update_components_ops:
            has_root = any(
                isinstance(c, dict) and c.get("id") == "root"
                for op in update_components_ops
                for c in _components_of(op.get("updateComponents"))
            )
            if not has_root:
                errors.append(
                    "No component with id 'root' found across updateComponents — "
                    "the A2UI spec requires the component tree to have a 'root' node."
                )

        return A2UIValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
        )

    def _drill_into_components(self, op: ServerToClientMessage) -> list[str]:
        """Validate individual components in an updateComponents operation."""
        if not isinstance(op, dict):
            return [f"Expected operation to be an object, got {type(op).__name__}"]
        if "updateComponents" not in op or not self._component_schemas:
            return []

        components = _components_of(op.get("updateComponents"))
        if not components:
            return []

        comp_errors: list[str] = []
        for comp in components:
            if not isinstance(comp, dict):
                comp_errors.append(f"Expected component to be an object, got {type(comp).__name__}")
                continue
            comp_type = comp.get("component", "unknown")
            comp_id = comp.get("id", "?")
            schema = self._component_schemas.get(comp_type)
            if schema is None:
                comp_errors.append(f"Component '{comp_id}': unknown component type '{comp_type}'")
                continue
            # Prefer a ref via catalog_id (resolves via registry against the merged
            # catalog) but fall back to validating against the inlined component
            # schema directly when no catalog_id is configured.
            ref_or_schema: JsonSchema = (
                {"$ref": f"{self._catalog_id}#/components/{comp_type}"} if self._catalog_id else schema
            )
            try:
                if self._registry is not None:
                    validator_cls = jsonschema.validators.validator_for(ref_or_schema)
                    comp_validator = validator_cls(ref_or_schema, registry=self._registry)
                    comp_validator.validate(comp)
                else:
                    jsonschema.validate(instance=comp, schema=ref_or_schema)
            except jsonschema.ValidationError as ce:
                comp_errors.append(f"Component '{comp_id}' ({comp_type}): {ce.message}")
            except Unresolvable as re:
                # Do not swallow: a ref we cannot resolve means we could not
                # validate the component, so surface it as an error instead of
                # silently treating an unvalidated component as valid.
                # (``Unresolvable`` replaces deprecated ``jsonschema.RefResolutionError``.)
                logger.warning("Schema ref resolution failed for component '%s' (%s): %s", comp_id, comp_type, re)
                comp_errors.append(f"Component '{comp_id}' ({comp_type}): could not resolve schema reference ({re})")
            except Exception as exc:
                logger.warning("Unexpected error validating component '%s' (%s): %s", comp_id, comp_type, exc)
                comp_errors.append(f"Component '{comp_id}' ({comp_type}): unexpected validation error ({exc})")
        return comp_errors
