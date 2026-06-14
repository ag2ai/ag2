# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jsonschema

from ._types import JsonSchema, ServerToClientMessage
from .constants import A2UI_DEFAULT_DELIMITER

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


class A2UIResponseParser:
    """Parses and validates A2UI JSON from agent responses.

    Looks for a delimiter in the agent's response text, extracts the JSON
    array after it, and optionally validates against the A2UI schema.
    """

    def __init__(
        self,
        version_string: str,
        delimiter: str = A2UI_DEFAULT_DELIMITER,
        server_to_client_schema: JsonSchema | None = None,
        schema_registry: "Registry | None" = None,
        component_schemas: dict[str, JsonSchema] | None = None,
        catalog_id: str | None = None,
    ) -> None:
        self._delimiter = delimiter
        self._schema = server_to_client_schema
        self._registry = schema_registry
        self._version_string = version_string
        self._component_schemas = component_schemas or {}
        self._catalog_id = catalog_id

    @property
    def delimiter(self) -> str:
        return self._delimiter

    @property
    def version_string(self) -> str:
        return self._version_string

    def parse(self, response: str) -> A2UIParseResult:
        """Extract text and A2UI operations from an agent response."""
        if self._delimiter not in response:
            return A2UIParseResult(
                text=response.strip(),
                operations=[],
                has_a2ui=False,
            )

        text_part, json_part = response.split(self._delimiter, 1)
        json_part = strip_markdown_fences(json_part)

        if not json_part:
            return A2UIParseResult(
                text=text_part.strip(),
                operations=[],
                has_a2ui=False,
            )

        try:
            parsed = json.loads(json_part)
        except json.JSONDecodeError as e:
            return A2UIParseResult(
                text=text_part.strip(),
                operations=[],
                has_a2ui=True,
                raw_json=json_part,
                parse_error=f"Invalid JSON: {e}",
            )

        if isinstance(parsed, dict):
            parsed = [parsed]

        if not isinstance(parsed, list):
            return A2UIParseResult(
                text=text_part.strip(),
                operations=[],
                has_a2ui=True,
                raw_json=json_part,
                parse_error=f"Expected JSON array or object, got {type(parsed).__name__}",
            )

        return A2UIParseResult(
            text=text_part.strip(),
            operations=parsed,
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
            except jsonschema.ValidationError:
                comp_errors = self._drill_into_components(op)
                if comp_errors:
                    errors.extend(f"Operation {i}: {ce}" for ce in comp_errors)
                else:
                    try:
                        if validator is not None:
                            validator.validate(op)
                        else:
                            jsonschema.validate(instance=op, schema=self._schema)
                    except jsonschema.ValidationError as e2:
                        errors.append(f"Operation {i}: {e2.message}")
            except jsonschema.RefResolutionError as re:
                # A ref we cannot resolve means we could not validate this op —
                # surface it instead of letting it escape as an unhandled crash.
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
                for c in op.get("updateComponents", {}).get("components", [])
            )
            if not has_root:
                errors.append(
                    "No component with id 'root' found across updateComponents — "
                    "the A2UI v0.9 spec requires the component tree to have a 'root' node."
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

        update_components = op.get("updateComponents")
        components = update_components.get("components", []) if isinstance(update_components, dict) else []
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
            except jsonschema.RefResolutionError as re:
                # Do not swallow: a ref we cannot resolve means we could not
                # validate the component, so surface it as an error instead of
                # silently treating an unvalidated component as valid.
                logger.warning("Schema ref resolution failed for component '%s' (%s): %s", comp_id, comp_type, re)
                comp_errors.append(f"Component '{comp_id}' ({comp_type}): could not resolve schema reference ({re})")
            except Exception as exc:
                logger.warning("Unexpected error validating component '%s' (%s): %s", comp_id, comp_type, exc)
                comp_errors.append(f"Component '{comp_id}' ({comp_type}): unexpected validation error ({exc})")
        return comp_errors
