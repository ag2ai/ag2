# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ....import_utils import optional_import_block, require_optional_import
from .a2a_helpers import A2UI_DEFAULT_DELIMITER

logger = logging.getLogger(__name__)

with optional_import_block():
    import jsonschema


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

    operations: list[dict[str, Any]]
    """The parsed A2UI operation objects."""

    has_a2ui: bool
    """Whether the response contained A2UI content."""

    raw_json: str | None = None
    """The raw JSON string extracted from the response, if any."""

    parse_error: str | None = None
    """Error message if JSON parsing failed."""


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) wrapping JSON content."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence line (e.g. ```json or just ```)
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
        server_to_client_schema: dict[str, Any] | None = None,
        schema_registry: Any | None = None,
        component_schemas: dict[str, dict[str, Any]] | None = None,
        catalog_id: str | None = None,
    ) -> None:
        self._delimiter = delimiter
        self._schema = server_to_client_schema
        self._registry = schema_registry
        self._version_string = version_string
        self._component_schemas = component_schemas or {}
        self._catalog_id = catalog_id

    def parse(self, response: str) -> A2UIParseResult:
        """Extract text and A2UI operations from an agent response.

        Args:
            response: The full agent response string.

        Returns:
            An A2UIParseResult with the text, operations, and any errors.
        """
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

        # Normalize: accept both a single object and an array
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
        """Format validation errors as feedback for the LLM to self-correct.

        Args:
            parse_result: The parsed response that failed validation.
            validation_result: The validation result with error details.

        Returns:
            A feedback message instructing the LLM to fix the errors.
        """
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

    @require_optional_import(["jsonschema"], "a2ui")
    def validate(self, operations: list[dict[str, Any]]) -> A2UIValidationResult:
        """Validate A2UI operations against the server_to_client schema.

        Args:
            operations: List of A2UI operation dicts to validate.

        Returns:
            An A2UIValidationResult indicating validity and any errors.
        """
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
                # Try to produce per-component errors for updateComponents
                comp_errors = self._drill_into_components(op)
                if comp_errors:
                    errors.extend(f"Operation {i}: {ce}" for ce in comp_errors)
                else:
                    # Fall back to re-validating for the top-level message
                    try:
                        if validator is not None:
                            validator.validate(op)
                        else:
                            jsonschema.validate(instance=op, schema=self._schema)
                    except jsonschema.ValidationError as e2:
                        errors.append(f"Operation {i}: {e2.message}")

        return A2UIValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
        )

    def _drill_into_components(self, op: dict[str, Any]) -> list[str]:
        """Validate individual components in an updateComponents operation.

        Returns a list of per-component error strings, or an empty list if
        this operation is not an updateComponents or has no component schemas.
        """
        if "updateComponents" not in op or not self._component_schemas:
            return []
        if not self._catalog_id:
            logger.debug("Skipping per-component validation: no catalog_id configured")
            return []

        components = op.get("updateComponents", {}).get("components", [])
        if not components:
            return []

        comp_errors: list[str] = []
        for comp in components:
            comp_type = comp.get("component", "unknown")
            comp_id = comp.get("id", "?")
            if comp_type not in self._component_schemas:
                comp_errors.append(f"Component '{comp_id}': unknown component type '{comp_type}'")
                continue
            # Build a tiny schema that $refs into the catalog's component definition
            ref_schema = {"$ref": f"{self._catalog_id}#/components/{comp_type}"}
            try:
                if self._registry is not None:
                    validator_cls = jsonschema.validators.validator_for(ref_schema)
                    comp_validator = validator_cls(ref_schema, registry=self._registry)
                    comp_validator.validate(comp)
                else:
                    jsonschema.validate(instance=comp, schema=ref_schema)
            except jsonschema.ValidationError as ce:
                comp_errors.append(f"Component '{comp_id}' ({comp_type}): {ce.message}")
            except jsonschema.RefResolutionError as re:
                logger.warning("Schema ref resolution failed for component '%s' (%s): %s", comp_id, comp_type, re)
            except Exception as exc:
                logger.warning("Unexpected error validating component '%s' (%s): %s", comp_id, comp_type, exc)
        return comp_errors
