# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeGuard, cast

from referencing import Registry
from referencing.exceptions import Unresolvable

from ._types import A2UIVersion, JsonObject, JsonSchema, JsonValue, ServerToClientMessage, server_to_client_payload_keys
from .constants import A2UI_JSON_CLOSE_TAG, A2UI_JSON_OPEN_TAG
from .diagnostics import MessageChecker

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


def _components_of(update_components: JsonValue) -> list[JsonValue]:
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
        schema_registry: Registry | None = None,
        function_callers: Mapping[str, str] | None = None,
    ) -> None:
        self._checker = (
            MessageChecker(server_to_client_schema, schema_registry) if server_to_client_schema is not None else None
        )
        self._version_string = version_string
        self._function_callers = function_callers

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
            f'Make sure each message includes "version": "{self._version_string}" and all required properties. '
            "The user never sees this exchange: do not mention the errors or the correction in your reply."
        )
        return "\n".join(lines)

    def validate(self, operations: Sequence[ServerToClientMessage]) -> A2UIValidationResult:
        """Validate A2UI operations against the server_to_client schema."""
        if self._checker is None:
            return A2UIValidationResult(is_valid=True)

        # Message TypedDicts are JSON objects on the wire; the checker reads them as such.
        objects = cast(Sequence[JsonObject], operations)
        errors: list[str] = []
        for i, op in enumerate(objects):
            try:
                errors.extend(f"Operation {i}: {issue.render(op)}" for issue in self._checker.issues(op))
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

        errors.extend(self._check_function_messages(operations))

        # Spec rule (server_to_client.json + a2ui.org): across all component-carrying
        # ops, at least one component must have id 'root'. v1.0's createSurface can
        # carry the initial components itself, so those count alongside updateComponents.
        update_ops = [op for op in objects if isinstance(op, dict) and "updateComponents" in op]
        create_lists = [
            _components_of(op.get("createSurface")) for op in objects if isinstance(op, dict) and "createSurface" in op
        ]
        component_lists = [_components_of(op.get("updateComponents")) for op in update_ops] + create_lists
        if update_ops or any(create_lists):
            has_root = any(isinstance(c, dict) and c.get("id") == "root" for lst in component_lists for c in lst)
            if not has_root:
                errors.append(
                    "No component with id 'root' found across the surface's components — "
                    "the A2UI spec requires the component tree to have a 'root' node."
                )

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
            call_function = call.get("callFunction") if isinstance(call, dict) else None
            name = call_function.get("@call") if isinstance(call_function, dict) else None
            if (
                isinstance(name, str)
                and self._function_callers is not None
                and self._function_callers.get(name, "rendererOnly") == "rendererOnly"
            ):
                errors.append(
                    f"Operation {i}: function '{name}' is 'rendererOnly' in the catalog (allowedCallers), "
                    "so the agent cannot call it with 'callRendererFunction'."
                )
        return errors
