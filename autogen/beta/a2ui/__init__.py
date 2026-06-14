# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_optional_dependency

try:
    from .actions import A2UIAction
    from .agent import A2UIAgent
    from .events import A2UIMessageEvent
    from .incoming import (
        A2UIIncomingAction,
        A2UIIncomingError,
        A2UIIncomingParseResult,
        parse_incoming_message,
    )
    from .middleware import A2UIValidationMiddleware
    from .parser import A2UIParseResult, A2UIResponseParser, A2UIValidationResult
    from .schema_manager import A2UISchemaManager
    from .serialize import to_jsonl
except ImportError as e:
    A2UIAction = missing_optional_dependency("A2UIAction", "a2ui", e)  # type: ignore[misc]
    A2UIAgent = missing_optional_dependency("A2UIAgent", "a2ui", e)  # type: ignore[misc]
    A2UIMessageEvent = missing_optional_dependency("A2UIMessageEvent", "a2ui", e)  # type: ignore[misc]
    A2UIIncomingAction = missing_optional_dependency("A2UIIncomingAction", "a2ui", e)  # type: ignore[misc]
    A2UIIncomingError = missing_optional_dependency("A2UIIncomingError", "a2ui", e)  # type: ignore[misc]
    A2UIIncomingParseResult = missing_optional_dependency("A2UIIncomingParseResult", "a2ui", e)  # type: ignore[misc]
    parse_incoming_message = missing_optional_dependency("parse_incoming_message", "a2ui", e)  # type: ignore[misc]
    A2UIValidationMiddleware = missing_optional_dependency("A2UIValidationMiddleware", "a2ui", e)  # type: ignore[misc]
    A2UIParseResult = missing_optional_dependency("A2UIParseResult", "a2ui", e)  # type: ignore[misc]
    A2UIResponseParser = missing_optional_dependency("A2UIResponseParser", "a2ui", e)  # type: ignore[misc]
    A2UIValidationResult = missing_optional_dependency("A2UIValidationResult", "a2ui", e)  # type: ignore[misc]
    A2UISchemaManager = missing_optional_dependency("A2UISchemaManager", "a2ui", e)  # type: ignore[misc]
    to_jsonl = missing_optional_dependency("to_jsonl", "a2ui", e)  # type: ignore[misc]

__all__ = (
    "A2UIAction",
    "A2UIAgent",
    "A2UIIncomingAction",
    "A2UIIncomingError",
    "A2UIIncomingParseResult",
    "A2UIMessageEvent",
    "A2UIParseResult",
    "A2UIResponseParser",
    "A2UISchemaManager",
    "A2UIValidationMiddleware",
    "A2UIValidationResult",
    "parse_incoming_message",
    "to_jsonl",
)
