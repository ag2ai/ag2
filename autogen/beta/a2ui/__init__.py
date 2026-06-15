# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_optional_dependency

try:
    from .action_tool import A2UIActionTool, a2ui_action
    from .actions import A2UIAction
    from .agent import A2UIAgent
    from .capabilities import (
        A2UI_CLIENT_CAPABILITIES_DEPENDENCY_KEY,
        A2UIClientCapabilities,
        A2UIClientDataModel,
        capabilities_to_prompt,
        parse_client_capabilities,
        parse_client_data_model,
    )
    from .events import A2UIMessageEvent
    from .incoming import (
        A2UIIncomingAction,
        A2UIIncomingError,
        A2UIIncomingFunctionResponse,
        A2UIIncomingParseResult,
        action_to_prompt,
        error_to_prompt,
        function_response_to_prompt,
        parse_incoming_message,
        sanitize_for_prompt,
    )
    from .middleware import A2UIValidationMiddleware
    from .parser import A2UIParseResult, A2UIResponseParser, A2UIValidationResult
    from .schema_manager import A2UISchemaManager
    from .serialize import to_jsonl
except ImportError as e:
    A2UIAction = missing_optional_dependency("A2UIAction", "a2ui", e)  # type: ignore[misc]
    A2UIActionTool = missing_optional_dependency("A2UIActionTool", "a2ui", e)  # type: ignore[misc]
    a2ui_action = missing_optional_dependency("a2ui_action", "a2ui", e)  # type: ignore[misc]
    A2UIAgent = missing_optional_dependency("A2UIAgent", "a2ui", e)  # type: ignore[misc]
    A2UI_CLIENT_CAPABILITIES_DEPENDENCY_KEY = missing_optional_dependency(  # type: ignore[misc]
        "A2UI_CLIENT_CAPABILITIES_DEPENDENCY_KEY", "a2ui", e
    )
    A2UIClientCapabilities = missing_optional_dependency("A2UIClientCapabilities", "a2ui", e)  # type: ignore[misc]
    A2UIClientDataModel = missing_optional_dependency("A2UIClientDataModel", "a2ui", e)  # type: ignore[misc]
    capabilities_to_prompt = missing_optional_dependency("capabilities_to_prompt", "a2ui", e)  # type: ignore[misc]
    parse_client_capabilities = missing_optional_dependency("parse_client_capabilities", "a2ui", e)  # type: ignore[misc]
    parse_client_data_model = missing_optional_dependency("parse_client_data_model", "a2ui", e)  # type: ignore[misc]
    A2UIMessageEvent = missing_optional_dependency("A2UIMessageEvent", "a2ui", e)  # type: ignore[misc]
    A2UIIncomingAction = missing_optional_dependency("A2UIIncomingAction", "a2ui", e)  # type: ignore[misc]
    A2UIIncomingError = missing_optional_dependency("A2UIIncomingError", "a2ui", e)  # type: ignore[misc]
    A2UIIncomingFunctionResponse = missing_optional_dependency("A2UIIncomingFunctionResponse", "a2ui", e)  # type: ignore[misc]
    A2UIIncomingParseResult = missing_optional_dependency("A2UIIncomingParseResult", "a2ui", e)  # type: ignore[misc]
    action_to_prompt = missing_optional_dependency("action_to_prompt", "a2ui", e)  # type: ignore[misc]
    error_to_prompt = missing_optional_dependency("error_to_prompt", "a2ui", e)  # type: ignore[misc]
    function_response_to_prompt = missing_optional_dependency("function_response_to_prompt", "a2ui", e)  # type: ignore[misc]
    parse_incoming_message = missing_optional_dependency("parse_incoming_message", "a2ui", e)  # type: ignore[misc]
    sanitize_for_prompt = missing_optional_dependency("sanitize_for_prompt", "a2ui", e)  # type: ignore[misc]
    A2UIValidationMiddleware = missing_optional_dependency("A2UIValidationMiddleware", "a2ui", e)  # type: ignore[misc]
    A2UIParseResult = missing_optional_dependency("A2UIParseResult", "a2ui", e)  # type: ignore[misc]
    A2UIResponseParser = missing_optional_dependency("A2UIResponseParser", "a2ui", e)  # type: ignore[misc]
    A2UIValidationResult = missing_optional_dependency("A2UIValidationResult", "a2ui", e)  # type: ignore[misc]
    A2UISchemaManager = missing_optional_dependency("A2UISchemaManager", "a2ui", e)  # type: ignore[misc]
    to_jsonl = missing_optional_dependency("to_jsonl", "a2ui", e)  # type: ignore[misc]

__all__ = (
    "A2UI_CLIENT_CAPABILITIES_DEPENDENCY_KEY",
    "A2UIAction",
    "A2UIActionTool",
    "A2UIAgent",
    "A2UIClientCapabilities",
    "A2UIClientDataModel",
    "A2UIIncomingAction",
    "A2UIIncomingError",
    "A2UIIncomingFunctionResponse",
    "A2UIIncomingParseResult",
    "A2UIMessageEvent",
    "A2UIParseResult",
    "A2UIResponseParser",
    "A2UISchemaManager",
    "A2UIValidationMiddleware",
    "A2UIValidationResult",
    "a2ui_action",
    "action_to_prompt",
    "capabilities_to_prompt",
    "error_to_prompt",
    "function_response_to_prompt",
    "parse_client_capabilities",
    "parse_client_data_model",
    "parse_incoming_message",
    "sanitize_for_prompt",
    "to_jsonl",
)
