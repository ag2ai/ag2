# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.exceptions import missing_optional_dependency

try:
    from .executor import AskContext, ContextProvider
    from .info import build_ask_tool
    from .prompts import Prompt, PromptArgument, PromptMessage
    from .resources import Resource, ResourceTemplate
    from .server import MCPServer
    from .sessions import SessionConfig
except ImportError as e:  # pragma: no cover - exercised only when ag2[mcp] is absent
    MCPServer = missing_optional_dependency("MCPServer", "mcp", e)  # type: ignore[misc]
    build_ask_tool = missing_optional_dependency("build_ask_tool", "mcp", e)  # type: ignore[misc]
    AskContext = missing_optional_dependency("AskContext", "mcp", e)  # type: ignore[misc]
    ContextProvider = missing_optional_dependency("ContextProvider", "mcp", e)  # type: ignore[misc]
    SessionConfig = missing_optional_dependency("SessionConfig", "mcp", e)  # type: ignore[misc]
    Resource = missing_optional_dependency("Resource", "mcp", e)  # type: ignore[misc]
    ResourceTemplate = missing_optional_dependency("ResourceTemplate", "mcp", e)  # type: ignore[misc]
    Prompt = missing_optional_dependency("Prompt", "mcp", e)  # type: ignore[misc]
    PromptArgument = missing_optional_dependency("PromptArgument", "mcp", e)  # type: ignore[misc]
    PromptMessage = missing_optional_dependency("PromptMessage", "mcp", e)  # type: ignore[misc]

__all__ = (
    "AskContext",
    "ContextProvider",
    "MCPServer",
    "Prompt",
    "PromptArgument",
    "PromptMessage",
    "Resource",
    "ResourceTemplate",
    "SessionConfig",
    "build_ask_tool",
)
