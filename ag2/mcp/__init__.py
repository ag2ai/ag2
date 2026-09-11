# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.exceptions import missing_optional_dependency

try:
    # Curated MCP SDK re-exports: the names AG2's own deterministic-tool examples
    # type. The rest of the SDK is not mirrored — import wire models and less
    # common types from ``mcp`` directly.
    from mcp.server.mcpserver import Elicit, ListRoots, Resolve, Sample
    from mcp.server.request_state import RequestStateSecurity

    from .apps import (
        AppContent,
        AppSandbox,
        AppText,
        MCPApp,
        ResourceCsp,
        ResourcePermissions,
        Visibility,
        client_supports_apps,
    )
    from .executor import AskContext, ContextProvider
    from .extensions import ExtensionMap, client_extension
    from .info import build_ask_tool
    from .prompts import Prompt, PromptArgument, PromptMessage
    from .resources import Resource, ResourceTemplate
    from .server import MCPServer
    from .sessions import SessionConfig
    from .tools import MCPFunctionTool, MCPRequestContext, mcp_tool
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
    MCPFunctionTool = missing_optional_dependency("MCPFunctionTool", "mcp", e)  # type: ignore[misc]
    mcp_tool = missing_optional_dependency("mcp_tool", "mcp", e)  # type: ignore[misc]
    Elicit = missing_optional_dependency("Elicit", "mcp", e)  # type: ignore[misc]
    ListRoots = missing_optional_dependency("ListRoots", "mcp", e)  # type: ignore[misc]
    Resolve = missing_optional_dependency("Resolve", "mcp", e)  # type: ignore[misc]
    Sample = missing_optional_dependency("Sample", "mcp", e)  # type: ignore[misc]
    RequestStateSecurity = missing_optional_dependency("RequestStateSecurity", "mcp", e)  # type: ignore[misc]
    client_extension = missing_optional_dependency("client_extension", "mcp", e)  # type: ignore[misc]
    ExtensionMap = missing_optional_dependency("ExtensionMap", "mcp", e)  # type: ignore[misc]
    AppSandbox = missing_optional_dependency("AppSandbox", "mcp", e)  # type: ignore[misc]
    MCPApp = missing_optional_dependency("MCPApp", "mcp", e)  # type: ignore[misc]
    client_supports_apps = missing_optional_dependency("client_supports_apps", "mcp", e)  # type: ignore[misc]
    MCPRequestContext = missing_optional_dependency("MCPRequestContext", "mcp", e)  # type: ignore[misc]
    ResourceCsp = missing_optional_dependency("ResourceCsp", "mcp", e)  # type: ignore[misc]
    ResourcePermissions = missing_optional_dependency("ResourcePermissions", "mcp", e)  # type: ignore[misc]
    Visibility = missing_optional_dependency("Visibility", "mcp", e)  # type: ignore[misc]
    AppContent = missing_optional_dependency("AppContent", "mcp", e)  # type: ignore[misc]
    AppText = missing_optional_dependency("AppText", "mcp", e)  # type: ignore[misc]

__all__ = (
    "AppContent",
    "AppSandbox",
    "AppText",
    "AskContext",
    "ContextProvider",
    "Elicit",
    "ExtensionMap",
    "ListRoots",
    "MCPApp",
    "MCPFunctionTool",
    "MCPRequestContext",
    "MCPServer",
    "Prompt",
    "PromptArgument",
    "PromptMessage",
    "RequestStateSecurity",
    "Resolve",
    "Resource",
    "ResourceCsp",
    "ResourcePermissions",
    "ResourceTemplate",
    "Sample",
    "SessionConfig",
    "Visibility",
    "build_ask_tool",
    "client_extension",
    "client_supports_apps",
    "mcp_tool",
)
