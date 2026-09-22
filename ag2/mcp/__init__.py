# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag2.exceptions import missing_optional_dependency

# The fallback rebinds a name mypy has bound to a class, which it rejects; it
# sees only the real import. See website/docs/contributor-guide/type-checking.mdx.
if TYPE_CHECKING:
    from mcp.server.mcpserver import Elicit, ListRoots, Resolve, Sample
    from mcp.server.request_state import RequestStateSecurity
    from mcp.server.streamable_http import EventStore
    from mcp.server.transport_security import TransportSecuritySettings

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
    from .transport import TransportConfig
else:
    try:
        # Curated MCP SDK re-exports: the names AG2's own deterministic-tool examples
        # type. The rest of the SDK is not mirrored — import wire models and less
        # common types from ``mcp`` directly.
        from mcp.server.mcpserver import Elicit, ListRoots, Resolve, Sample
        from mcp.server.request_state import RequestStateSecurity
        from mcp.server.streamable_http import EventStore
        from mcp.server.transport_security import TransportSecuritySettings

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
        from .transport import TransportConfig
    except ImportError as e:  # pragma: no cover - exercised only when ag2[mcp] is absent
        MCPServer = missing_optional_dependency("MCPServer", "mcp", e)
        build_ask_tool = missing_optional_dependency("build_ask_tool", "mcp", e)
        AskContext = missing_optional_dependency("AskContext", "mcp", e)
        ContextProvider = missing_optional_dependency("ContextProvider", "mcp", e)
        SessionConfig = missing_optional_dependency("SessionConfig", "mcp", e)
        TransportConfig = missing_optional_dependency("TransportConfig", "mcp", e)
        TransportSecuritySettings = missing_optional_dependency("TransportSecuritySettings", "mcp", e)
        EventStore = missing_optional_dependency("EventStore", "mcp", e)
        Resource = missing_optional_dependency("Resource", "mcp", e)
        ResourceTemplate = missing_optional_dependency("ResourceTemplate", "mcp", e)
        Prompt = missing_optional_dependency("Prompt", "mcp", e)
        PromptArgument = missing_optional_dependency("PromptArgument", "mcp", e)
        PromptMessage = missing_optional_dependency("PromptMessage", "mcp", e)
        MCPFunctionTool = missing_optional_dependency("MCPFunctionTool", "mcp", e)
        mcp_tool = missing_optional_dependency("mcp_tool", "mcp", e)
        Elicit = missing_optional_dependency("Elicit", "mcp", e)
        ListRoots = missing_optional_dependency("ListRoots", "mcp", e)
        Resolve = missing_optional_dependency("Resolve", "mcp", e)
        Sample = missing_optional_dependency("Sample", "mcp", e)
        RequestStateSecurity = missing_optional_dependency("RequestStateSecurity", "mcp", e)
        client_extension = missing_optional_dependency("client_extension", "mcp", e)
        ExtensionMap = missing_optional_dependency("ExtensionMap", "mcp", e)
        AppSandbox = missing_optional_dependency("AppSandbox", "mcp", e)
        MCPApp = missing_optional_dependency("MCPApp", "mcp", e)
        client_supports_apps = missing_optional_dependency("client_supports_apps", "mcp", e)
        MCPRequestContext = missing_optional_dependency("MCPRequestContext", "mcp", e)
        ResourceCsp = missing_optional_dependency("ResourceCsp", "mcp", e)
        ResourcePermissions = missing_optional_dependency("ResourcePermissions", "mcp", e)
        Visibility = missing_optional_dependency("Visibility", "mcp", e)
        AppContent = missing_optional_dependency("AppContent", "mcp", e)
        AppText = missing_optional_dependency("AppText", "mcp", e)

__all__ = (
    "AppContent",
    "AppSandbox",
    "AppText",
    "AskContext",
    "ContextProvider",
    "Elicit",
    "EventStore",
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
    "TransportConfig",
    "TransportSecuritySettings",
    "Visibility",
    "build_ask_tool",
    "client_extension",
    "client_supports_apps",
    "mcp_tool",
)
