# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_optional_dependency

try:
    from .executor import AskContext, ContextProvider
    from .info import build_ask_tool, build_server_info
    from .security import AuthorizationServerMetadata, proxy_authorization_server
    from .server import MCPServer
except ImportError as e:  # pragma: no cover - exercised only when ag2[mcp] is absent
    MCPServer = missing_optional_dependency("MCPServer", "mcp", e)  # type: ignore[misc]
    build_ask_tool = missing_optional_dependency("build_ask_tool", "mcp", e)  # type: ignore[misc]
    build_server_info = missing_optional_dependency("build_server_info", "mcp", e)  # type: ignore[misc]
    AskContext = missing_optional_dependency("AskContext", "mcp", e)  # type: ignore[misc]
    ContextProvider = missing_optional_dependency("ContextProvider", "mcp", e)  # type: ignore[misc]
    AuthorizationServerMetadata = missing_optional_dependency("AuthorizationServerMetadata", "mcp", e)  # type: ignore[misc]
    proxy_authorization_server = missing_optional_dependency("proxy_authorization_server", "mcp", e)  # type: ignore[misc]

__all__ = (
    "AskContext",
    "AuthorizationServerMetadata",
    "ContextProvider",
    "MCPServer",
    "build_ask_tool",
    "build_server_info",
    "proxy_authorization_server",
)
