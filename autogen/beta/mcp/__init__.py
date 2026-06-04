# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_optional_dependency

try:
    from .info import build_ask_tool, build_server_info
    from .server import MCPServer
except ImportError as e:
    MCPServer = missing_optional_dependency("MCPServer", "mcp", e)  # type: ignore[misc]
    build_ask_tool = missing_optional_dependency("build_ask_tool", "mcp", e)  # type: ignore[misc]
    build_server_info = missing_optional_dependency("build_server_info", "mcp", e)  # type: ignore[misc]

__all__ = (
    "MCPServer",
    "build_ask_tool",
    "build_server_info",
)
