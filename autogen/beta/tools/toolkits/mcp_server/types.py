# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

from autogen.beta.annotations import Variable


@dataclass
class MCPServerConfig:
    """
    Configuration for a single MCP server.
    It's important to specify AUTH headers as most MCP servers force auth nowadays.
    """

    server_url: str | Variable
    server_label: str | Variable = ""
    authorization_token: str | Variable | None = None
    description: str | Variable | None = None
    allowed_tools: list[str] | Variable | None = None
    blocked_tools: list[str] | Variable | None = None
    headers: dict[str, str] | Variable | None = None
    connection_timeout: float = 30.0


class JsonRpcResponse(TypedDict):
    jsonrpc: str
    id: str
    result: NotRequired[dict[str, Any]]
    error: NotRequired[dict[str, Any]]


class RawMCPTool(TypedDict):
    name: str
    description: NotRequired[str]
    inputSchema: NotRequired[dict[str, Any]]


class JsonRpcRequest(TypedDict):
    id: str
    jsonrpc: str
    method: str
    params: dict[str, Any]


class JsonRpcNotification(TypedDict):
    jsonrpc: str
    method: str
