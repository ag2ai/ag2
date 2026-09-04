# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ag2.annotations import Variable

# How the connection settles which protocol revision it speaks. ``"legacy"`` (the
# default) performs the ``initialize`` handshake; ``"auto"`` probes
# ``server/discover`` first, so a modern-era server is met on the modern era —
# where a question comes back as the *result* of a call rather than as a
# standalone request. Which one happens is then the revision's doing, not a
# second setting.
ProtocolMode = Literal["legacy", "auto"]


@dataclass
class MCPServerConfig:
    """
    Configuration for a remote (HTTP / streamable-http) MCP server.
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
    proxy: str | None = None
    verify: bool = True
    protocol_mode: ProtocolMode = "legacy"


@dataclass
class MCPStdioServerConfig:
    """
    Configuration for a local MCP server that communicates over stdin/stdout.

    The server is launched as a subprocess (``command`` + ``args``) and the
    MCP protocol is spoken across its stdio pipes. Use this for locally
    installed MCP servers shipped as CLIs (e.g. ``npx -y @some/mcp-server``,
    ``uvx some-mcp-server``, or a script in your project).
    """

    command: str | Variable
    args: list[str] | Variable = field(default_factory=list)
    env: dict[str, str] | Variable | None = None
    cwd: str | Path | Variable | None = None
    server_label: str | Variable = ""
    description: str | Variable | None = None
    allowed_tools: list[str] | Variable | None = None
    blocked_tools: list[str] | Variable | None = None
    encoding: str = "utf-8"
    protocol_mode: ProtocolMode = "legacy"
