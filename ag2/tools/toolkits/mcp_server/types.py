# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ag2.annotations import Variable

# Which protocol era a connection settles on: ``"legacy"`` performs the
# ``initialize`` handshake, ``"auto"`` probes ``server/discover`` first.
ProtocolMode = Literal["legacy", "auto"]


@dataclass(kw_only=True)
class MCPServerConfig:
    """Reach a remote MCP server over HTTP (streamable-http).

    Most public servers require authentication, so set ``authorization_token`` or
    ``headers`` unless you know the server is open.
    """

    server_url: str | Variable
    """Where the server listens, as a full URL including the MCP endpoint path."""

    authorization_token: str | Variable | None = None
    """Bearer token sent on every request to the server."""

    headers: dict[str, str] | Variable | None = None
    """Extra HTTP headers sent on every request, for anything a bearer token cannot carry."""

    connection_timeout: float = 30.0
    """How long, in seconds, to wait on the server before giving up on a request."""

    proxy: str | None = None
    """HTTP proxy to route the connection through."""

    verify: bool = True
    """Whether to verify the server's TLS certificate. Turn it off only against a server you run."""

    protocol_mode: ProtocolMode = "legacy"
    """Which protocol era to speak.

    ``"legacy"`` (the default) keeps the era released code already spoke: the
    ``initialize`` handshake, and nothing else on connect. ``"auto"`` probes
    ``server/discover`` first and falls back to the handshake when the server does
    not answer it. Only the modern era can carry a server's request for input back
    as a tool call's result, so a server that asks needs ``"auto"``.
    """

    server_label: str | Variable = ""
    """Name the toolkit reports itself under, used to namespace its tools."""

    description: str | Variable | None = None
    """What this server is for, shown wherever the toolkit is described."""

    allowed_tools: list[str] | Variable | None = None
    """Server tool names to expose; with none set every tool the server lists is exposed."""

    blocked_tools: list[str] | Variable | None = None
    """Server tool names to hide, applied after ``allowed_tools``."""

    tool_name_prefix: str | Variable = ""
    """Prefix put in front of the agent-visible tool names, to keep two servers' generic names apart.

    The server never sees it: ``allowed_tools``, ``blocked_tools`` and the outbound
    call all use the server's own names.
    """


@dataclass(kw_only=True)
class MCPStdioServerConfig:
    """Launch a local MCP server as a subprocess and speak MCP over its stdio pipes.

    Use this for MCP servers shipped as CLIs — ``npx -y @some/mcp-server``,
    ``uvx some-mcp-server``, or a script in your own project.
    """

    command: str | Variable
    """The executable to launch."""

    args: list[str] | Variable = field(default_factory=list)
    """Arguments passed to the executable."""

    env: dict[str, str] | Variable | None = None
    """Environment variables for the subprocess; with none set it inherits this process's."""

    cwd: str | Path | Variable | None = None
    """Working directory the subprocess is launched in."""

    encoding: str = "utf-8"
    """Text encoding of the subprocess's stdio pipes."""

    protocol_mode: ProtocolMode = "legacy"
    """Which protocol era to speak.

    ``"legacy"`` (the default) keeps the era released code already spoke: the
    ``initialize`` handshake, and nothing else on connect. ``"auto"`` probes
    ``server/discover`` first and falls back to the handshake when the server does
    not answer it. Only the modern era can carry a server's request for input back
    as a tool call's result, so a server that asks needs ``"auto"``.
    """

    server_label: str | Variable = ""
    """Name the toolkit reports itself under, used to namespace its tools."""

    description: str | Variable | None = None
    """What this server is for, shown wherever the toolkit is described."""

    allowed_tools: list[str] | Variable | None = None
    """Server tool names to expose; with none set every tool the server lists is exposed."""

    blocked_tools: list[str] | Variable | None = None
    """Server tool names to hide, applied after ``allowed_tools``."""

    tool_name_prefix: str | Variable = ""
    """Prefix put in front of the agent-visible tool names, to keep two servers' generic names apart.

    The server never sees it: ``allowed_tools``, ``blocked_tools`` and the outbound
    call all use the server's own names.
    """
