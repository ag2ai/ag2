# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Both halves of AG2's Agent Client Protocol support.

**Consume** — drive external CLI coding agents (Claude Code, Codex, …) from an
AG2 agent. AG2 plays the ACP *Client* role; each CLI agent runs as an ACP *Agent*
subprocess. The integration is a :class:`ModelConfig` + :class:`LLMClient` pair —
no changes to the :class:`~ag2.Agent` class.

**Serve** — expose an AG2 agent *as* an ACP Agent, so any ACP Client can drive
it. That is :class:`ACPAgent`: the same job :class:`ag2.mcp.MCPServer` and
:class:`ag2.a2a.A2AServer` do for their protocols, named the way ACP names the
role (there is no "server" in ACP — its two roles are Client and Agent).
"""

from ag2.exceptions import missing_optional_dependency

try:
    from .agent import ACPAgent, PromptContent
    from .auth import AuthProvider, AuthenticationFailedError, StaticTokenAuth
    from .config import ACPConfig, ClaudeCodeConfig, CodexConfig, KiloCodeConfig, OpenCodeConfig
    from .sessions import SessionConfig
    from .tool_gateway import MCPCapabilityError
except ImportError as e:  # pragma: no cover - exercised only when ag2[acp] is absent
    ACPConfig = missing_optional_dependency("ACPConfig", "acp", e)  # type: ignore[misc]
    ACPAgent = missing_optional_dependency("ACPAgent", "acp", e)  # type: ignore[misc]
    PromptContent = missing_optional_dependency("PromptContent", "acp", e)  # type: ignore[misc]
    AuthProvider = missing_optional_dependency("AuthProvider", "acp", e)  # type: ignore[misc]
    AuthenticationFailedError = missing_optional_dependency("AuthenticationFailedError", "acp", e)  # type: ignore[misc]
    ClaudeCodeConfig = missing_optional_dependency("ClaudeCodeConfig", "acp", e)  # type: ignore[misc]
    CodexConfig = missing_optional_dependency("CodexConfig", "acp", e)  # type: ignore[misc]
    KiloCodeConfig = missing_optional_dependency("KiloCodeConfig", "acp", e)  # type: ignore[misc]
    OpenCodeConfig = missing_optional_dependency("OpenCodeConfig", "acp", e)  # type: ignore[misc]
    MCPCapabilityError = missing_optional_dependency("MCPCapabilityError", "acp", e)  # type: ignore[misc]
    SessionConfig = missing_optional_dependency("SessionConfig", "acp", e)  # type: ignore[misc]
    StaticTokenAuth = missing_optional_dependency("StaticTokenAuth", "acp", e)  # type: ignore[misc]

__all__ = [
    "ACPAgent",
    "ACPConfig",
    "AuthProvider",
    "AuthenticationFailedError",
    "ClaudeCodeConfig",
    "CodexConfig",
    "KiloCodeConfig",
    "MCPCapabilityError",
    "OpenCodeConfig",
    "PromptContent",
    "SessionConfig",
    "StaticTokenAuth",
]
