# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.exceptions import missing_additional_dependency, missing_optional_dependency

try:
    from .agent import ACPAgent, PromptContent
    from .auth import AuthProvider, AuthenticationFailedError, StaticTokenAuth
    from .config import ACPConfig, ClaudeCodeConfig, CodexConfig, KiloCodeConfig, OpenCodeConfig
    from .sessions import SessionConfig
    from .tool_gateway import MCPCapabilityError
    from .transport import ACPTransportError
except ImportError as e:  # pragma: no cover - exercised only when ag2[acp] is absent
    ACPConfig = missing_optional_dependency("ACPConfig", "acp", e)  # type: ignore[misc]
    ACPTransportError = missing_optional_dependency("ACPTransportError", "acp", e)  # type: ignore[misc]
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

try:
    from .remote import ACPRemoteConfig
except ImportError as e:  # pragma: no cover - exercised only when agent-client-protocol[http] is absent
    ACPRemoteConfig = missing_additional_dependency("ACPRemoteConfig", "agent-client-protocol[http]", e)  # type: ignore[misc]

__all__ = [
    "ACPAgent",
    "ACPConfig",
    "ACPRemoteConfig",
    "ACPTransportError",
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
